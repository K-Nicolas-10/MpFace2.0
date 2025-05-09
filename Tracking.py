import cv2
import time

class FaceAnnotation:
    #Represents a single tracked face and its associated annotation.
    #It stores the face's ID, current bounding box, and handles drawing the annotations
    def __init__(self, identifier, initial_detection_data, frame_shape):
        #identifier = unique ID for a specific tracked face
        #initial_detection_data (mediapipe.framework.formats.detection_pb2.Detection)
        #frame_shape = tuple containing (height, width, channels) at the time of initial detection
        self.id = identifier
        self.name = None
        self.text_to_display = f"ID: {self.id}"

        #Stores MediaPipe's relative_bounding_box
        self.current_bounding_box_relative = None
        #Stores relative_bounding_box in absolute coordinates
        self.current_bounding_box_abs = None

        self.last_seen_time = time.time()
        # stores the shape of the frame at the time of the last update to correctly calculate bbox
        self.frame_shape_at_last_update = frame_shape

        #initialize with the first detection data
        self.update_detection_data(initial_detection_data, frame_shape)

    def update_detection_data(self, detection_data, frame_shape):
        #Updates the face's data based on a new MediaPipe detection
        self.current_bounding_box_relative= detection_data.location_data.relative_bounding_box
        self.frame_shape_at_last_update = frame_shape
        self.update_absolute_bbox()
        self.last_seen_time = time.time()
    def update_absolute_bbox(self):
        if not self.current_bounding_box_relative or not self.frame_shape_at_last_update:
            self.current_bounding_box_abs = None
            return
        image_height, image_width, _ = self.frame_shape_at_last_update
        bb = self.current_bounding_box_relative
        #Converting relative coordinates to absolute coordinates
        xmin_abs = int(bb.xmin * image_width)
        ymin_abs = int(bb.ymin * image_height)
        width_abs = int(bb.width * image_width)
        height_abs = int(bb.height * image_height)

        self.current_bounding_box_abs = (xmin_abs, ymin_abs, xmin_abs + width_abs,ymin_abs + height_abs)

    def get_position_for_text(self):
        #position for drawing the text a little above the bounding box
        if not self.current_bounding_box_abs:
            return None
        xmin_abs, ymin_abs, _, _ = self.current_bounding_box_abs
        text_x = xmin_abs
        text_y = ymin_abs - 10
        #Boundary checks for text pos
        if text_y < 20:
            text_y = ymin_abs + 20
        if text_x < 0:
            text_x = 0
        return (text_x, text_y)

    def draw(self, frame):
        #Drawing the annotations on the frame
        #First we ensure the absolute bbox is up to date
        if self.current_bounding_box_abs is None and self.current_bounding_box_relative:
            self.update_absolute_bbox()
        if not self.current_bounding_box_abs:
            return
        xmin_abs, ymin_abs, xmax_abs, ymax_abs = self.current_bounding_box_abs

        cv2.rectangle(frame,(xmin_abs,ymin_abs),(xmax_abs,ymax_abs),(0,255,0),2)
        text_position = self.get_position_for_text()
        if text_position:
            cv2.putText(frame, self.text_to_display,text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    def set_recognized_name(self, recognized_name):
        self.name = recognized_name
        if self.name:
            self.text_to_display = f"Name: {self.name}"
        else:
            self.text_to_display = f"ID: {self.id}"

def calculate_iou(box1_rel_bbox, box2_rel_bbox):
    # Convert relative coordinates to (x1, y1, x2, y2) for easier calculation
    # These are still relative to image dimensions, but represent corners.
    b1_x1, b1_y1 = box1_rel_bbox.xmin, box1_rel_bbox.ymin
    b1_x2, b1_y2 = box1_rel_bbox.xmin + box1_rel_bbox.width, box1_rel_bbox.ymin + box1_rel_bbox.height

    b2_x1, b2_y1 = box2_rel_bbox.xmin, box2_rel_bbox.ymin
    b2_x2, b2_y2 = box2_rel_bbox.xmin + box2_rel_bbox.width, box2_rel_bbox.ymin + box2_rel_bbox.height

    # Determine the coordinates of the intersection rectangle
    xi1 = max(b1_x1, b2_x1)
    yi1 = max(b1_y1, b2_y1)
    xi2 = min(b1_x2, b2_x2)
    yi2 = min(b1_y2, b2_y2)

    # Calculate area of intersection
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Calculate area of both bounding boxes
    area1 = box1_rel_bbox.width * box1_rel_bbox.height
    area2 = box2_rel_bbox.width * box2_rel_bbox.height

    # Calculate union area
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0  # Avoid division by zero

    iou = inter_area / union_area
    return iou

class FaceTracker:
    #Manages multiple FaceAnnotation objects to track faces across frames
    #It uses IoU to match detections to existing tracks, assigns new IDs, and also removes tracks no longer seen
    def __init__(self, iou_threshold=0.3, last_track_timeout=1.0):
        #iou_threshold = minimum IoU to detect a match
        #lost_track_timeout = time in seconds after a track is considered lost if not re-detected
        self.tracked_faces = {} # FaceAnnotation objects keyed by unique ID
        self.next_face_id = 0 #Iterator for new ID's
        self.iou_threshold = iou_threshold
        self.last_track_timeout = last_track_timeout

    def update_tracks(self, current_mp_detections, frame_shape):
        current_time = time.time()
        #in case of no detections
        if not current_mp_detections:
            current_mp_detections = []
        matched_current_detection_indices = [False] * len(current_mp_detections) # initializes an array of the length and every element initialized with False

        #attempting to match existing tracked_faces with current_detections
        for face_id, face_obj in list(self.tracked_faces.items()):
            best_match_idx = -1
            max_iou_for_this_tracked_face = self.iou_threshold

            for i, mp_detection in enumerate(current_mp_detections):
                if not matched_current_detection_indices[i]:
                    #ensuring we have a bounding box to compare with\
                    if not matched_current_detection_indices[i]:
                        if face_obj.current_bounding_box_relative:
                            iou = calculate_iou(face_obj.current_bounding_box_relative, mp_detection.location_data.relative_bounding_box)

                        if iou > max_iou_for_this_tracked_face:
                            max_iou_for_this_tracked_face = iou
                            best_match_idx = i

            if best_match_idx != -1:
                #found a match: update the FaceAnnotation object
                face_obj.update_detection_data(current_mp_detections[best_match_idx], frame_shape)
                matched_current_detection_indices[best_match_idx] = True
        #Adding any new unmatched detections in tracked_faces
        for i, mp_detection in enumerate(current_mp_detections):
            if not matched_current_detection_indices[i]:
                new_id = self.next_face_id
                self.next_face_id += 1
                self.tracked_faces[new_id] = FaceAnnotation(identifier=new_id,
                                                            initial_detection_data=mp_detection,
                                                            frame_shape=frame_shape)
        #Removing tracks that have not been seen for a timeout period
        for face_id, face_obj in list(self.tracked_faces.items()):
            if current_time - face_obj.last_seen_time > self.last_track_timeout:
                del self.tracked_faces[face_id]

    def draw_annotations(self,frame):
        for face_obj in self.tracked_faces.values():
            face_obj.draw(frame)

    def get_tracked_faces_count(self):
        return len(self.tracked_faces)

    def get_face_annotation(self,face_id):
        return self.tracked_faces.get(face_id)

    def update_face_name_by_id(self, face_id, name):
        face_obj = self.get_face_annotation(face_id)
        if face_obj:
            face_obj.set_recognized_name(name)

    def get_all_tracked_faces_data(self):
        #returns a list of dictionaries each containing the data for the tracke face
        #will use for face recognition (deepface)
        #each dict will contain : {'id': face_id, 'bbox_relative' : face_obj.current_bounding_box_relative}
        faces_data = []
        for face_id, face_obj in self.tracked_faces.items():
            if face_obj.current_bounding_box_relative:
                faces_data.append({
                    'id':face_id,
                    'bbox_relative': face_obj.current_bounding_box_relative,
                    'name':face_obj.name # none if not recognized
                })
        return faces_data

