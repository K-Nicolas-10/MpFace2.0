import cv2
import threading
import mediapipe as mp
import Tracking
import Recognition
from pathlib import Path

from Recognition import compare_embeddings

folder_path_str = 'Students'
folder = Path(folder_path_str)

#Creating dictionary for Name:Embedding
#Every photo in the folder must contain the student's name in the name of the file
#Must be PNG or JPG as to not complicate our existence
embedded_db = []
if folder.is_dir():
    for item in folder.iterdir():
        temp = cv2.imread(str(item))
        embedded_db.append({
            # slicing the .png extension and only assigning the name
            f"{item.name[:-4]}" : Recognition.embed_face(temp)
        })

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Initialize video capture
cap = cv2.VideoCapture(0)

# Frame skipping
frame_count = 0
frame_skip = 2 # Process every (frame_skip + 1)-th frame


#initializing FaceTracker
face_tracker = Tracking.FaceTracker()
#initializing Recognition
face_recognition = Recognition

with mp_face_detection.FaceDetection(
                                    model_selection=1,
                                    min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read camera input.")
            break

        if frame_count % (frame_skip + 1) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)
            current_detections = []

            if results.detections:
                current_detections = list(results.detections)
            face_tracker.update_tracks(current_detections, frame.shape)

            #drawing the annotations on the frame
            #displaying number of tracked faces
            num_tracked_faces= face_tracker.get_tracked_faces_count()
            tracked_faces_text = f"Tracked faces: { num_tracked_faces }"
            cv2.putText(frame, tracked_faces_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2 )

            #Face Recognition part:
            if frame_count % 5 == 0: # every 30 frames run face recognition
                tracked_faces_for_recognition = face_tracker.get_all_tracked_faces_data()
                for face_data in tracked_faces_for_recognition:
                    face_id = face_data["id"]
                    current_name = face_data["name"]
                    relative_bbox = face_data["bbox_relative"]
                    if current_name is None and relative_bbox:
                        extracted_face = face_recognition.extract_face(frame, relative_bbox)
                        live_embedding = face_recognition.embed_face(extracted_face)
                        if live_embedding is not None:
                            for student in embedded_db:
                                for db_name, db_embedding in student.items():
                                    if face_recognition.compare_embeddings(live_embedding, db_embedding):
                                        face_tracker.update_face_name_by_id(face_id, db_name)


            face_tracker.draw_annotations(frame)
            cv2.imshow("Face Detect", frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

# Release capture
cap.release()
cv2.destroyAllWindows()