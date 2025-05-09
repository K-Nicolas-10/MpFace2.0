import cv2
import threading
import mediapipe as mp
import Tracking

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Initialize video capture
cap = cv2.VideoCapture(0)

# Frame skipping
frame_count = 0
frame_skip = 2 # Process every (frame_skip + 1)-th frame

test_image = cv2.imread("img.png")

#initializing FaceTracker
face_tracker = Tracking.FaceTracker()


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
            face_tracker.draw_annotations(frame)

            #displaying number of tracked faces
            num_tracked_faces= face_tracker.get_tracked_faces_count()
            tracked_faces_text = f"Tracked faces: { num_tracked_faces }"
            cv2.putText(frame, tracked_faces_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2 )

            #Face Recognition part:


            cv2.imshow("Face Detect", frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

# Release capture
cap.release()
cv2.destroyAllWindows()