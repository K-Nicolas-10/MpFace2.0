import cv2
import threading
import mediapipe as mp
import Tracking
import Recognition
from pathlib import Path
from gui import AppGui
from PyQt5.QtWidgets import QApplication
import sys
import numpy as np

from Recognition import compare_embeddings

from db import Session, Student, Embed, get_student_by_id

embedded_db = {}

with Session() as session:
    students = session.query(Student).all()
    for student in students:
        embedded_db[student.name] = [
            np.frombuffer(embed.embedding, dtype=np.float32)
            for embed in student.embeds
        ]

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

# initializing gui
app = QApplication(sys.argv)
app_gui = AppGui()
app_gui.show()

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
            if frame_count % 5 == 0: # every 5 frames run face recognition
                tracked_faces_for_recognition = face_tracker.get_all_tracked_faces_data()
                for face_data in tracked_faces_for_recognition:
                    face_id = face_data["id"]
                    current_name = face_data["name"]
                    relative_bbox = face_data["bbox_relative"]
                    if current_name is None and relative_bbox:
                        extracted_face = face_recognition.extract_face(frame, relative_bbox)
                        live_embedding = face_recognition.embed_face(extracted_face)
                        if live_embedding is not None:
                            recognized = False
                            for db_name, db_embeddings in embedded_db.items():
                                for db_embedding in db_embeddings:
                                    if face_recognition.compare_embeddings(live_embedding, db_embedding):
                                        face_tracker.update_face_name_by_id(face_id, db_name)
                                        recognized = True
                                        app_gui.add_student(db_name)
                                        break
                            if not recognized:
                                res = app_gui.prompt_for_info(cv2.cvtColor(extracted_face, cv2.COLOR_BGR2RGB))
                                if res:
                                    if res["new"]:
                                        with Session() as session:
                                            new_student = Student(name=res["name"], group=res["group"])
                                            embed = Embed(student=new_student, embedding=np.array(live_embedding, dtype=np.float32).tobytes())
                                            session.add(new_student)
                                            session.add(embed)
                                            session.commit()
                                            embedded_db[res["name"]] = [live_embedding]
                                            face_tracker.update_face_name_by_id(face_id, res["name"])
                                            app_gui.add_student(new_student.name)
                                    else:
                                        with Session() as session:
                                            student = res["student"]
                                            if student not in embedded_db:
                                                embed = Embed(student=student, embedding=np.array(live_embedding, dtype=np.float32).tobytes())
                                                session.add(embed)
                                                session.commit()
                                                embedded_db[student.name].append(live_embedding)
                                                face_tracker.update_face_name_by_id(face_id, student.name)
                                                app_gui.add_student(student.name)


            face_tracker.draw_annotations(frame)
            new_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            app_gui.update_frame(new_frame_rgb)

        frame_count += 1
        # update the gui
        app.processEvents()

# Release capture
cap.release()
cv2.destroyAllWindows()