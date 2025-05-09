import cv2
from deepface import DeepFace as dp
import numpy as np
def extract_face(frame, relative_bounding_box):
    #extracting face from the bound box detected by MP
    frame_height, frame_width,_ = frame.shape
    xmin = int(relative_bounding_box.xmin * frame_width)
    ymin = int(relative_bounding_box.ymin * frame_height)
    width = int(relative_bounding_box.width * frame_width)
    height = int(relative_bounding_box.height * frame_height)

    # Ensure coordinates are within the frame boundaries
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(frame_width, xmin + width)
    ymax = min(frame_height, ymin + height)

    face = frame[ymin:ymax, xmin:xmax]
    face_resized = cv2.resize(face, (224, 224))
    bbox = relative_bounding_box
    y_center = bbox.ymin + bbox.height / 2
    x_center = bbox.xmin + bbox.width / 2
    return face_resized

def embed_face(face_image):
    embeddings = dp.represent(face_image, model_name = "Facenet", enforce_detection = False)
    if not embeddings:
        print ("embed_face: no face detected")
        return None
    else:
        print ("embed face: face detected, embedding extracted")
        embedding = embeddings[0]["embedding"]
        return embedding

def cosine_similarity (embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return float('inf')

    return dot_product / (norm1 * norm2)

def compare_embeddings(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        print(float('inf'))# Or some other indicator of no face/embedding
        return False
    similarity = cosine_similarity(embedding1, embedding2)
    distance = 1 - similarity
    print(f"Cosine similarity: {similarity:.4f}")
    print(f"Cosine distance: {distance:.4f}")

    # Using cosine distance
    if similarity > 0.6:

        print ("Recognised")
        return True
    else:
        print ("Not Recognised")
        return False


def visualization_wrapper (frame, relative_bounding_box, img_test, saved_cropped = False):
    cropped_face, x_center, y_center = extract_face(frame, relative_bounding_box)

    if cropped_face is None or cropped_face.shape[0] < 100 or cropped_face.shape[1] < 100:
        print ("Cropped face too small of invalid")
        return

    if saved_cropped:
        cv2.imwrite("last_cropped.png", cropped_face)

    identify_person(frame,cropped_face, img_test, 2, x_center, y_center, "victors")

def identify_person(frame, face_live, face_db, nrof_person, x_center, y_center, name):
    face_live_embedding = embed_face(face_live)
    face_db_embedding = embed_face(face_db)

    for i in range(nrof_person):
        if compare_embeddings(face_live_embedding, face_db_embedding):
            # Write name at (x_center, y_center)
            cv2.putText(frame, name, (int(x_center), int(y_center)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 255, 0), thickness=2)
            return