import cv2
from deepface import DeepFace as dp
import numpy as np
import logging

logger = logging.getLogger(__name__)

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
        logger.warning("embed_face: no face detected")
        return None
    else:
        logger.info("embed_face: face detected, embedding extracted")
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
        logger.warning("compare_embeddings: one of the embeddings is None")
        return False
    similarity = cosine_similarity(embedding1, embedding2)
    distance = 1 - similarity
    logger.info(f"Cosine similarity: {similarity:.4f}")
    logger.info(f"Cosine distance: {distance:.4f}")

    # Using cosine distance
    if similarity > 0.6:
        logger.info( "compare_embeddings: embeddings match")
        return True
    else:
        logger.info( "compare_embeddings: embeddings do not match")
        return False


def visualization_wrapper (frame, relative_bounding_box, img_test, saved_cropped = False):
    cropped_face, x_center, y_center = extract_face(frame, relative_bounding_box)

    if cropped_face is None or cropped_face.shape[0] < 100 or cropped_face.shape[1] < 100:
        logger.warning("visualization_wrapper: cropped face is None or too small")
        return
