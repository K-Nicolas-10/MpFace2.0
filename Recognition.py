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

def get_distance(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        logger.warning("get_distance: one of the embeddings is None")
        return float('inf')
    distance = 1 - cosine_similarity(embedding1, embedding2)
    logger.info(f"get_distance: Cosine distance: {distance:.4f}")
    return distance

def detect_and_extract_face_from_image(full_image, detector_backend='mtcnn', target_size=(224, 224)):
    if full_image is None:
        logger.warning("detect_and_extract_face_from_image: full_image is None")
        return None
    try:
        extracted_res = dp.extract_faces(
            img_path=full_image,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True,
        )

        print("extracted_res: ", extracted_res)

        high_confidence_faces = [
            face_data for face_data in extracted_res
            if face_data['confidence'] >= 0.8
        ]

        if len(high_confidence_faces) == 0:
            logger.warning("detect_and_extract_face_from_image: no face detected")
            return None
        elif len(high_confidence_faces) > 1:
            # @TODO: We could try to get the largest face (?)
            logger.warning("detect_and_extract_face_from_image: more than one face detected")
            return None
        else:
            logger.info("detect_and_extract_face_from_image: one face detected")
            face_data = high_confidence_faces[0]
            face_img_raw = face_data["face"]

            if face_img_raw.dtype == np.float32 or face_img_raw.dtype == np.float64:
                 face_img_bgr = cv2.cvtColor((face_img_raw * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            elif len(face_img_raw.shape) == 3 and face_img_raw.shape[2] == 3:
                 face_img_bgr = cv2.cvtColor(face_img_raw, cv2.COLOR_RGB2BGR)
            else:
                 face_img_bgr = face_img_raw
            
            if target_size:
                face_img_resized = cv2.resize(face_img_bgr, target_size, interpolation=cv2.INTER_AREA)
                return face_img_resized
            else:
                return face_img_bgr
    except Exception as e:
        logger.error(f"detect_and_extract_face_from_image: error in face detection: {e}")
        return None

