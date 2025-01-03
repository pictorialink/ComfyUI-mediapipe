import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic



def save_json(results):
    
    
    landmarks_dict = {
        'pose_world_landmarks': [],
        'pose_landmarks': [],
        'left_hand_landmarks': [],
        'right_hand_landmarks': [],
        'face_landmarks': []
    }
    for landmark in results.pose_world_landmarks.landmark:
        landmarks_dict['pose_world_landmarks'].append({
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z,
            'visibility': landmark.visibility,
            'presence': landmark.presence
        })
    for landmark in results.pose_landmarks.landmark:
        landmarks_dict['pose_landmarks'].append({
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z,
            'visibility': landmark.visibility,
            'presence': landmark.presence
        })
    if hasattr(results, 'left_hand_landmarks'):
        if results.left_hand_landmarks is not None:
            for landmark in results.left_hand_landmarks.landmark:
                landmarks_dict['left_hand_landmarks'].append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
            })
    if hasattr(results, 'right_hand_landmarks'):
        if results.right_hand_landmarks is not None:
            for landmark in results.right_hand_landmarks.landmark:
                landmarks_dict['right_hand_landmarks'].append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                })
    if hasattr(results, 'face_landmarks'):
        if results.face_landmarks is not None:
            for landmark in results.face_landmarks.landmark:
                landmarks_dict['face_landmarks'].append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                })

    

    return landmarks_dict


def holistic_static(image):
    BG_COLOR = (192, 192, 192) # gray
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True) as holistic:
        image_height, image_width, _  = image.shape
        results = holistic.process(image)
        landmarks_dict = save_json(results)
        landmarks_dict["canvas_height"] = image_height
        landmarks_dict["canvas_width"] = image_width

        annotated_image = image.copy()
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)
        annotated_image = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8)

        mp_drawing.draw_landmarks(
            annotated_image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.
            get_default_pose_landmarks_style())
        mp_drawing.draw_landmarks(
            annotated_image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.
            get_default_hand_landmarks_style())
        mp_drawing.draw_landmarks(
            annotated_image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.
            get_default_hand_landmarks_style())
        
        return annotated_image, landmarks_dict

