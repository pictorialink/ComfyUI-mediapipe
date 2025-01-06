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
        'face_landmarks': {
            
        }
    }

    face_landmarks_dict = {
            'left_brow': [70, 105, 107],
            'right_brow': [300, 334, 336],
            'right_eye': [159, 133, 145, 33],
            'left_eye': [386, 263, 374, 362],
            'left_ball': [473],
            'right_ball': [468],
            'mouth': [0, 269, 409, 405, 17, 181, 62, 39]
        }
    if hasattr(results, 'pose_world_landmarks'):
        if results.pose_world_landmarks is not None:
            for landmark in results.pose_world_landmarks.landmark:
                landmarks_dict['pose_world_landmarks'].append({
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
            left_brow = []
            for index in face_landmarks_dict["left_brow"]:
                value = results.face_landmarks.landmark[index]
                left_brow.append({
                    'x': value.x,
                    'y': value.y,
                    'z': value.z,
                })
            landmarks_dict['face_landmarks']['left_brow'] = left_brow

            right_brow = []
            for index in face_landmarks_dict["right_brow"]:
                value = results.face_landmarks.landmark[index]
                right_brow.append({
                    'x': value.x,
                    'y': value.y,
                    'z': value.z,
                })
            landmarks_dict['face_landmarks']['right_brow'] = right_brow

            left_eye = []
            for index in face_landmarks_dict["left_eye"]:
                value = results.face_landmarks.landmark[index]
                left_eye.append({
                    'x': value.x,
                    'y': value.y,
                    'z': value.z,
                })
            landmarks_dict['face_landmarks']['left_eye'] = left_eye

            right_eye = []
            for index in face_landmarks_dict["right_eye"]:
                value = results.face_landmarks.landmark[index]
                right_eye.append({
                    'x': value.x,
                    'y': value.y,
                    'z': value.z,
                })
            landmarks_dict['face_landmarks']['right_eye'] = right_eye

            left_ball = []
            for index in face_landmarks_dict["left_ball"]:
                value = results.face_landmarks.landmark[index]
                left_ball.append({
                    'x': value.x,
                    'y': value.y,
                    'z': value.z,
                })
            landmarks_dict['face_landmarks']['left_ball'] = left_ball

            right_ball = []
            for index in face_landmarks_dict["right_ball"]:
                value = results.face_landmarks.landmark[index]
                right_ball.append({
                    'x': value.x,
                    'y': value.y,
                    'z': value.z,
                })
            landmarks_dict['face_landmarks']['right_ball'] = right_ball

            mouth = []
            for index in face_landmarks_dict["mouth"]:  
                value = results.face_landmarks.landmark[index]
                mouth.append({
                    'x': value.x,
                    'y': value.y,
                    'z': value.z,
                })
            landmarks_dict['face_landmarks']['mouth'] = mouth
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

