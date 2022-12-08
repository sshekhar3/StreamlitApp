from utilities import *
import os


def predict_alphabet(image_file_path, method, model):
    #image_path = r'data\lexset\Test_Alphabet\E'
    #image_file = '0b5de7ca-1772-4a42-9d0c-ac942967debf.rgb_0000.png'
    #image_file_path = os.path.join(image_path, image_file)

    #methodology_available = ['Object Detection', 'Hand Landmark Detection']

    #selected_methodology = methodology_available[1]
    selected_methodology = method

    ml_models_path = './models'

    hand_landmark_model_file = 'ml_mediapipe_hand_landmark_newdata.tflite'
    hand_landmark_model = os.path.join(ml_models_path, hand_landmark_model_file)

    object_detection_models_available=[
        'yolov5_small_trained_signlanguage',
        'yolov5_medium_trained_signlanguage',
        'yolov5_large_trained_signlanguage_v1',
        'yolov5_medium_newdata',
    ]

    if selected_methodology == 'Hand Landmark Detection':
        pre_processed_landmark_list = process_image_to_list(image_file_path)
        pred_result = predict_from_hand_landmark(pre_processed_landmark_list, hand_landmark_model)
    elif selected_methodology == 'Object Detection':
        object_detection_model_file = object_detection_models_available[object_detection_models_available.index(model)]
        object_detection_model = os.path.join(ml_models_path, object_detection_model_file, r'weights/best.pt')
        pred_result = predict_from_object_detection(image_file_path, object_detection_model)

    return pred_result