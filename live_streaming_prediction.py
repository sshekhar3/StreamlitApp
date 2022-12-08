from utilities import *
import datetime
import math


def predict_alphabet_live(method, model):
    #methodology_available = ['Object Detection', 'Hand Landmark Detection']
    #selected_methodology = methodology_available[1]
    selected_methodology = method
    ml_models_path = './models'

    hand_landmark_model_file = 'ml_mediapipe_hand_landmark_newdata.tflite'
    #hand_landmark_model_file = 'ml_mediapipe_hand_landmark_newdata.tflite'
    hand_landmark_model = os.path.join(ml_models_path, hand_landmark_model_file)

    object_detection_models_available=[
        'yolov5_small_trained_signlanguage',
        'yolov5_medium_trained_signlanguage',
        'yolov5_large_trained_signlanguage_v1',
        'yolov5_medium_newdata',
    ]

    #object_detection_model_file = object_detection_models_available[0]
    #object_detection_model = os.path.join(ml_models_path, object_detection_model_file, r'weights\best.pt')

    padding = 0.1

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    capture = cv2.VideoCapture(0)
    full_text = ''

    change_background_mp = mp.solutions.selfie_segmentation
    change_bg_segment = change_background_mp.SelfieSegmentation()

    text_to_show_hand_1 = ''
    text_to_show_hand_2 = ''

    # initialize time
    time_start = datetime.datetime.now()
    time_start_no_hand = datetime.datetime.now()

    # initialize list of letter
    letters = ['', '', '', '', ''] # use to check if the first, second and third character is same

    # initialize empty words
    words = 'test: '
    count_image = 0

    color = (0, 0, 0)

    with mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                        max_num_hands=2) as hands:
        while True:
            ret, frame = capture.read()
            cv2.flip(frame, 0)
            frame_height, frame_width, _ = frame.shape
            size_square = min(frame_height, frame_width)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks is not None:
                hands_locations_x = []
                hands_locations_y = []
                for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    current_hand_x = [int(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x * frame_width) for i
                                      in range(21)]
                    current_hand_y = [int(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y * frame_height) for
                                      i in range(21)]
                    x_min = int(min(current_hand_x))
                    y_min = int(min(current_hand_y))
                    x_max = int(max(current_hand_x))
                    y_max = int(max(current_hand_y))
                    hand_image_width = int(x_max - x_min)
                    hand_image_height = int(y_max - y_min)

                    x_hand_avg = int(np.mean(current_hand_x))
                    y_hand_avg = int(np.mean(current_hand_y))

                    x = max(int((1 - padding) * x_min), 0)
                    y = max(int((1 - padding) * y_min), 0)
                    w = max(int(hand_image_width + 2 * (x_min - int((1 - padding) * x_min))),
                            int(hand_image_height + 2 * (y_min - int((1 - padding) * y_min))))
                    h = w

                    center_box_x = int((2 * x + w) / 2)
                    center_box_y = int((2 * y + h) / 2)

                    dx = center_box_x - x_hand_avg
                    dy = center_box_y - y_hand_avg
                    x = max(x - dx, 0)
                    y = max(y - dy, 0)
                    w += max(dx, dy)
                    h += max(dx, dy)


                    if selected_methodology == 'Hand Landmark Detection':
                        pre_processed_landmark_list = process_image_to_list(frame, static_mode=False)
                        letter = predict_from_hand_landmark(pre_processed_landmark_list, hand_landmark_model)
                    elif selected_methodology == 'Object Detection':
                        cropped_image = frame[0:size_square, int((frame_width-frame_height)/2):int((frame_width-frame_height)/2)+size_square]
                        object_detection_model_file = object_detection_models_available[object_detection_models_available.index(model)]
                        object_detection_model = os.path.join(ml_models_path, object_detection_model_file, r'weights/best.pt')
                        letter = predict_from_object_detection(cropped_image, object_detection_model, static_mode=False)
                        clean_runs_detect_folder()

                    # time counter to record letter only if same character is capture for 2 seconds
                    time_start_no_hand = datetime.datetime.now()
                    time_later = datetime.datetime.now()
                    deltatime = time_later - time_start
                    deltatime_seconds = math.floor(deltatime.total_seconds())
                    try:
                        letters[int(deltatime_seconds)] = letter
                    except:
                        pass

                    if deltatime_seconds >= 2:
                        if letters[0] == letters[1] and letters[1] == letters[2]:
                            time_start = datetime.datetime.now()
                            words += letters[2]
                            letters = ['', '', '', '', '']
                        else:
                            time_start = datetime.datetime.now()
                            letters = ['', '', '', '', '']
                    #print(words)

                    # if hand_no==0:
                    #    text_to_show_hand_1+=letter
                    # elif hand_no==1:
                    #    text_to_show_hand_2+=letter

                    cv2.putText(img=frame, text=letter, org=(10, 15), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7,
                                color=color)
            else:
                # time counter to record 'space' character only if no hand on webcam screen
                time_start = datetime.datetime.now()
                time_later_no_hand = datetime.datetime.now()
                deltatime_no_hand = time_later_no_hand - time_start_no_hand
                deltatime_seconds_no_hand = math.floor(deltatime_no_hand.total_seconds())

                letters = ['', '', '', '', '']

                if deltatime_seconds_no_hand == 3:
                    if words[-1] != ' ':
                        words += ' '

            cv2.putText(img=frame, text=words, org=(30,30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=color)
            # cv2.putText(img=frame, text=text_to_show_hand_2, org=(30,60), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=color)
            #        cv2.rectangle(frame, (int((frame_width-frame_height)/2), 0), (int((frame_width-frame_height)/2)+size_square, size_square), color, 2)
            cv2.imshow('Updated Frame', frame)

            if cv2.waitKey(1) == 27:
                break
            elif cv2.waitKey(1) == 8:
                if len(words) > 0:
                    words = words[:-1]
            elif cv2.waitKey(1) == 32:
                count_image+=1
                cropped_image = frame[0:size_square,
                                int((frame_width - frame_height) / 2):int((frame_width - frame_height) / 2) + size_square]
                cv2.imwrite(rf'new_training_data\img_{count_image}.jpg', cropped_image)
                print(rf'Wrote file in new_training_data\img_{count_image}.jpg')


    cv2.destroyAllWindows()
    capture.release()
    
    return words



def predict_alphabet_live_mod(frame, method, model):
    selected_methodology = method
    ml_models_path = './models'

    hand_landmark_model_file = 'ml_mediapipe_hand_landmark_newdata.tflite'
    hand_landmark_model = os.path.join(ml_models_path, hand_landmark_model_file)
    letter = ''
    object_detection_models_available = [
        'yolov5_small_trained_signlanguage',
        'yolov5_medium_trained_signlanguage',
        'yolov5_large_trained_signlanguage_v1',
        'yolov5_medium_newdata',
    ]

    padding = 0.1

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # initialize time


    # initialize list of letter
    #  # use to check if the first, second and third character is same
    # initialize empty words

    count_image = 0

    color = (0, 0, 0)

    with mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                        max_num_hands=2) as hands:
        frame_height, frame_width, _ = frame.shape
        size_square = min(frame_height, frame_width)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks is not None:
            hands_locations_x = []
            hands_locations_y = []
            for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                current_hand_x = [int(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x * frame_width) for i
                                  in range(21)]
                current_hand_y = [int(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y * frame_height) for
                                  i in range(21)]
                x_min = int(min(current_hand_x))
                y_min = int(min(current_hand_y))
                x_max = int(max(current_hand_x))
                y_max = int(max(current_hand_y))
                hand_image_width = int(x_max - x_min)
                hand_image_height = int(y_max - y_min)

                x_hand_avg = int(np.mean(current_hand_x))
                y_hand_avg = int(np.mean(current_hand_y))

                x = max(int((1 - padding) * x_min), 0)
                y = max(int((1 - padding) * y_min), 0)
                w = max(int(hand_image_width + 2 * (x_min - int((1 - padding) * x_min))),
                        int(hand_image_height + 2 * (y_min - int((1 - padding) * y_min))))
                h = w

                center_box_x = int((2 * x + w) / 2)
                center_box_y = int((2 * y + h) / 2)

                dx = center_box_x - x_hand_avg
                dy = center_box_y - y_hand_avg
                x = max(x - dx, 0)
                y = max(y - dy, 0)
                w += max(dx, dy)
                h += max(dx, dy)

                if selected_methodology == 'Hand Landmark Detection':
                    pre_processed_landmark_list = process_image_to_list(frame, static_mode=False)
                    letter = predict_from_hand_landmark(pre_processed_landmark_list, hand_landmark_model)
                elif selected_methodology == 'Object Detection':
                    cropped_image = frame[0:size_square, int((frame_width - frame_height) / 2):int(
                        (frame_width - frame_height) / 2) + size_square]
                    object_detection_model_file = object_detection_models_available[
                        object_detection_models_available.index(model)]
                    object_detection_model = os.path.join(ml_models_path, object_detection_model_file,
                                                          r'weights/best.pt')
                    letter = predict_from_object_detection(cropped_image, object_detection_model, static_mode=False)
                    clean_runs_detect_folder()


                cv2.putText(img=frame, text=letter, org=(10, 15), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7,
                            color=color)


    return frame, letter