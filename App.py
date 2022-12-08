import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
import av
from PIL import Image
from flowchart import *
from single_image_prediction import *
from live_streaming_prediction import *
from convert_to_audio import *
from pathlib import Path
from tempfile import NamedTemporaryFile
import moviepy
from moviepy.editor import VideoFileClip
import math
from datetime import timedelta
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    
def load_image(image_file):
    img = Image.open(image_file)
    return img



def streamlit_menu():
    st.set_page_config(layout="wide")
    col1, col2, col3 = st.columns((1,2,1))
    col1.image("https://w7.pngwing.com/pngs/215/960/png-transparent-deaf-culture-deaf-news-british-sign-language-british-deaf-association-united-kingdom-united-kingdom-hand-united-kingdom-electric-blue-thumbnail.png", width=100)

    col2.title("Sign Language Detector & Translator")
    cwd = os.getcwd()
    logoimg = load_image(os.path.join(cwd, "images/slb_logo.png"))
    col3.image(logoimg, width = 100)
    
    selected = option_menu(
        menu_title=None,
        options=["Home", "About us", "Solution Design", "Image data", "Video data", "Real time stream"],
        icons=["house", "file-person","code-slash", "images", "skip-end-btn", "hourglass" ],
        menu_icon="cast", 
        default_index=0, 
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "white", "font-size": "20px"},
            "nav-link": {
                "font-size": "20px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
                "border-radius": 0.5,
            },
            "nav-link-selected": {"background-color": "gray"},
        },
    )
    return selected



def upload_plot_images():

    data = st.file_uploader("Upload an image", type=["jpeg", "png"])

    col1, col2 = st.columns(2)
    image = None
    grayscale = None
    path = ""
    
    if data is not None:

        image = load_image(data)
        path = './images/temp.jpg'
        image.save(path, 'JPEG')

        col1.header ("Original image")
        col1.image(image,use_column_width=True)
        
        col2.header("Grayscale image")
        grayscale = image.convert('LA')
        col2.image(grayscale, use_column_width=True)
    

    return image, grayscale,path
    
    
def upload_video():
    data = st.file_uploader("Upload a video", type = ["mp4"])
    vf = None
    path = ""
    if data is not None:
        path = './videos/temp.mp4'
        with open(path,"wb") as f:
            f.write(data.getbuffer())
            
        #vf = cv2.VideoCapture(path)
        
        video_file = open(path, 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)
    
    return path


def convert_video_to_images(video_path, time_interval):
    #Time_to_Extract_Frame = 1 
    Time_to_Extract_Frame = time_interval

    clip = VideoFileClip(video_path)
    duration = math.floor(clip.duration)
    total_step = int(duration/Time_to_Extract_Frame)
    folder_name = "./Extracted_Frames/"

    for step in range(total_step):
        if step < 10:
            frame_filename = os.path.join(folder_name, f"0{step}.jpg")
        if step >=10:
            frame_filename = os.path.join(folder_name, f"{step}.jpg")
        print(frame_filename)
        clip.save_frame(frame_filename, step*Time_to_Extract_Frame)



selected = streamlit_menu()


if selected == "Home":
    st.markdown("\n \n \n")
    col1, col2 = st.columns((1, 3))
    col1.markdown("\n")
    col1.markdown("Sign languages are languages that use the visual-manual modality to convey meaning, instead of just spoken words. Sign languages are expressed through manual articulation in combination with non-manual markers. Sign languages are full-fledged natural languages with their own grammar and lexicon.")
    col1.markdown("\n \n \n")
    col1.markdown("Sign Language detector & translator is a webapp that takes any form of media images, video or real time stream as input, detects sign language in the medium and translates them to text. This is a ML based web based application that uses object detection, deep learning and transfer learning to successfully detect hand gestures and translate the signs.")
    cwd = os.getcwd()
    signlangimg = load_image(os.path.join(cwd, "images/sign_language.PNG"))
    col2.image(signlangimg, use_column_width = True)
    
    col3, col4 = st.columns((1, 3))
    col3.markdown("")
    detectorimg = load_image(os.path.join(cwd, "images/detector.PNG"))
    col2.image(detectorimg, use_column_width = True)
    
    col5, col6 = st.columns((1, 3))
    col5.markdown("")
    translatorimg = load_image(os.path.join(cwd, "images/translator.PNG"))
    col6.image(translatorimg, use_column_width = True)


if selected == "About us":
    cwd = os.getcwd()
    st.markdown("\n \n \n")
    col1, col2, col3 = st.columns(3)
    img1 = load_image(os.path.join(cwd, "images/Shi.PNG"))
    img1 = img1.resize((300,300))
    col1.image(img1)
    col1.header("Shi Su")
    col1.markdown("Senior II Reservoir Engineer")
    col1.markdown("Abu Dhabi, UAE")
    col1.markdown("https://www.eureka.slb.com/CNP.cfm?uid=shi-20121101")
    
    img2 = load_image(os.path.join(cwd, "images/Hengky.PNG"))
    img2 = img2.resize((300,300))
    col2.image(img2)
    col2.header("Hengky Ng")
    col2.markdown("Petrophysicist")
    col2.markdown("Jakarta, Indonesia")
    col2.markdown("https://www.eureka.slb.com/CNP.cfm?uid=hengky-20070603")
    
    img3 = load_image(os.path.join(cwd, "images/Sushant.PNG"))
    img3 = img3.resize((300,300))    
    col3.image(img3)
    col3.header("Sushant Shekhar")
    col3.markdown("Geologist")
    col3.markdown("Pune Data Hub, India")
    col3.markdown("https://www.eureka.slb.com/CNP.cfm?uid=sushant-20180709")
    
    


if selected == "Solution Design":
    col1, col2 = st.columns(2)
    graph1, graph2, graph3 = drawGraph()
    
    with col1:
        wf_exp1 = st.expander(label='Object detection workflow')
        with wf_exp1:
            st.header("Object detection workflow")
            st.graphviz_chart(graph1, use_container_width=True)
    #st.markdown("\n \n \n \n \n")
    with col2:
        wf_exp2 = st.expander(label='Hand landmark workflow')
        with wf_exp2:
            st.header("Hand landmark workflow")
            st.graphviz_chart(graph2, use_container_width=True)
    

    wf_exp3 = st.expander(label='Video/Live stream workflow')
    with wf_exp3:
        st.header("Video/Live stream workflow")
        st.graphviz_chart(graph3, use_container_width = True)


if selected == "Image data":
    st.header("Select the file to upload and test")
    img, grayimg, image_file_path = upload_plot_images()
    
    st.header("Select the model to apply")
    model_expander = st.expander(label='Select model(s)')

    path = Path(os.getcwd())
    parentpath = path.parent.absolute()
    models = ['Hand Landmark Detection : ml_mediapipe_hand_landmark_newdata.tflite', 'Object Detection : yolov5_small_trained_signlanguage','Object Detection : yolov5_medium_trained_signlanguage','Object Detection : yolov5_large_trained_signlanguage_v1', 'Object Detection : yolov5_medium_newdata']
    
    method = ""
    prediction_list = []
    model_list = []
    with model_expander:
        selected_option = st.multiselect("", models)
        clicked = st.button('Run Prediction')
    
    if clicked:
        st.header("\n Prediction results")
        cols = st.columns(len(selected_option))
        
        for selected in selected_option:
            method = selected.split(" : ")[0]
            model = selected.split(" : ")[1]

            prediction= predict_alphabet(image_file_path, method, model)
            #st.image(img)
            prediction_list.append(prediction)
            model_list.append(model)
        
        for i, x in enumerate(cols):
            x.markdown(model_list[i])
            x.image(img)
            x.header("Predicted Alphabet: " + str(prediction_list[i]))
            


if selected == "Video data":
    st.header("Select the video to upload and test")
    video_path = upload_video()
    
    if video_path != "":
        st.markdown("Select time interval between frames to capture images:")
        interval = st.slider("Time interval", 1, 10)
        convert_video_to_images(video_path, interval)
    
    frame_path = "./Extracted_Frames/"
    st.header("Select the model to apply")
    model_expander = st.expander(label='Select model(s)')

    path = Path(os.getcwd())
    parentpath = path.parent.absolute()
    models = ['Hand Landmark Detection : ml_mediapipe_hand_landmark_newdata.tflite', 'Object Detection : yolov5_small_trained_signlanguage','Object Detection : yolov5_medium_trained_signlanguage','Object Detection : yolov5_large_trained_signlanguage_v1', 'Object Detection : yolov5_medium_newdata']

    method = ""
    prediction_list = []
    model_list = []
    image_list = []

    with model_expander:
        selected_option = st.multiselect("", models)
        clicked = st.button('Run Prediction')
        
    if clicked:
        st.header("\n Prediction results")

        framefiles = [x for x in os.listdir(frame_path)]
        for selected in selected_option:
            method = selected.split(" : ")[0]
            model = selected.split(" : ")[1]

            for f in sorted(framefiles):

                image_file_path = os.path.join(frame_path, f)
                prediction= predict_alphabet(image_file_path, method, model)
                img = load_image(image_file_path)
                image_list.append(img)
                prediction_list.append(prediction)
                model_list.append(model)

        #cols = st.columns(len(image_list))
        nrows = int(len(image_list)/4)
        count = 0
        for i in range(nrows):
            if count < len(image_list)-1:
                cols = st.columns(4)
                for j, x in enumerate(cols):
                    x.markdown(model_list[count])
                    x.image(image_list[count])
                    x.header("Predicted Alphabet: " + str(prediction_list[count]))
                    count += 1


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


if selected == "Real time stream":

    st.header("Select the model to apply")
    models = ['Hand Landmark Detection : ml_mediapipe_hand_landmark_newdata.tflite', 'Object Detection : yolov5_small_trained_signlanguage','Object Detection : yolov5_medium_trained_signlanguage','Object Detection : yolov5_large_trained_signlanguage_v1', 'Object Detection : yolov5_medium_newdata']


    selected_option = st.multiselect("Select model(s)", models)

    st.title("Webcam Live Feed")
    #run = st.checkbox('Run')
    for selected in selected_option:
        method = selected.split(" : ")[0]
        model = selected.split(" : ")[1]
    letters = ['', '', '', '', '']
    time_start = ''
    time_start_no_hand = ''
    words = 'Output: '

    class VideoProcessor:
        def recv(self, frame):
            global time_start
            global time_start_no_hand
            global letters
            global words
            if time_start == '':
                time_start = datetime.datetime.now()
                time_start_no_hand = datetime.datetime.now()
            img = frame.to_ndarray(format="bgr24")
            img, letter = predict_alphabet_live_mod(img, method, model)

            print(letter)

            if letter!='':
                time_start_no_hand = datetime.datetime.now()
                time_later = datetime.datetime.now()
                deltatime = time_later - time_start
                deltatime_seconds = math.floor(deltatime.total_seconds())
                try:
                    letters[int(deltatime_seconds)] = letter
                except:
                    pass
                print(deltatime_seconds)
                if deltatime_seconds >= 2:
                    if letters[0] == letters[1] and letters[1] == letters[2]:
                        time_start = datetime.datetime.now()
                        words += letter
                        letters = ['', '', '', '', '']
                    else:
                        time_start = datetime.datetime.now()
                        letters = ['', '', '', '', '']
            else:
                # time counter to record 'space' character only if no hand on webcam screen
                time_start = datetime.datetime.now()
                time_later_no_hand = datetime.datetime.now()
                deltatime_no_hand = time_later_no_hand - time_start_no_hand
                deltatime_seconds_no_hand = math.floor(deltatime_no_hand.total_seconds())

                letters = ['', '', '', '', '']

                if deltatime_seconds_no_hand == 2:
                    if words[-1] != ' ':
                        words += ' '
            print(letters, words)
            cv2.putText(img=img, text=words, org=(30, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7,
            color=(0,0,0))

            cv2.imshow('Updated Frame', img)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="TEST",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,

    )


    #FRAME_WINDOW = st.image([])
    #camera = cv2.VideoCapture(0)
    #method = ""

#    words = ""
#    if clicked:
#        for selected in selected_option:
#            method = selected.split(" : ")[0]
#            model = selected.split(" : ")[1]
#
#            prediction= predict_alphabet_live(method, model)
#            st.header("Predicted Text:")
#            st.header(prediction)
#
#            if st.button('Convert to audio'):
#                convert_to_audio(prediction)

                
    #if st.button('Convert to audio'):
    #    audiopath = convert_to_audio(words)
    #    audio_file = open("./audio/temp.mp3", 'rb')
    #    audio_bytes = audio_file.read()
    #   st.audio(audio_bytes, format='audio/mp3')

    
