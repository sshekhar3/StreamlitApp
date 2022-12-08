from gtts import gTTS
import os
import streamlit as st


def convert_to_audio(text):
    st.markdown(text)
    audio = gTTS(text= text, lang="en", slow=False)
    path = "./audio/temp.mp3"
    audio.save(path)
    
    return path
