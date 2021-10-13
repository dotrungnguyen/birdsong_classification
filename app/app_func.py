#import database_func as db
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from PIL import Image
import matplotlib.pyplot as plt

# pip install keras (on env-anaconda)
from keras.models import load_model
import tensorflow as tf
import librosa
import librosa.display

# chọn file và thực hiện phân loại file đó
def classification():
    classname_list = classnames_info()
    red_list = conservation_status()
    model = reload_model()
    uploaded_file = None
    dataset_path = 'F:\\birds_test\\'

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if i == 0:
            selec = st.sidebar.selectbox('choose file', filenames)
            uploaded_file = os.path.join(dirpath,selec)
                
            if uploaded_file:
                audio_file = open(uploaded_file, 'rb')
                audio_bytes = audio_file.read()
                st.sidebar.audio(audio_bytes)

                list_mfcc, signal, sr = convert_wav_to_mfcc(uploaded_file,num_mfcc=13, n_fft=2048, hop_length=512, fmin = 0)
                kq = prediction_list(model,list_mfcc)

                birds_inf = classname_list[(classname_list['id'] == kq)]
                classnames = birds_inf['classname'].to_string(index = False)
                conservation = birds_inf['conservation status'].to_string(index = False)

                show_img(classnames)

                #with st.form(key='my_form'):
                with st.beta_expander("show spectrogram"):
                    show_spectrogram(signal, sr)

                with st.beta_expander("show info"):
                    st.write(birds_inf[birds_inf.columns.difference(['classname','id','conservation status'])])

                with st.beta_expander("conservation status"):
                    conservation_bird = red_list[red_list['categories'] == conservation]
                    # st.write(conservation_bird)
                    st.write('status: ',conservation)
                    st.write('symbol: ',conservation_bird['symbol'].to_string(index = False))
                    st.write('describe: ',conservation_bird['describe'].to_string(index = False))

#lưu file model vào cache, lần đầu load chận, lần sau nhanh do đã được lưu
# 2.1
@st.cache(allow_output_mutation=True)
def reload_model():
    with st.spinner('load model ...'):
        MODEL_PATH = './model_final_6s.h5'
        loaded_model = load_model(MODEL_PATH)
        return loaded_model



def convert_wav_to_mfcc(sound_file,num_mfcc=13, n_fft=2048, hop_length=512, fmin = 0):

    with st.spinner('sound processing ...'):

        list_mfccs = []

        sample_rate = 22050
        duration_segment = 6
        samples_per_segment = duration_segment * sample_rate

        try:
            signal, sr = librosa.load(sound_file)
        except:
            st.text('tệp âm thanh quá tải ... xin thử tệp khác!!!')
            return
        duration = librosa.get_duration(signal)
        num_segment = int(duration // duration_segment)

        for n in range(num_segment):
            start = samples_per_segment * n
            finish = start + samples_per_segment

            mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length ,fmin = 0)
            mfcc = mfcc.T

            list_mfccs.append(mfcc.tolist())

        return np.array(list_mfccs)[..., np.newaxis], signal, sr

def show_spectrogram(signal, sr):
    with st.spinner('spectrogram create ...'):
        fig, ax = plt.subplots(nrows = 2)
        fig.set_figheight(3)
        fig.patch.set_visible(False)
        ax[0].axis('off')
        ax[1].axis('off')

        librosa.display.waveplot(signal, sr=sr,color='pink', ax=ax[0])
        D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='linear', x_axis='time',sr=sr,fmin =300 ,fmax = 800, ax=ax[1])

        st.pyplot(fig)



# phaan boo dia ly


# một vài hàm con
def prediction(model,X_pred):
    with st.spinner('prediction ...'):
        X_pred = X_pred[np.newaxis, ...] 
        predictions = model.predict(X_pred)
        score = tf.nn.softmax(predictions[0])
        id = np.argmax(score)
        return id

def prediction_list(model,list_mfcc):
    a_list = []
    for m in list_mfcc:
        a_list.append(prediction(model,m))

    return most_common_in_list(a_list)

def most_common_in_list(a_list):
    most_common = max(a_list, key = a_list.count)
    return most_common

@st.cache(allow_output_mutation=True)
def classnames_info():
    classname_path = './classnames_info.csv'
    classnames_info = pd.read_csv(classname_path)
    return classnames_info

@st.cache(allow_output_mutation=True)
def conservation_status():
    red_list_path = './IUCN_red_list.csv'
    red_list = pd.read_csv(red_list_path)
    return red_list
    

def show_img(name):
    #classnames_info = pd.read_csv('./classnames_info.csv', index_col='id')
    image = Image.open(f'./image/{name}.jpg')
    st.sidebar.image(image, caption=name,width = 220)

def listen(p):
    audio_file = open(p, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)