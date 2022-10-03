import math
import os
import random
from numpy.random import seed
from sklearn.model_selection import train_test_split

seed(20)
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
import glob
import librosa
from pygame import mixer
import numpy as np
import pandas as pd
from tensorflow import keras

class_label = {"music": 1, "speech": 0}
pred_level = {1: "music", 0: "speech"}

FRAME_SIZE = 2048
HOP_SIZE = 512

#######GUI Builder#######
def build_ui(files):
    root = tk.Tk()
    root.title('Audio Analysis Tool')
    root.geometry("600x700")
    root.grid_columnconfigure(0, weight=1, uniform="group1")
    root.grid_columnconfigure(1, weight=1, uniform="group1")
    root.grid_rowconfigure(0, weight=1)
    mixer.init()

    # Create Picture chooser frame.
    listFrame = Frame(root, width=200, height=200)
    listFrame.pack(side=LEFT)
    # Layout Picture Listbox.
    l3 = Label(listFrame, text="Select audio from the list below:")
    l3.pack(side=TOP)

    # Create Control frame.
    controlFrame = Frame(root, width=200, height=200)
    controlFrame.pack(side=LEFT)

    # Create Preview frame.
    previewFrame = Frame(root,
                         width=200, height=200)
    previewFrame.pack_propagate(0)
    previewFrame.pack(side=LEFT)

    l1 = Label(previewFrame, text="Ground Truth")
    l1.grid(row=0, column=0, sticky=E)
    l2 = Label(previewFrame, text="Prediction")
    l2.grid(row=1, column=0, sticky=E)
    listScrollbar = Scrollbar(listFrame)
    listScrollbar.pack(side=RIGHT, fill=Y)
    # creating a list box for list of all image names and add it to the scroll pane
    list = Listbox(listFrame,
                   yscrollcommand=listScrollbar.set,
                   selectmode=BROWSE,
                   height=40, width=20)
    for i in range(len(files)):
        list.insert(i, os.path.basename(files[i]))

    list.pack(side=RIGHT, fill=BOTH, padx=5)
    list.config(background="white", foreground="black")
    list.activate(1)

    listScrollbar.config(command=list.yview)

    # Layout Controls.
    b0 = ttk.Button(controlFrame, text="Play",
                    width=10,
                    command=lambda: play(
                        list.get(ACTIVE)))
    b0.grid(row=0, column=0, sticky=E)

    b1 = ttk.Button(controlFrame, text="Stop",
                    width=10,
                    command=lambda: stop())
    b1.grid(row=0, column=1, sticky=E)

    def predict():
        # ind = 0
        val = list.curselection()
        if val == None:
            ind = 0
        else:
            ind = val[0]
        v = results[ind]
        c = Y_truthValues[ind]
        l3 = Label(previewFrame, text=pred_level[c])
        l3.grid(row=0, column=1, sticky=E)
        l4 = Label(previewFrame, text=pred_level[v])
        l4.grid(row=1, column=1, sticky=E)

    b2 = ttk.Button(controlFrame, text="Predict",
                    width=10,
                    command=lambda: predict())
    b2.grid(row=0, column=2, sticky=E)
    root.mainloop()

#####Helper functions for GUI#####
def play(file):
    audiofile = "audio\\speech\\" + file
    if not (os.path.exists(audiofile)):
        audiofile = "audio\\music\\" + file
    mixer.music.load(audiofile)
    mixer.music.play()

#####Helper functions for GUI#####
def stop():
    mixer.music.stop()

#####Helper functions for GUI#####
def add_song_to_home(c, r, master, audio):
    cell = ttk.Button(master, name=f'{c}cell{r}', text="Play", command=lambda: play(audio))
    cell.place(x=c, y=r, width=10, height=10)

    text = tk.Label(master, name=f'{c}text{r}', text=audio)
    text.place(x=c, y=(r + 1), width=10, height=10)

#######Helper function to help with feature extraction#######
def spectral_centroid(data, samplerate=44100):
    magnitudes = np.abs(np.fft.rfft(data))  # magnitudes of positive frequencies
    length = len(data)
    freqs = np.abs(np.fft.fftfreq(length, 1.0 / samplerate)[:length // 2 + 1])  # positive frequencies
    return np.sum(magnitudes * freqs) / np.sum(magnitudes)  # return weighted mean

#######Helper function to help with feature extraction#######
def rms(data, sr):
    fft = np.fft.fft(data)

    magnitude = np.abs(fft)
    frequency = np.linspace(0, sr, len(magnitude))

    rms = 0

    for each in magnitude:
        rms += each ** 2

    rms = rms / len(magnitude)
    rms = math.sqrt(rms)
    return rms

#######Helper function to help with feature extraction#######
def calc_zero_crossing(data):
    # magnitudes = np.abs(np.fft.rfft(data))
    # zero_crosses = np.nonzero(np.diff(magnitudes > 0))[0]

    rate = 0
    for i in range(1, len(data)):
        if not (data[i] and data[i - 1]):
            rate += 1

    rate = rate / (2 * (len(data) - 1))
    return rate

#######Helper function to help with feature extraction#######
def calc_split_frequency_bin(spectrogram, split_frequency, sample_rate):
    frequency_range = sample_rate / 2
    frequency_delta_per_bin = frequency_range / spectrogram.shape[0]
    split_frequency_bin = np.floor(split_frequency / frequency_delta_per_bin)
    return int(split_frequency_bin)

#######Helper function to help with feature extraction#######
def calculate_band_energy_ratio(spectrogram, split_frequency, sample_rate):
    split_frequency_bin = calc_split_frequency_bin(spectrogram, split_frequency, sample_rate)

    power_spec = np.abs(spectrogram) ** 2
    power_spec = power_spec.T

    band_energy_ratio = []

    for frequencies_in_frame in power_spec:
        sum_power_low_frequencies = np.sum(frequencies_in_frame[:split_frequency_bin])
        sum_power_high_frequencies = np.sum(frequencies_in_frame[split_frequency_bin:])
        ber_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
        band_energy_ratio.append(ber_current_frame)

    return np.array(band_energy_ratio)


################Returns an array###############################
####The below function is used to extract features from the audio files##########
def extract_features(file_name):
    data, sr = librosa.load(file)
    zero_cross = librosa.zero_crossings(data)
    z_rate = calc_zero_crossing(zero_cross)

    feature = np.array(z_rate)

    spectral_centr = spectral_centroid(data, samplerate=sr)
    feature = np.append(feature, spectral_centr)
    # print(spectral_centroids)

    STE = rms(data, sr)
    RMS = rms(data, sr)
    feature = np.append(feature, STE)
    feature = np.append(feature, RMS)

    chroma_stft = librosa.feature.chroma_stft(y=data, sr=sr)
    feature = np.append(feature, np.mean(chroma_stft))

    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    feature = np.append(feature, np.mean(mfcc.T, axis=0))

    return feature.tolist()

####This function loads our model back into the application######
def load_model():
    modelPath = "MusicSpeechClassifierNewModel70.h5"
    new_model = keras.models.load_model(modelPath)

    X_prediction = featuresdf.iloc[:, 0:24].values
    predictions = new_model.predict(X_prediction)

    classes = np.argmax(predictions, axis=1)

    ####Prints the accuracy of the model####
    acc = new_model.evaluate(X_prediction, Y_truthValues)


    ###Used to manually calculate Precision and Recall######
    ####Uncomment if needed#####
    ####Outputs 0 0 or 1 1 are correct predictions####
    # for i in range(0, len(predictions)):
    #     print(np.argmax(predictions[i]), Y_truthValues[i])

    return classes


if __name__ == '__main__':

    audioFiles = []

    for infile in glob.glob('audio/music/*.wav'):
        audioFiles.append(infile)
    for infile in glob.glob('audio/speech/*.wav'):
        audioFiles.append(infile)

    random.shuffle(audioFiles)
    # extract features
    features = []
    for file in audioFiles:
        # print(file)
        typ = file.split("/")[1].split("\\")[0]
        dat = extract_features(file)

        features.append([dat, class_label[typ]])

    featureList = []
    for i in features:
        m = i[0]
        c = []
        for fs in m:
            c.append(fs)
        c.append(i[1])
        featureList.append(c)
    print(len(featureList[0]))
    cols = []
    # making each extracted features in a features list with columns, and last column is the class label
    for i in range(len(featureList[0]) - 1):
        cols.append("f" + str(i))

    cols.append('class_label')
    featuresdf = pd.DataFrame(featureList, columns=cols)

    Y_truthValues = featuresdf.iloc[:, 25].values

    ######Uncomment this if you would like to test the model manually######
    # X = featuresdf.iloc[:, 0:24].values
    # X_train, X_test, y_train, y_test = train_test_split(X, Y_truthValues, test_size=0.3)

    ############Commented the ML section down below to make sure we are not constructing the neural network and fitting it all over again############
    # model = Sequential()
    # model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(2, activation='softmax'))
    # model.compile(optimizer='adam',
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    #
    # classifier = model.fit(X_train,
    #                        Y_train,
    #                        epochs=100,
    #                        batch_size=128)
    #
    #
    # #model.save("MusicSpeech.h5")
    ##########################################################

    # predict the classification for the files using a pretrained model
    results = load_model()

    # bring up the UI to display the results
    build_ui(audioFiles)