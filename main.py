import numpy as np
import matplotlib.pyplot as plt
import librosa
import PreEmphasis_Framing_Windowing as pfw
import frft
import mel_filter_banks
import os
import pandas as pd
import joblib
from numpy.fft import fftshift
from scipy.fftpack import dct

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

nSnapshots = 11
alpha = np.linspace(0., 2., nSnapshots)

def signaltonoise_dB(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))
def extract_features(signal, sample_rate):
    result = np.array(signal)
    frft_signal = frft.frft(signal, 2)
    signal_cal = fftshift(frft_signal)
    filter_banks = mel_filter_banks.mel_fil_ban(signal_cal, sample_rate)
    num_ceps = 12
    frfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13
    cep_lifter = 22
    (nframes, ncoeff) = frfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    frfcc *= lift  # *
    result = np.hstack((result, frfcc))  # stacking horizontally

    return result

def get_features(path):
    signal, sample_rate = librosa.load(path)
    sig_filt = pfw.preemp_fra_win(signal, sample_rate)
    res1 = extract_features(sig_filt, sample_rate)
    result = np.array(res1)

    return result

Crema = "E:/SkripsiProject/CREMA_D/"
crema_directory_list = os.listdir(Crema)
file_emotion = []
file_path = []
for file in crema_directory_list:
    # storing file paths
    file_path.append(Crema + file)
    # storing file emotions
    part = file.split('_')
    if part[3] == 'HI.wav':
        file_emotion.append('high')
    elif part[3] == 'LO.wav':
        file_emotion.append('low')
    elif part[3] == 'MD.wav':
        file_emotion.append('medium')
    else:
        file_emotion.append('Unknown')

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Crema_df = pd.concat([emotion_df, path_df], axis=1)
print(Crema_df.head())

Crema_df.to_csv("data_path.csv",index=False)

X, Y = [], []
for path, emotion in zip(Crema_df.Path, Crema_df.Emotions):
    print(path)
    feature = get_features(path)
    signal, sample_rate = librosa.load(path)
    print(signal.shape)
    snr = signaltonoise_dB(signal)
    snr_frfcc = signaltonoise_dB(feature)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40)
    snr_mfccs = signaltonoise_dB(mfccs)
    print(snr)
    print(snr_frfcc)
    print(snr_mfccs)
    print("---")
    for ele in feature:
        X.append(ele)
        Y.append(emotion)

len(X), len(Y), Crema_df.Path.shape

Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv('features.csv', index=False)
print(Features.head())

X = Features.iloc[: ,:-1].values
Y = Features['labels'].values

print(X), print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
model = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
model.fit(x_train, y_train)
preds = model.predict(x_test)
print('Accuracy:', (accuracy_score(y_test, preds)))

joblib.dump(model, "./random_forest.joblib")

df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = preds.flatten()
df['Actual Labels'] = y_test.flatten()

print(df.head(20))
