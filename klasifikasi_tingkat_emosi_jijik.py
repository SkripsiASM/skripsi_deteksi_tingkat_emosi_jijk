import pyaudio
import wave
import joblib
import librosa
import numpy as np
import frft
import PreEmphasis_Framing_Windowing as pfw
import mel_filter_banks
from numpy.fft import fftshift
from scipy.fftpack import dct

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

while True:
    user_input = input("Do you want to continue? (yes/no): ")
    if user_input.lower() in ["yes", "y"]:
        print("Continuing...")
        pa = pyaudio.PyAudio()

        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER
        )

        print('start recording')

        seconds = 3
        frames = []
        second_tracking = 0
        second_count = 0
        for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
            data = stream.read(FRAMES_PER_BUFFER)
            frames.append(data)
            second_tracking += 1
            if second_tracking == RATE / FRAMES_PER_BUFFER:
                second_count += 1
                second_tracking = 0
                print(f'Time Left: {seconds - second_count} seconds')

        stream.stop_stream()
        stream.close()
        pa.terminate()

        obj = wave.open('emosi jijik.wav', 'wb')
        obj.setnchannels(CHANNELS)
        obj.setsampwidth(pa.get_sample_size(FORMAT))
        obj.setframerate(RATE)
        obj.writeframes(b''.join(frames))
        obj.close()

        signal, sample_rate = librosa.load("emosi jijik.wav")
        sig_filt = pfw.preemp_fra_win(signal, sample_rate)
        frft_signal = frft.frft(signal, 2)
        signal_cal = fftshift(frft_signal)
        filter_banks = mel_filter_banks.mel_fil_ban(signal_cal, sample_rate)
        num_ceps = 12
        frfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13
        cep_lifter = 22
        (nframes, ncoeff) = frfcc.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        frfcc *= lift

        loaded_rf = joblib.load("./random_forest.joblib")
        result = loaded_rf.predict(frfcc)
        print(result)

    elif user_input.lower() in ["no", "n"]:
        print("Exiting...")
        break
    else:
        print("Invalid input. Please enter yes/no.")