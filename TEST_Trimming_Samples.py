# ------------------------------------------  SOUND PROCESSING OF RECORDED XYLOPHONE KEYS
# --------------------- IMPORT LIBRARIES
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from glob import glob
import itertools
#from itertools import cycle

#from pydub import AudioSegment
#from pydub.playback import play
import joblib
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


# ----------------------------- FUNCTIONS
# Display Graph for Audio file
def plot_wave(file):
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.plot(file, 'g')
    plt.show()

# Create a Spetrogram
def spectrogram(note):
    note_ft = librosa.stft(note)
    note_db = librosa.amplitude_to_db(np.abs(note_ft), ref = np.max)

    img, ax = plt.subplots(figsize=(10,5))
    note_spectrogram = librosa.display.specshow(note_db, x_axis='time', y_axis='log', ax = ax)
    #print(A_ft)
    #print(A_db)
    #print(A_db.shape)
    ax.set_title('Spectogram', fontsize=20)
    img.colorbar(note_spectrogram, ax=ax, format=f'%0.2f')
    plt.show()

audio_files = glob('C:/Users/elena/Desktop/University Stuff/3rd Year/Final Major Project/Code/Keys_Recording/*.wav')

raw_audios = []
fourier_Db = []

for sample in audio_files:
    raw, sr = librosa.load(sample)
    raw_trim, _ = librosa.effects.trim(raw, top_db=20)
    raw_audios.append(raw_trim)
    plot_wave(raw_trim)
'''
for sample in audio_files:
    raw, sr = librosa.load(sample)
    raw_audios.append(raw)
'''
