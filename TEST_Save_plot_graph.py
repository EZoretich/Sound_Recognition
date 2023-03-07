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
def plot_wave(file, name):
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title(name)
    plt.plot(file, '0.8')
    #plt.savefig(name)
    plt.show()

# Create a Spetrogram
def spectrogram(note, name):
    note_ft = librosa.stft(note)
    note_db = librosa.amplitude_to_db(np.abs(note_ft), ref = np.max)

    img, ax = plt.subplots(figsize=(10,5))
    note_spectrogram = librosa.display.specshow(note_db, x_axis='time', y_axis='log', ax = ax)
    #print(A_ft)
    #print(A_db)
    #print(A_db.shape)
    ax.set_title(name, fontsize=20)
    img.colorbar(note_spectrogram, ax=ax, format=f'%0.2f')
    plt.show()

#sns.set_theme(style='white', palette=None)
#color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

audio_files = glob('C:/Users/elena/Desktop/University Stuff/3rd Year/Final Major Project/Code/Keys_Recording/*.wav')

raw_audios = []
fourier_Db = []
names = []

for sample in audio_files:
    raw, sr = librosa.load(sample)
    #raw_audios.append(raw)
    name = sample.replace("C:/Users/elena/Desktop/University Stuff/3rd Year/Final Major Project/Code/Keys_Recording", "")
    name = name.replace('.wav', "")
    print(name)
    spectrogram(raw, name)
    #names.append(name)
    #plot_wave(raw, name)



    #print(name)
#print(len(raw_audios))
'''
for uncompressed in raw_audios:
    fourier_trans = librosa.stft(uncompressed) # Short-time Fourier transform
    #
    ft_Db = librosa.amplitude_to_db(np.abs(fourier_trans), ref = np.max)
    fourier_Db.append(ft_Db)
#print(type(fourier_trans))
#print(len(fourier_Db))

# ---------------------------------------------- TRAINING MACHINE LEARNING MODEL
# made a list containing each xylophoen key
# with a for loop, I am populating teh list with 10 of each key label
ml_audio_data = []
xylophone_keys = ["A", "B", "C_First_Inv", "C_Root", "D", "E", "F", "G"]
keys_labels = []
for element in xylophone_keys:
    for x in range(10):
        keys_labels.append(element)'''
