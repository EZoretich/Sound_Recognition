# ------------------------------------------  UNO CARD RECOGNITION FROM IMAGES
# --------------------- IMPORT LIBRARIES
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from glob import glob
from itertools import cycle

#from pydub import AudioSegment
#from pydub.playback import play
#import joblib
#from sklearn import datasets
#from sklearn.ensemble import RandomForestClassifier


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

sns.set_theme(style='white', palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

audio_files = glob('C:/Users/elena/Desktop/University Stuff/3rd Year/Final Major Project/Code/Keys_Recording/*.wav')

# ------ For some reason, only one file does not work. ????
#key = glob('C:/Users/elena/Desktop/University Stuff/3rd Year/Final Major Project/Code/Keys_Recording/A_1.wav')



# --- I CANNOT PLAY AUDIO ---
#audio = AudioSegment.from_file(audio_files[0], format='wav')
#play(audio)

# ra = raw audio, sr = sample rate
ra , sr = librosa.load(audio_files[0]) # A_1
ra2, sr2 = librosa.load(audio_files[11]) #B_2
ra3, sr3 = librosa.load(audio_files[21]) # C_First_Riv_2
ra4, sr4 = librosa.load(audio_files[31]) #  C_Root_2
ra5, sr5 = librosa.load(audio_files[41]) #D_2
ra6, sr6 = librosa.load(audio_files[51]) #E_2
ra7, sr7 = librosa.load(audio_files[61]) # F_2
ra8, sr8 = librosa.load(audio_files[71]) #G_2
#print(ra)
#print(sr)


# -------------------- PLOT AUDIO --> NORMAL (NOT TRIMMED)
'''pd.Series(ra).plot(figsize=(10, 5), lw=1, title='Raw Audio Trial', color=color_pal[0])
plt.show()'''

# -------------------- PLOT AUDIO --> TRIMMED (LOWERED THRESHOLD)
'''ra_trim, _ = librosa.effects.trim(ra, top_db=20)
pd.Series(ra_trim).plot(figsize=(10, 5), lw=1, title='Raw Audio Trimmed Trial', color=color_pal[2])
plt.show()'''

# -------------------- PLOT AUDIO --> SLICED (ZOOMED IN)
#19000:21000 19000:25000
'''pd.Series(ra[19000:22000]).plot(figsize=(10, 5), lw=1, title='Raw Audio Zoomed Trial', color=color_pal[3])
plt.show()'''

# -------------------- PLOT EACH NOTE AUDIO --> NORMAL (NOT TRIMMED)
'''pd.Series(ra).plot(figsize=(10, 5), lw=1, title='Raw Audio Zoomed Trial', color=color_pal[3])
#plt.show()
pd.Series(ra2).plot(figsize=(10, 5), lw=1, title='Raw Audio Zoomed Trial', color=color_pal[3])
#plt.show()
pd.Series(ra3).plot(figsize=(10, 5), lw=1, title='Raw Audio Zoomed Trial', color=color_pal[3])
#plt.show()
pd.Series(ra4).plot(figsize=(10, 5), lw=1, title='Raw Audio Zoomed Trial', color=color_pal[3])
#plt.show()
pd.Series(ra5).plot(figsize=(10, 5), lw=1, title='Raw Audio Zoomed Trial', color=color_pal[3])
#plt.show()
pd.Series(ra6).plot(figsize=(10, 5), lw=1, title='Raw Audio Zoomed Trial', color=color_pal[3])
#plt.show()
pd.Series(ra7).plot(figsize=(10, 5), lw=1, title='Raw Audio Zoomed Trial', color=color_pal[3])
#plt.show()
pd.Series(ra8).plot(figsize=(10, 5), lw=1, title='Raw Audio Zoomed Trial', color=color_pal[3])
plt.show()''' # if only last ~plt.show~ uncommented --> show all together

# --------- Display each note separately
'''
plot_wave(ra)
plot_wave(ra2)
plot_wave(ra3)
plot_wave(ra4)
plot_wave(ra5)'''

# --------- Display each note in same graph, to see differences (same as pandas approach)
'''plt.plot(ra, 'b')
plt.plot(ra2, 'g')
plt.plot(ra3, 'r')
plt.plot(ra4, 'c')
plt.plot(ra5, 'm')
plt.plot(ra6, 'y')
plt.plot(ra7, 'k')
plt.plot(ra8, 'purple')

plt.show()'''

# --------- Fourier transform (in Librosa) and Spectrogram ~~ Singular file
'''
A_ft = librosa.stft(ra)
A_db = librosa.amplitude_to_db(np.abs(A_ft), ref = np.max)

fix, ax = plt.subplots(figsize=(10,5))
A_spectrogram = librosa.display.specshow(A_db, x_axis='time', y_axis='log', ax = ax)
#print(A_ft)
#print(A_db)
#print(A_db.shape)
plt.show()'''

# ----------- Spectrogram for each note (they are a little similar)
'''spectrogram(ra)
spectrogram(ra2)
spectrogram(ra3)
spectrogram(ra4)
spectrogram(ra5)
spectrogram(ra6)
spectrogram(ra7)
spectrogram(ra8)'''

