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
import csv
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


# ----------------------------- FUNCTIONS
# Display Graph for Audio file
def plot_wave(file):
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.plot(file, 'g')
    plt.show()

# Create a Spetrogram
def spectrogram(note):
    note_ft = librosa.stft(note)
    note_db = librosa.amplitude_to_db(np.abs(note_ft), ref = np.max)

    img, ax = plt.subplots(figsize=(10,5))
    note_spectrogram = librosa.display.specshow(note_db, x_axis='time', y_axis='linear', ax = ax)
    ax.set_title('Spectogram', fontsize=20)
    img.colorbar(note_spectrogram, ax=ax, format=f'%0.2f')
    plt.show()



audio_files = glob('C:/Users/elena/Desktop/University Stuff/3rd Year/Final Major Project/Code/Keys_Recording/*.wav')

raw_audios = []
fourier_Db = []
min_freq = 1000
max_freq = 3000

for sample in audio_files:
    raw, sr = librosa.load(sample)
    raw = raw[7000:48951] # Give all samples same length
    raw_audios.append(raw)
    #plot_wave(raw) # Plot waveform graph
    #spectrogram(raw) # Plot Spectrogram


#print(len(raw_audios))

for uncompressed in raw_audios:
    # Fouries Transform and Convertion to Db
    
    fourier_trans = librosa.stft(uncompressed) # Short-time Fourier transform
    ft_Db = librosa.amplitude_to_db(np.abs(fourier_trans), ref = np.max)
    
    # Rolloff -- It is possible to plot both highest and lowest percentage of rolloff
    # In the spectrogram. Although they appear all the same
    '''rolloff = librosa.feature.spectral_rolloff(y = raw, sr = sr) # default 0.85%
    print(rolloff)
    #print(rolloff.shape)
    fig, ax = plt.subplots()
    sh = librosa.display.specshow(librosa.amplitude_to_db(fourier_trans, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax)
    ax.plot(librosa.times_like(rolloff), rolloff[0], label='Roll-off frequency (0.85)')
    ax.legend(loc='lower right')
    ax.set(title='log Power spectrogram')
    #fig.colorbar(sh, ax=ax, format=f'%0.2f')
    plt.show()'''

    # fft_frequencies & amplitude
    
    st_ft = librosa.stft(uncompressed, n_fft = 2048)
    frequencies = librosa.fft_frequencies(sr = sr, n_fft = 2048)
    #ft_Db = librosa.amplitude_to_db(np.abs(st_ft), ref = np.max)
    freq = librosa.fft_frequencies(sr = sr, n_fft = 4096)
    min_freq_ind = (np.abs(freq - min_freq)).argmin()
    max_freq_ind = (np.abs(freq - max_freq)).argmin()
    amplitudes = np.abs(st_ft[min_freq_ind:max_freq_ind, :])

    #plot_wave(amplitudes)
    plt.plot(frequencies, librosa.amplitude_to_db(np.abs(st_ft), ref = np.max))#np.abs(st_ft))
    #plt.plot(frequencies, np.abs(st_ft))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    #plt.xlim(0,3000)
    plt.show()

    
#print(frequencies)
#print(type(frequencies))
#print(frequencies.shape)

# Just tests for my own clarity

#print("---------- FREQ ----------")
#print(freq)
#print("---------- MIN FREQ ----------")
#print(min_freq_ind)
#print("---------- MAX FREQ ----------")
#print(max_freq_ind)
#print("---------- AMPLITUDES ----------")
#print(amplitudes.shape)
#print(amplitudes)

'''
print(type(fourier_trans)) #np.array # Just Testing and Checking
print(fourier_trans[0]) 
print(fourier_trans.shape) # (1025, 82)
print(len(fourier_Db)) # 80
'''
'''
print(type(ft_Db)) # np.array
print(ft_Db)
print(ft_Db.shape) #(1025, 82)
print(len(ft_Db)) #1025
'''

# ---------------------------------------------- TRAINING MACHINE LEARNING MODEL
# made a list containing each xylophoen key
# with a for loop, I am populating the list with 10 of each key label
#ml_audio_data = []
'''xylophone_keys = ["A", "B", "C_First_Inv", "C_Root", "D", "E", "F", "G"]
keys_labels = []
for element in xylophone_keys:
    for x in range(10):
        keys_labels.append(element)
        
#print(keys_labels)
#print(len(keys_labels))
#print(type(fourier_Db))
#print(type(fourier_Db[1]))
ml_audio_data = fourier_Db
ml_audio_data = np.array(ml_audio_data)
#ml_audio_data.flatten()
print(ml_audio_data.shape) #Block size, row size and column size --> (80, 1025, 82)
#print(ml_audio_data[0])
print(ml_audio_data)
#print(len(ml_audio_data))
#ml_audio_data = np.array(ml_audio_data)

# ----------------------------------------------------------


X = ml_audio_data
y = keys_labels

model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
#model = GaussianProcessClassifier(1.0 * RBF(1.0)),
# 12.5% --> 10 / 80
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.125, shuffle=True)

# The train_test_split parameter 'shuffle = True' will allow to randomly select the cards (features)
# used for training and testing, providing a non-bias model
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
score = model.score(X_test, y_test)
print(score)
for h in range(len(y_predict)):
    print(y_predict[h], y_test[h])

# --------------------  SAVING THE TRAINED MODEL
# The trained model is saved in a 'joblib file,
# and it will be later loaded in another code,

ml_filename = 'ML_model_samples'
trained_model = joblib.dump(model, ml_filename + ".joblib", compress=0)
'''




# --------------------------------------- TO LOAD MODEL
# ------- ~~ For another script ~~
'''
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
import csv
from sklearn.model_selection import train_test_split

audio_files = glob('C:/Users/elena/Desktop/University Stuff/3rd Year/Final Major Project/Code/Keys_Recording/*.wav')

model_path = ## Copy path to trained model ##

raw_audios = []
fourier_Db = []

for sample in audio_files:
    raw, sr = librosa.load(sample)
    raw = raw[7000:48951]
    #print(len(raw))#, sample)
    #plot_wave(raw)
    raw_audios.append(raw)

#plot_wave(raw_audios[36])
#print(len(raw_audios))

for uncompressed in raw_audios:
    fourier_trans = librosa.stft(uncompressed) # Short-time Fourier transform
    #print(fourier_trans.shape)
    #
    ft_Db = librosa.amplitude_to_db(np.abs(fourier_trans), ref = np.max)
    fourier_Db.append(ft_Db)


ml_audio_data = []
xylophone_keys = ["A", "B", "C_First_Inv", "C_Root", "D", "E", "F", "G"]
keys_labels = []
for element in xylophone_keys:
    for x in range(10):
        keys_labels.append(element)
        

X = ml_audio_data
y = keys_labels

# Split features and labels into training and testing sets.
# ±86% of cards(features) will be used for training, and teh remaining 14% will be used for testing
# train_test_split() argument 'shuffle' has been set to True, allowing a random selection of card
# to be used each time for training and testing. This will create a non-bias model.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.125, shuffle=True)

# Loading the Machine Learning trained model, with joblib
load_model = joblib.load(model_path)

#Print out theaccuracy percentage (0-1)
print(load_model.score(X_test, y_test))
prediction = load_model.predict(X_test)


# The following loops allow to display the result from machine learning number estimation
# in the respective card image
j = 0
for element in X_test:
    j += 1
    for samples in audio_files:
        if (element == samples).all():
            print(prediction[j-1])

'''
