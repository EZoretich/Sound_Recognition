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

import joblib
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import csv

#from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


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
ml_audio_data = []

for sample in audio_files:
    raw, sr = librosa.load(sample)
    #raw = raw[7000:48951] # Give all samples same length
    raw_audios.append(raw)
    #plot_wave(raw) # Plot waveform graph
    #spectrogram(raw) # Plot Spectrogram

for uncompressed in raw_audios:
    fourier_trans = librosa.stft(uncompressed) # Short-time Fourier transform
    ft_avg = np.mean(np.abs(fourier_trans), axis = 1)
    ml_audio_data.append(ft_avg)

ml_audio_data = np.array(ml_audio_data)


#-------------------------------------------- TRAINING MACHINE LEARNING MODEL
# made a list containing each xylophoen key
# with a for loop, I am populating the list with 10 of each key label
#ml_audio_data = []
xylophone_keys = ["A", "B", "C_First_Inv", "C_Root", "D", "E", "F", "G"]
keys_labels = []
for element in xylophone_keys:
    for x in range(10):
        keys_labels.append(element)

# ----------------------------------------------------------

X = ml_audio_data
y = keys_labels


#model = MLPClassifier(alpha=1, max_iter=1000)
#print(" NEURAL NETWORK - MLP")

#model = SVC(kernel="linear", C=0.025) # extremely good
#print(" LINEAR SVM")

#model = GaussianNB()
#print("---------- NAIVE BAYES ----------")

model = KNeighborsClassifier(3) # Best one so far --> from 56-85%
#print(" K NEAREST NEIGHBORS")

#model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
#print(" RANDOM FOREST")

#model = GaussianProcessClassifier(1.0 * RBF(1.0))
#print(" GAUSSIAN PROCESS")

# make your own test/train, so ML model trains better.
# because you do not kwow.
# Bar chart to see accuracy of data and their position.

# Split train/test dataset to include every note in testing

X_train = np.concatenate((X[1:11,:], X[12:21,:], X[22:31,:], X[32:41,:], X[42:51,:], X[52:61,:], X[62:71,:], X[72:,:]))
X_test = np.concatenate((X[:1,:], X[11:12,:], X[21:22,:], X[31:32,:], X[41:42,:], X[51:52,:], X[61:62,:], X[71:72,:]))
y_train = np.concatenate((y[1:11], y[12:21], y[22:31], y[32:41], y[42:51], y[52:61], y[62:71], y[72:]))
y_test = np.concatenate((y[:1], y[11:12], y[21:22], y[31:32], y[41:42], y[51:52], y[61:62], y[71:72]))


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.8, shuffle=True)

# The train_test_split parameter 'shuffle = True' will allow to randomly select the cards (features)
# used for training and testing, providing a non-bias model
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
score = model.score(X_test, y_test)
print("-- Accuracy:     ", score*100, '%')
for h in range(len(y_predict)):
    print(y_predict[h], y_test[h])


# --------------------  SAVING THE TRAINED MODEL
# The trained model is saved in a 'joblib file,
# and it will be later loaded in another code,

### Trained model already saved as CSV file (100%)
'''
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
