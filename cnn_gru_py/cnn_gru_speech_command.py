from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import math, time, datetime
import os.path
import random
import re
import sys
import tarfile
import shutil
#print(sys.executable)
import matplotlib.pyplot as plt
import numpy as np
import librosa as rosa
import librosa.display
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
from sklearn import preprocessing #copy from echo 1110/2018

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Lambda, BatchNormalization, Activation, LSTM, GRU, concatenate
#from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
#from tensorflow.python.ops import io_ops
#from tensorflow.python.platform import gfile
#from tensorflow.python.util import compat

default_number_of_mfcc=128
default_sample_rate=16000
default_hop_length=512 
default_wav_duration=1 # 1 second
default_train_samples=10000
default_test_samples=100
default_epochs=10
default_batch_size=1024
default_train_wanted_words=["one", "two",  "backward", "bird", "cat", "dog", "five", "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off", "right", "seven", "sheila", "stop", "three", "tree",  "wow", "zero","up"]
default_test_wanted_words=["bed", "eight", "visual", "follow"]
#for mac
#speech_data_dir="/Users/hermitwang/Downloads/speech_dataset"
#default_model_path="/Users/hermitwang/Downloads/pretrained/speech_siamese"
#for ubuntu
#speech_data_dir="/home/hermitwang/TrainingData/datasets/speech_dataset"
#default_model_path="/home/hermitwang/TrainingData/pretrained/speech_siamese"
#default_processed_data_path="/home/hermitwang/TrainingData/pretrained/speech_siamese"
#for windows of work
#speech_data_dir="D:\\HermitWang\\DeepLearning\\dataset\\speech_dataset"
#default_model_path="D:\\HermitWang\\DeepLearning\\dataset\\trained\\siamese"
#for himl 
speech_data_dir="/home/zhangjun/tensorflow/speech_siamese_zj/speech_dataset"
default_model_path="/home/zhangjun/tensorflow/speech_siamese_zj/trained"
default_processed_data_path="/home/hermitwang/Projects/pretrained/speech_siamese"

tf.set_random_seed(2)



def load_wav_mfcc(filename):
    wav_loader, sample_rate = rosa.load(filename, sr=default_sample_rate)
    #print(rosa.get_duration(wav_loader, sample_rate))
    wav_mfcc = rosa.feature.mfcc(y=wav_loader, sr=default_sample_rate, n_mfcc=default_number_of_mfcc)
    wav_mfcc = np.transpose(wav_mfcc)
    return wav_mfcc

def get_default_mfcc_length(default_wav_duration=1):
    length = int(math.ceil(default_wav_duration * default_sample_rate / default_hop_length))
    return length

def mfcc_display(mfccs):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()


class WavMFCCLoader(object):
    def __init__(self, data_dir, wanted, validation_percentage=0, testing_percentage=0):
        self.data_dir = data_dir
        self.wanted = wanted
        self.default_mfcc_length=get_default_mfcc_length(default_wav_duration)
        self.wav_files = dict()
        self.wav_file_index()
        self.mfccs_mean = 0
        self.mfccs_std = 0
        
        
    def wav_file_index(self):
        for dirpath, dirnames, files in os.walk(self.data_dir):
            for name in files:
                if name.lower().endswith('.wav'):
                    #for windows
                    #word_name = dirpath.rsplit('\\', 1)[1];
                    #for others
                    word_name = dirpath.rsplit('/', 1)[1];
                    if word_name in self.wanted:
                        file_name = os.path.join(dirpath, name)
                        #print(file_name, dirpath, word_name)
    
                        if word_name in self.wav_files.keys():
                            self.wav_files[word_name].append(file_name)
                        else:
                            self.wav_files[word_name] = [file_name]
                    
        return self.wav_files


    def wavs_to_mfcc_pair(self):
        how_many_words = len(self.wanted)
        a_index = random.randint(0, how_many_words - 1)
        b_index = random.randint(0, how_many_words - 1)
        a_wav_index = b_wav_index = -1
        mfcc_pair = np.array([3, 1])
        if (a_index > b_index):
            a_wav_index = random.randint(0, len(self.wav_files[self.wanted[a_index]]) - 1)
            b_wav_index = random.randint(0, len(self.wav_files[self.wanted[b_index]]) - 1)
            mfcc_1 = load_wav_mfcc(self.wav_files[self.wanted[a_index]][a_wav_index])
            mfcc_2 = load_wav_mfcc(self.wav_files[self.wanted[b_index]][b_wav_index])
            mfcc_pair = 0            
        else:
            a_wav_index = random.randint(0, len(self.wav_files[self.wanted[a_index]]) - 1)
            b_wav_index = random.randint(0, len(self.wav_files[self.wanted[a_index]]) - 1)
            mfcc_1 = load_wav_mfcc(self.wav_files[self.wanted[a_index]][a_wav_index])
            mfcc_2 = load_wav_mfcc(self.wav_files[self.wanted[a_index]][b_wav_index])
            mfcc_pair = 1
            
        #print("aaa", mfcc_1.shape, mfcc_2.shape)    
        return mfcc_1, mfcc_2, mfcc_pair
        
    def get_mfcc_pairs(self, how_many, preprocess='global'):
        mfcc1_data = np.zeros((how_many, self.default_mfcc_length, default_number_of_mfcc))
        mfcc2_data = np.zeros((how_many, self.default_mfcc_length, default_number_of_mfcc))
        same_data = np.zeros(how_many)
        for i in range(how_many):
            
            mfcc1_data_, mfcc2_data_, same_data[i] = self.wavs_to_mfcc_pair()
            
            
            if preprocess == "global":
                self.get_sample_mean_std()
                mfcc1_data_ = (mfcc1_data_ - self.mfccs_mean[0:mfcc1_data_.shape[0], :]) / self.mfccs_std[0:mfcc1_data_.shape[0], :]
                mfcc2_data_ = (mfcc2_data_ - self.mfccs_mean[0:mfcc2_data_.shape[0], :]) / self.mfccs_std[0:mfcc2_data_.shape[0], :]
            elif preprocess == "z-score":
                mfcc1_data_ = preprocessing.scale(mfcc1_data_)
                mfcc2_data_ = preprocessing.scale(mfcc2_data_)               
            elif preprocess == "max-min-scaler":
                mfcc1_data_ = preprocessing.MinMaxScaler().fit_transform(mfcc1_data_)
                mfcc2_data_ = preprocessing.MinMaxScaler().fit_transform(mfcc2_data_)
            elif preprocess == "l2-normalize":
                mfcc1_data_ = preprocessing.normalize(mfcc1_data_, norm='l2')
                mfcc2_data_ = preprocessing.normalize(mfcc2_data_, norm='l2')
            else:
                raise ValueError("unknown proprocess")
            
            
            mfcc1_data[i, 0:mfcc1_data_.shape[0], : ] = mfcc1_data_
            mfcc2_data[i, 0:mfcc2_data_.shape[0], : ] = mfcc2_data_

    
        return mfcc1_data, mfcc2_data, same_data
    
    def get_mfcc_for_word(self, wanted, preprocess='l2-normalize'):
        mfcc_data = np.zeros((self.default_mfcc_length, default_number_of_mfcc))
        if (wanted in self.wanted):
            wav_index = random.randint(0, len(self.wav_files[wanted]) - 1)
            mfcc_data_tmp = load_wav_mfcc(self.wav_files[wanted][wav_index])
            #print("wanted word is " + wanted + " the file name is " + self.wav_files[wanted][wav_index])
            if preprocess == "z-score":
                mfcc_data_tmp = preprocessing.scale(mfcc_data_tmp)
                
            elif preprocess == "max-min-scaler":
                mfcc_data_tmp = preprocessing.MinMaxScaler().fit_transform(mfcc_data_tmp)
                
            elif preprocess == "l2-normalize":
                mfcc_data_tmp = preprocessing.normalize(mfcc_data_tmp, norm='l2')
                
            else:
                raise ValueError("unknown proprocess") 
        
            mfcc_data[0:mfcc_data_tmp.shape[0], : ] = mfcc_data_tmp
        else:
            raise ValueError("the word is not in the list")
            
        return mfcc_data, self.wav_files[wanted][wav_index]
        
    def get_sample_mean_std(self):
        count = 0
        for i in (self.wanted):
            #print(i)
            for j in range(len(self.wav_files[i])):
                #print(self.wav_files[i][j])
                count += 1
        #print("count:", count)
        mfccs = np.zeros((count, self.default_mfcc_length, default_number_of_mfcc))
        index = 0
        for i in (self.wanted):
            for j in range(len(self.wav_files[i])):       
                mfcc_ = load_wav_mfcc(self.wav_files[i][j])
                mfccs[index, 0:mfcc_.shape[0], : ] = mfcc_
                index += 1
        self.mfccs_mean = mfccs.mean(0)
        self.mfccs_std = mfccs.std(0)




def read_data_file(file_name):
    f = open(file_name, "rb")
    t = np.load(f)
    print("shape:", t.shape)
    f.close()
    return t

def write_data_file(file_name, data):
    f = open(file_name, "wb")
    np.save(f, data)
    f.close()
    

def store_processed_data(local_train_samples=default_train_samples, local_test_samples=default_test_samples, local_process_data_path=default_processed_data_path, local_preprocess='l2-normalize', local_train_wanted=default_train_wanted_words, local_test_wanted=default_test_wanted_words):
    train_loader = WavMFCCLoader(speech_data_dir, wanted=local_train_wanted)
    mfcc1_train_data, mfcc2_train_data, train_pairs = train_loader.get_mfcc_pairs(local_train_samples, preprocess=local_preprocess)
    mfcc1_eval_data, mfcc2_eval_data, eval_pairs = train_loader.get_mfcc_pairs(local_test_samples, preprocess=local_preprocess)

    test_loader  = WavMFCCLoader(speech_data_dir, wanted=local_test_wanted)
    mfcc1_test_data, mfcc2_test_data, test_pairs = test_loader.get_mfcc_pairs(local_test_samples, preprocess=local_preprocess)

    #store the training mfcc1 data to np file
    filename_train_mfcc1 = local_process_data_path + "/mfcc1_train_" + str(local_train_samples) + "_" + local_preprocess + ".npy"
    write_data_file(filename_train_mfcc1, mfcc1_train_data)
    
    #store the training mfcc2 data to np file
    filename_train_mfcc2 = local_process_data_path + "/mfcc2_train_" + str(local_train_samples) + "_" + local_preprocess + ".npy"
    write_data_file(filename_train_mfcc2, mfcc2_train_data)
    
    #store the training pair data to np file
    filename_train_pairs = local_process_data_path + "/pairs_train_" + str(local_train_samples) + "_" + local_preprocess + ".npy"
    write_data_file(filename_train_pairs, train_pairs)
    

    #store the eval mfcc1 data to np file
    filename_eval_mfcc1 = local_process_data_path + "/mfcc1_eval_" + str(local_test_samples) + "_" + local_preprocess + ".npy"
    write_data_file(filename_eval_mfcc1, mfcc1_eval_data)
    
    
    #store the eval mfcc2 data to np file
    filename_eval_mfcc2 = local_process_data_path + "/mfcc2_eval_" + str(local_test_samples) + "_" + local_preprocess + ".npy"
    write_data_file(filename_eval_mfcc2, mfcc2_eval_data)
    
    
    #store the eval pair data to np file
    filename_eval_pairs = local_process_data_path + "/pairs_eval_" + str(local_test_samples) + "_" + local_preprocess + ".npy"
    write_data_file(filename_eval_pairs, eval_pairs)
    
    
    
    #store the test mfcc1 data to np file
    filename_test_mfcc1 = local_process_data_path + "/mfcc1_test_" + str(local_test_samples) + "_" + local_preprocess + ".npy"
    write_data_file(filename_test_mfcc1, mfcc1_test_data)
    
    
    #store the test mfcc2 data to np file
    filename_test_mfcc2 = local_process_data_path + "/mfcc2_test_" + str(local_test_samples) + "_" + local_preprocess + ".npy"
    write_data_file(filename_test_mfcc2, mfcc2_test_data)
    
    
    #store the test pair data to np file
    filename_test_pairs = local_process_data_path + "/pairs_test_" + str(local_test_samples) + "_" + local_preprocess + ".npy"
    write_data_file(filename_test_pairs, test_pairs)
    
    
    
def load_processed_data(local_train_samples=default_train_samples, local_test_samples=default_test_samples, local_process_data_path=default_processed_data_path, local_preprocess='l2-normalize'):
    
    filename_train_mfcc1 = local_process_data_path + "/mfcc1_train_" + str(local_train_samples) + "_" + local_preprocess + ".npy"    
    
    if not (os.path.isfile(filename_train_mfcc1)):
        store_processed_data(local_train_samples, local_test_samples, local_process_data_path, local_preprocess)
    
    filename_train_mfcc1 = local_process_data_path + "/mfcc1_train_" + str(local_train_samples) + "_" + local_preprocess + ".npy"
    mfcc1_train_data = read_data_file(filename_train_mfcc1)
    filename_train_mfcc2 = local_process_data_path + "/mfcc2_train_" + str(local_train_samples) + "_" + local_preprocess + ".npy"
    mfcc2_train_data = read_data_file(filename_train_mfcc2)
    filename_train_pairs = local_process_data_path + "/pairs_train_" + str(local_train_samples) + "_" + local_preprocess + ".npy"
    train_pairs = read_data_file(filename_train_pairs)
    filename_eval_mfcc1 = local_process_data_path + "/mfcc1_eval_" + str(local_test_samples) + "_" + local_preprocess + ".npy"
    mfcc1_eval_data = read_data_file(filename_eval_mfcc1)
    filename_eval_mfcc2 = local_process_data_path + "/mfcc2_eval_" + str(local_test_samples) + "_" + local_preprocess + ".npy"
    mfcc2_eval_data = read_data_file(filename_eval_mfcc2)
    filename_eval_pairs = local_process_data_path + "/pairs_eval_" + str(local_test_samples) + "_" + local_preprocess + ".npy"
    eval_pairs = read_data_file(filename_eval_pairs)
    filename_test_mfcc1 = local_process_data_path + "/mfcc1_test_" + str(local_test_samples) + "_" + local_preprocess + ".npy"
    mfcc1_test_data = read_data_file(filename_test_mfcc1)
    filename_test_mfcc2 = local_process_data_path + "/mfcc2_test_" + str(local_test_samples) + "_" + local_preprocess + ".npy"
    mfcc2_test_data = read_data_file(filename_test_mfcc2)
    filename_test_pairs = local_process_data_path + "/pairs_test_" + str(local_test_samples) + "_" + local_preprocess + ".npy"
    test_pairs = read_data_file(filename_test_pairs)    
    
    return mfcc1_train_data, mfcc2_train_data, train_pairs, mfcc1_eval_data, mfcc2_eval_data, eval_pairs, mfcc1_test_data, mfcc2_test_data, test_pairs


def create_cnn_model(fingerprint_shape, is_training=True):
    model = Sequential()
    model.add(Permute((2, 1)), input_shape=fingerprint_shape)
    model.add(Reshape((fingerprint_shape[1], fingerprint_shape[0], 1)))
    
    model.add(Conv2D(filters=64, kernel_size=3, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())
    #if (is_training):
    #    model.add(Dropout(0.5))
    model.add(Conv2D(filters=64, kernel_size=3, use_bias=False)) 
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(MaxPooling2D())
    #if (is_training):
    #    model.add(Dropout(0.5))
    model.add(Conv2D(filters=64, kernel_size=3, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(MaxPooling2D())
    
    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))
    
    return model

def create_lstm_model(local_input_shape, is_training=True):
    model = Sequential()
    #model.add(BatchNormalization(input_shape=local_input_shape))
    model.add(GRU(1024, return_sequences=False, stateful=False, input_shape=local_input_shape))
    #model.add(GRU(256, return_sequences=True, stateful=False))
    #model.add(GRU(256, stateful=False))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid")) 
    #if (is_training):
    #    model.add(Dropout(0.5))
    #model.add(Dense(labels_count, activation="softmax"))
    
    return model


 def create_siamese_model(input_shape, siamese_mode = 'concat'):
    right_input = Input(input_shape)
    left_input = Input(input_shape)
    
    lstm_model = create_lstm_model(input_shape)
    cnn_model = create_cnn_model(input_shape)
    
    right_lstm_encoder = lstm_model(right_input)
    left_lstm_encoder = lstm_model(left_input)
    
    right_cnn_encoder = cnn_model(right_input)
    left_cnn_encoder = cnn_model(left_input)
    
    right_encoder = concatenate(inputs=[right_cnn_encoder, right_lstm_encoder])
    left_encoder = concatenate(inputs=[left_cnn_encoder, left_lstm_encoder])
    
    if (siamese_mode == 'minus'):
        concatenated_layer = Lambda(lambda x: x[0]-x[1], output_shape=lambda x: x[0])([right_encoder, left_encoder])
    elif (siamese_mode == 'abs'):
        concatenated_layer = Lambda(lambda x: tf.abs(x[0]-x[1]), output_shape=lambda x: x[0])([right_encoder, left_encoder])
    elif (siamese_mode == "concat"):
        concatenated_layer = concatenate(inputs=[right_encoder, left_encoder])
    else:
        raise ValueError("unknown siamese_mode")
        
    last_layer = Dense(1024, activation='relu')(concatenated_layer)
    output_layer = Dense(1, activation='sigmoid')(last_layer)
    
    siamese_model = Model([right_input, left_input], output_layer)
    return siamese_model
    
def siamese_train(mfcc1_train_data, mfcc2_train_data, train_pairs, mfcc1_eval_data, mfcc2_eval_data, eval_pairs, local_siamese_mode='concat', local_batch_size=default_batch_size, local_epochs= default_epochs):
    default_mfcc_length = get_default_mfcc_length(default_wav_duration)
    siamese_model = create_siamese_model((default_mfcc_length, default_number_of_mfcc), siamese_mode=local_siamese_mode)

    siamese_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    x1_train = mfcc1_train_data #.reshape((train_samples, default_mfcc_length, default_number_of_mfcc)) #hop length is the time feature
    x2_train = mfcc2_train_data #.reshape((train_samples, default_mfcc_length, default_number_of_mfcc)) #mfcc_number is the data feature
    y_train = train_pairs  #keras.utils.to_categorical(pairs, num_classes=1)
    
    
    siamese_model.fit([x1_train, x2_train], y_train, epochs=local_epochs, batch_size=local_batch_size)
    
    
    x1_test = mfcc1_eval_data #.reshape((default_test_samples, default_mfcc_length, default_number_of_mfcc))
    x2_test = mfcc2_eval_data #.reshape((default_test_samples, default_mfcc_length, default_number_of_mfcc))
    y_test = eval_pairs
    
    loss, accuracy = siamese_model.evaluate([x1_test, x2_test], y_test)    
    
    siamese_model.save(default_model_path+"/speech_siamese.h5")

    print(loss)
    return accuracy

def siamese_test(mfcc1_test_data, mfcc2_test_data, test_pairs):
    default_mfcc_length = get_default_mfcc_length(default_wav_duration)    
    siamese_model = keras.models.load_model(default_model_path+"/speech_siamese.h5")    
    x1_test = mfcc1_test_data #.reshape((test_samples, default_mfcc_length, default_number_of_mfcc))
    x2_test = mfcc2_test_data #.reshape((test_samples, default_mfcc_length, default_number_of_mfcc))
    y_test = test_pairs
    
    loss, accuracy = siamese_model.test_on_batch(x=[x1_test, x2_test], y=y_test)
    print(loss)
    
    #keep the accurcy beyond .75 to a separated file
    if accuracy > .75:
        good_model_filename = default_model_path+ "/speech_siamese_"+str(int(accuracy * 100))+".h5"
        shutil.copyfile(default_model_path+"/speech_siamese.h5", good_model_filename)
        
    return accuracy


#load_processed_data(local_train_samples=10000, local_test_samples=100, local_preprocess='z-score')
load_processed_data(local_train_samples=10000, local_test_samples=100, local_preprocess='l2-normalize')
#load_processed_data(local_train_samples=10000, local_test_samples=100, local_preprocess='global')
#load_processed_data(local_train_samples=10000, local_test_samples=100, local_preprocess='max-min-scaler')

'''
train_samples = default_train_samples * 5
test_samples = default_test_samples
batch_size_numbers=[128, 256, 512, 1024, 2048]
train_epochs = 30
print("Start at " + str(datetime.datetime.now()))
mfcc1_train_data, mfcc2_train_data, train_pairs, mfcc1_eval_data, mfcc2_eval_data, eval_pairs, mfcc1_test_data, mfcc2_test_data, test_pairs = load_processed_data(local_train_samples=train_samples, local_test_samples=test_samples, local_preprocess='l2-normalize')

for i in range(10):
    for j in range(4, 5):
        print("Train batch size " + str(batch_size_numbers[j]) + " of " + str(i+1) + " run start at " + str(datetime.datetime.now()) + ":")
        score=siamese_train(mfcc1_train_data, mfcc2_train_data, train_pairs, mfcc1_eval_data, mfcc2_eval_data, eval_pairs, local_siamese_mode='abs', local_batch_size=batch_size_numbers[j], local_epochs=train_epochs)
        print(score)
        score=siamese_test(mfcc1_test_data, mfcc2_test_data, test_pairs)
    
        print(score)
'''