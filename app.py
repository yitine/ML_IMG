# github : https://github.com/yitine/ML_P6

import pandas as pd
import numpy as np
from joblib import load
#import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model, load_model

data = pd.read_csv("data.csv")
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)


input_df = Input(shape=(3,)) #nb_features

def baseline() :

	kmeans = load('kmeans.h5')

	score = silhouette_score(data_scaled, kmeans.labels_)
 
	return score

# New model
# reference : https://www.kaggle.com/code/gauravduttakiit/clustering-using-autoencoders-ann#APPLY-K-MEANS-METHOD

def build_autoencoder() :
    encoding_dim = 3 #nb_clusters
    # Glorot normal initializer (Xavier normal initializer) draws samples from a truncated normal distribution
    x = Dense(encoding_dim, activation='relu')(input_df)
    x = Dense(64, activation='relu', kernel_initializer='normal')(x)
    x = Dense(32, activation='relu', kernel_initializer='normal')(x)
    encoded = Dense(10, activation='relu', kernel_initializer='normal')(x)
    x = Dense(32, activation='relu', kernel_initializer='normal')(encoded)
    x = Dense(64, activation='relu', kernel_initializer='normal')(x)
    decoded = Dense(3, activation = 'sigmoid', kernel_initializer='normal')(x) #nb_features
    return encoded, decoded

encoded, decoded = build_autoencoder()


def compile_autoencoder(decoded = decoded, encoded=encoded) :
    # autoencoder
    autoencoder = Model(input_df, decoded)
    autoencoder.compile(optimizer= 'adam', loss='mean_squared_error')
    autoencoder.fit(data_scaled, data_scaled, batch_size = 128, epochs = 25,  verbose = 1)
    return autoencoder

def encoding(encoded = encoded) :
    #encoder - used for our dimention reduction
    encoder = Model(input_df, encoded)
    pred = encoder.predict(data_scaled)
    kmeans = load('kmeans.h5')
    ae_kmeans = kmeans.fit(pred)
    score = silhouette_score(pred, ae_kmeans.labels_)
    return score

def main() :
	baseline()
	build_autoencoder()
	compile_autoencoder()
	encoding()

if __name__ == "__main__" :
	main()