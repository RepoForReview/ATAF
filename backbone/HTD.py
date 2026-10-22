from scipy import stats
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from utils import *
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def cmp(a, b):
    return bool(a > b) - bool(a < b)

def extract_features(vector):
    # shape = [batch, channel, length]
    batch_features = []
    for sample in range(vector.shape[0]):
        sample_features = []
        for c in range(vector.shape[1]):
            sample_features.append(getMAV(vector[sample, c, :]))
            sample_features.append(getSSC(vector[sample, c, :], 0.01))
            # sample_features.append(getWAMP(vector[sample, c, :],0.1))
            sample_features.append(getWL(vector[sample, c, :]))
            sample_features.append(getZC(vector[sample, c, :],0.1))
        batch_features.append(np.array(sample_features))
    batch_features = np.array(batch_features)
    return batch_features

def getMAV(vector):
    return np.mean(np.abs(vector))

def getSSC(vector, threshold=0.1):
    vector = np.asarray(vector)
    slope_change = 0
    for i in range(1,len(vector)-1):
        get_x = (vector[i]-vector[i-1])*(vector[i]-vector[i+1])
        if(get_x >= threshold):
            slope_change += 1
    return slope_change

def getWAMP(vector, threshold=0.1):
    vector = np.asarray(vector)
    wamp_decision = 0
    for i in range(1, len(vector)):
        get_x = abs(vector[i] - vector[i - 1])
        if (get_x >= threshold):
            wamp_decision += 1
    return wamp_decision


def getWL(vector):
    vector = np.asarray(vector)
    return np.sum(np.abs(np.diff(vector)))

def getZC(vector,threshold=0.1):
    vector = np.asarray(vector)
    number_zero_crossing = 0
    current_sign = cmp(vector[0], 0)
    for i in range(0, len(vector)):
        if current_sign == -1:
            if current_sign != cmp(vector[i], threshold):  # We give a delta to consider that the zero was crossed
                current_sign = cmp(vector[i], 0)
                number_zero_crossing += 1
        else:
            if current_sign != cmp(vector[i], -threshold):
                current_sign = cmp(vector[i], 0)
                number_zero_crossing += 1
    return number_zero_crossing


def HTD(total_data):
    data_train, data_val, data_test = total_data[0], total_data[1], total_data[2]
    emg_train = data_train['emg']
    label_train = data_train['label']
    emg_test = data_test['emg']
    label_test = data_test['label']

    emg_train = extract_features(emg_train)
    emg_test = extract_features(emg_test)
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(emg_train)
    X_test_pca = pca.transform(emg_test)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_pca, label_train)
    y_pred = lda.predict(X_test_pca)
    accuracy = accuracy_score(label_test, y_pred)
    print(accuracy)