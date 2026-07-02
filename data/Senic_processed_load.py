import os
import re
import numpy as np
import pandas as pd
from scipy.signal import resample, butter, lfilter, iirnotch
from sklearn.model_selection import train_test_split
from data.inter_dataset_provide import Hyser_provide, Ninapro_provide, Senic_provide

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut=20, highcut=100, fs=1000, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data, axis=0)


def notch_filter(data, notch_freq=50, fs=1000, quality_factor=30):
    b, a = iirnotch(notch_freq / (fs / 2), quality_factor)
    return lfilter(b, a, data, axis=0)


# 重采样
def resample_emg(emg_data, original_fs=200, target_fs=1000):
    num_samples = int(emg_data.shape[0] * target_fs / original_fs)
    return resample(emg_data, num_samples, axis=0)


# 滑动窗口分割
def segment_emg(emg_data, label, window_size=200, stride=50):
    segments = []
    labels = []
    for start in range(0, len(emg_data) - window_size + 1, stride):
        segment = emg_data[start:start + window_size]  # 取窗口内的数据
        if segment.shape[0] == window_size:  # 确保窗口完整
            segments.append(segment)
            labels.append(label)
    return np.array(segments), np.array(labels)


def normalize_data(emg_train, emg_val, emg_test):
    # 计算每个通道的均值和标准差
    channel_mean = np.mean(emg_train, axis=(0, 2))
    channel_std = np.std(emg_train, axis=(0, 2))
    # 防止除零错误
    channel_std[channel_std == 0] = 1e-8
    # 标准化数据
    mean = channel_mean.reshape(1, -1, 1)
    std = channel_std.reshape(1, -1, 1)
    X_train_norm = (emg_train - mean) / std
    X_val_norm = (emg_val - mean) / std
    X_test_norm = (emg_test - mean) / std
    return X_train_norm, X_val_norm, X_test_norm


def Senic_signal_dataload(args):
    gesture_mapping = {"rest": 0, "eversion": 1, "fist": 2, "open_hand": 3,
                   "pinch_forefinger": 4, "pinch_middlefinger": 5, "two": 6, "varus": 7,}
    if args.adapt_mode == 'intra-subject':
        folder_path = "data/Senic/h"+str(args.source_parti)
        X_train, y_train = [], []
        X_test, y_test = [], []

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):
                match = re.match(r"emg_p(\d+)_r(\d+)_(\w+)\.csv", file_name)
                if match:
                    i, j, gesture_name = match.groups()
                    j = int(j)

                    file_path = os.path.join(folder_path, file_name)
                    df = pd.read_csv(file_path, header=None)
                    emg_data = df.to_numpy()

                    rest_emg = emg_data[:400]  # 前2秒数据（200Hz * 2s = 400样本）
                    rest_emg = resample_emg(rest_emg)
                    rest_emg = bandpass_filter(rest_emg)
                    rest_emg = notch_filter(rest_emg)

                    if gesture_name in gesture_mapping:
                        gesture = gesture_mapping[gesture_name]
                        gesture_emg = emg_data[400:]  # 取剩余数据
                        gesture_emg = resample_emg(gesture_emg)
                        gesture_emg = bandpass_filter(gesture_emg)
                        gesture_emg = notch_filter(gesture_emg)

                        rest_segments, rest_labels = segment_emg(rest_emg, 0)
                        gesture_segments, gesture_labels = segment_emg(gesture_emg, gesture)

                        if j in [0, 1]:  # 训练集
                            X_train.extend(rest_segments)
                            y_train.extend(rest_labels)
                            X_train.extend(gesture_segments)
                            y_train.extend(gesture_labels)
                        else:  # 测试集
                            X_test.extend(rest_segments)
                            y_test.extend(rest_labels)
                            X_test.extend(gesture_segments)
                            y_test.extend(gesture_labels)

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)

        X_train = np.transpose(X_train, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)

        train_data = {'emg': X_train, 'label': y_train}
        val_data = {'emg': X_val, 'label': y_val}
        test_data = {'emg': X_test, 'label': y_test}
        total_data = [train_data, val_data, test_data]
    elif args.adapt_mode == 'inter-subject':
        total_data = Senic_provide(args)

    elif args.adapt_mode == 'inter-device':
        if args.target_device == 'Hyser':
            total_data = Hyser_provide(args)
        elif args.target_device == 'Ninapro':
            total_data = Ninapro_provide(args)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


    return total_data

