from collections import defaultdict
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.signal import butter, lfilter, iirnotch
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from data.inter_dataset_provide import Hyser_provide, Ninapro_provide, Senic_provide


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def notch_filter(data, notch_freq, fs, quality_factor=30):
    b, a = iirnotch(notch_freq, quality_factor, fs)
    y = lfilter(b, a, data)
    return y


def normalize_data(emg_train, emg_val, emg_test):
    train_reshaped = emg_train.reshape(-1, emg_train.shape[1])
    channel_mean = np.mean(train_reshaped, axis=0)  # 形状 (12,)
    channel_std = np.std(train_reshaped, axis=0)  # 形状 (12,)
    channel_std[channel_std == 0] = 1e-8
    mean = channel_mean.reshape(1, -1, 1)
    std = channel_std.reshape(1, -1, 1)
    X_train_norm = (emg_train - mean) / std
    X_val_norm = (emg_val - mean) / std
    X_test_norm = (emg_test - mean) / std
    return X_train_norm, X_val_norm, X_test_norm


def normalize_data_2(emg_train, emg_test):
    train_reshaped = emg_train.reshape(-1, emg_train.shape[1])
    channel_mean = np.mean(train_reshaped, axis=0)  # 形状 (12,)
    channel_std = np.std(train_reshaped, axis=0)  # 形状 (12,)
    channel_std[channel_std == 0] = 1e-8
    mean = channel_mean.reshape(1, -1, 1)
    std = channel_std.reshape(1, -1, 1)
    X_train_norm = (emg_train - mean) / std
    X_test_norm = (emg_test - mean) / std
    return X_train_norm, X_test_norm


def filter(emg, fs, notch_freq):
    lowcut = 20.0
    highcut = 200.0
    emg = bandpass_filter(emg, lowcut, highcut, fs, order=5)
    filtered_emg = notch_filter(emg, notch_freq, fs)
    return filtered_emg


def resample_emg(emg_data, original_fs=2048, target_fs=1000):
    num_samples = int(emg_data.shape[0] * target_fs / original_fs)
    downsampled_emg = resample(emg_data, num_samples, axis=0)
    return downsampled_emg


def obtain_hyser_emg_label(emg, labels, cut_length, sliding):
    segmented_emg = []
    segmented_labels = []
    remove_points = int(0.25 * 1000)
    for i, emg in enumerate(emg):
        label = labels[i]
        emg = emg[remove_points:]
        num_points = emg.shape[0]
        emg = filter(emg, fs=1000, notch_freq=50)
        if num_points < cut_length:
            continue
        for start in range(0, num_points - cut_length + 1, sliding):
            segment = emg[start:start + cut_length]
            segmented_emg.append(segment)
            segmented_labels.append(label)
    segmented_emg = np.array(segmented_emg).transpose((0, 2, 1))  # shape: (num_segments, window_length, 256)
    segmented_labels = np.array(segmented_labels)  # shape: (num_segments,)
    return segmented_emg, segmented_labels


def hyser_read_mat(mat_emg, labels, cut_length, sliding):
    emg_data = mat_emg[0]
    labels = labels[0]-1
    label_indices = defaultdict(list)
    for i, label in enumerate(labels):
        label_indices[label].append(i)
    train_indices = []
    test_indices = []
    for label, indices in label_indices.items():
        indices.sort()
        if len(indices) >= 2:
            test_indices.append(indices[-1])
            train_indices.extend(indices[:-1])
        elif len(indices) == 1:
            train_indices.extend(indices)
    train_data = [resample_emg(emg_data[i]) for i in train_indices]
    test_data = [resample_emg(emg_data[i]) for i in test_indices]

    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    emg_ges_train, label_ges_train = obtain_hyser_emg_label(train_data, train_labels, cut_length, sliding)
    emg_ges_test, label_ges_test = obtain_hyser_emg_label(test_data, test_labels, cut_length, sliding)
    emg_ges_train, emg_ges_val, label_ges_train, label_ges_val = train_test_split(emg_ges_train, label_ges_train,
                                                                                  test_size=0.2, random_state=42)
    return emg_ges_train, label_ges_train, emg_ges_val, label_ges_val, emg_ges_test, label_ges_test


def select_and_shuffle_samples(emg_train, label_train, num_samples=5):
    unique_labels = np.unique(label_train)
    selected_emg = []
    selected_labels = []
    for label in unique_labels:
        indices = np.where(label_train == label)[0]
        chosen_indices = np.random.choice(indices, num_samples, replace=False)
        selected_emg.append(emg_train[chosen_indices])
        selected_labels.append(label_train[chosen_indices])
    selected_emg = np.vstack(selected_emg)
    selected_labels = np.hstack(selected_labels)
    shuffled_indices = np.random.permutation(len(selected_labels))
    return selected_emg[shuffled_indices], selected_labels[shuffled_indices]


def Hyser_signal_dataload(args):
    cut_length = 200
    sliding = 50
    if args.adapt_mode == 'intra-subject':
        source_path_1 = 'data/Hyser/subject'+str(args.source_parti)+'_session1.mat'
        source_path_2 = 'data/Hyser/subject'+str(args.source_parti)+'_session2.mat'
        source_mat_1 = scipy.io.loadmat(source_path_1)
        source_mat_2 = scipy.io.loadmat(source_path_2)
        data_1 = source_mat_1['data']
        label_1 = source_mat_1['label']
        data_2 = source_mat_2['data']
        label_2 = source_mat_2['label']

        emg_train_1, label_train_1, emg_val_1, label_val_1, emg_test_1, label_test_1 = hyser_read_mat(data_1, label_1, cut_length, sliding)
        emg_train_2, label_train_2, emg_val_2, label_val_2, emg_test_2, label_test_2 = hyser_read_mat(data_2, label_2, cut_length, sliding)
        emg_train = np.concatenate([emg_train_1, emg_train_2], axis=0)
        label_train = np.concatenate([label_train_1, label_train_2], axis=0)
        emg_val = np.concatenate([emg_val_1, emg_val_2], axis=0)
        label_val = np.concatenate([label_val_1, label_val_2], axis=0)
        emg_test = np.concatenate([emg_test_1, emg_test_2], axis=0)
        label_test = np.concatenate([label_test_1, label_test_2], axis=0)
        emg_train, emg_val, emg_test = normalize_data(emg_train, emg_val, emg_test)
        train_data = {'emg': emg_train, 'label': label_train}
        val_data = {'emg': emg_val, 'label': label_val}
        test_data = {'emg': emg_test, 'label': label_test}
        total_data = [train_data, val_data, test_data]

    elif args.adapt_mode == 'inter-subject':
        total_data = Hyser_provide(args)

    elif args.adapt_mode == 'inter-device':
        if args.target_device == 'Ninapro':
            total_data = Ninapro_provide(args)
        elif args.target_device == 'Senic':
            total_data = Senic_provide(args)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return total_data