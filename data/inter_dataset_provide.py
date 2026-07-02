from collections import defaultdict
import scipy
from scipy.io import loadmat
from scipy.signal import butter, lfilter, iirnotch
from scipy.signal import resample
from sklearn.model_selection import train_test_split
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
import os
import re
import numpy as np
import pandas as pd


def import_db3(folder_path, subject, rest_length_cap=5):
    """Function for extracting data from raw NinaiPro files for DB2.
    Args:
        folder_path (string): Path to folder containing raw mat files
        subject (int): 1-40 which subject's data to import
        rest_length_cap (int, optional): The number of seconds of rest data to keep before/after a movement
    Returns:
        Dictionary: Raw EMG data, corresponding repetition and movement labels, indices of where repetitions are
            demarked and the number of repetitions with capped off rest data
    Note:
        Last 9 "movements" are actually force exercises
    """
    fs = 2000

    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_E1_A1.mat')
    data = sio.loadmat(cur_path)
    emg = np.squeeze(np.array(data['emg']))
    rep = np.squeeze(np.array(data['rerepetition']))
    move = np.squeeze(np.array(data['restimulus']))

    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_E2_A1.mat')
    data = sio.loadmat(cur_path)
    emg = np.vstack((emg, np.array(data['emg'])))
    rep = np.append(rep, np.squeeze(np.array(data['rerepetition'])))
    move_tmp = np.squeeze(np.array(data['restimulus']))
    move = np.append(move, move_tmp)  # Note no fix needed for this exercise


    move = move.astype('int8')  # To minimise overhead

    # Label repetitions using new block style: rest-move-rest regions
    move_regions = np.where(np.diff(move))[0]
    rep_regions = np.zeros((move_regions.shape[0],), dtype=int)
    nb_reps = int(round(move_regions.shape[0] / 2))
    last_end_idx = int(round(move_regions[0] / 2))
    nb_unique_reps = np.unique(rep).shape[0]  # To account for 0 regions
    nb_capped = 0
    cur_rep = 1

    rep = np.zeros([rep.shape[0], ], dtype=np.int8)  # Reset rep array
    for i in range(nb_reps - 1):
        rep_regions[2 * i] = last_end_idx
        midpoint_idx = int(round((move_regions[2 * (i + 1) - 1] +
                                  move_regions[2 * (i + 1)]) / 2)) + 1

        trailing_rest_samps = midpoint_idx - move_regions[2 * (i + 1) - 1]
        if trailing_rest_samps <= rest_length_cap * fs:
            rep[last_end_idx:midpoint_idx] = cur_rep
            last_end_idx = midpoint_idx
            rep_regions[2 * i + 1] = midpoint_idx - 1
        else:
            rep_end_idx = (move_regions[2 * (i + 1) - 1] +
                           int(round(rest_length_cap * fs)))
            rep[last_end_idx:rep_end_idx] = cur_rep
            last_end_idx = ((move_regions[2 * (i + 1)] -
                             int(round(rest_length_cap * fs))))
            rep_regions[2 * i + 1] = rep_end_idx - 1
            nb_capped += 2

        cur_rep += 1
        if cur_rep > nb_unique_reps:
            cur_rep = 1

    end_idx = int(round((emg.shape[0] + move_regions[-1]) / 2))
    rep[last_end_idx:end_idx] = cur_rep
    rep_regions[-2] = last_end_idx
    rep_regions[-1] = end_idx - 1

    return {'emg': emg,
            'rep': rep,
            'move': move,
            'rep_regions': rep_regions,
            'nb_capped': nb_capped
            }


def segment_emg(emg_data, label, window_size=200, stride=50):
    segments = []
    labels = []
    for start in range(0, len(emg_data) - window_size + 1, stride):
        segment = emg_data[start:start + window_size]  # 取窗口内的数据
        if segment.shape[0] == window_size:  # 确保窗口完整
            segments.append(segment)
            labels.append(label)
    return np.array(segments), np.array(labels)

def normalise_emg(emg, reps, train_reps, movements=None, which_moves=None):
    """Preprocess train+test data to mean 0, std 1 based on training data only.
    Args:
        emg (array): Raw EMG data
        reps (array): Corresponding repetition information for each EMG observation
        train_reps (array): Which repetitions are in the training set
        movements (array, optional): Movement labels, required if using which_moves
        which_moves (array, optional): Which movements to return - if None use all

    Returns:
        array: Rescaled EMG data
    """
    train_targets = get_idxs(reps, train_reps)
    # Keep only selected movement(s)
    if which_moves is not None and movements is not None:
        move_targets = get_idxs(movements[train_targets], which_moves)
        train_targets = train_targets[move_targets]
    scaler = StandardScaler(with_mean=True,
                            with_std=True,
                            copy=False).fit(emg[train_targets, :])

    return scaler.transform(emg)


def get_windows(which_reps, window_len, window_inc, emg, movements, repetitons, which_moves=None, get_valid_windows=False, dtype=np.float32):
    """Get set of windows based on repetition and movement criteria and associated label + repetition data.

    Args:
        which_reps (array): Which repetitions to return
        window_len (int): Desired window length
        window_inc (int): Desired window increment
        emg (array): EMG data (should be normalise beforehand)
        movements (array): Movement labels
        repetitons (array): Repetition labels
        which_moves (array, optional): Which movements to return - if None use all
        dtype (TYPE, optional): What precision to use for EMG data
        get_valid_windows (BOOL, optional): option to exclude invalid windows with all zeros

    Returns:
        X_data (array): Windowed EMG data
        Y_data (array): Movement label for each window
        R_data (array): Repetition label for each window
    """
    nb_obs = emg.shape[0]
    nb_channels = emg.shape[1]

    # All possible window end locations given an increment size
    possible_targets = np.array(range(window_len - 1, nb_obs, window_inc))
    targets = get_idxs(repetitons[possible_targets], which_reps)
    # Re-adjust back to original range (for indexinging into rep/move)
    targets = (window_len - 1) + targets * window_inc

    # Keep only selected movement(s)
    if which_moves is not None:
        move_targets = get_idxs(movements[targets], which_moves)
        targets = targets[move_targets]

    valid_inds = []     # save valid window indices
    X_data = np.zeros([targets.shape[0], window_len, nb_channels, 1],
                      dtype=dtype)
    Y_data = np.zeros([targets.shape[0], ], dtype=np.int8)
    R_data = np.zeros([targets.shape[0], ], dtype=np.int8)
    for i, win_end in enumerate(targets):
        win_start = win_end - (window_len - 1)
        if movements[win_start] == movements[win_end]:
            X_data[i, :, :, 0] = emg[win_start:win_end + 1, :]  # Include end
            Y_data[i] = movements[win_end]
            R_data[i] = repetitons[win_end]
            valid_inds.append(i)
    if get_valid_windows:
        X_data = X_data[valid_inds]
        Y_data = Y_data[valid_inds]
        R_data = R_data[valid_inds]

    return X_data, Y_data, R_data


def to_categorical(y, nb_classes=None):
    """Convert a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to nb_classes).
        nb_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.

    Taken from:
    https://github.com/fchollet/keras/blob/master/keras/utils/np_utils.py
    v2.0.2 of Keras to remove unnecessary Keras dependency
    """
    y = np.array(y, dtype='int').ravel()
    if not nb_classes:
        nb_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, nb_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def get_idxs(in_array, to_find):
    """Utility function for finding the positions of observations of one array in another an array.

    Args:
        in_array (array): Array in which to locate elements of to_find
        to_find (array): Array of elements to locate in in_array

    Returns:
        TYPE: Indices of all elements of to_find in in_array
    """
    targets = ([np.where(in_array == x) for x in to_find])
    return np.squeeze(np.concatenate(targets, axis=1))


def split_sets(x_data, y_data, r_data, train_reps, test_reps):
    train_idx = get_idxs(r_data, train_reps)
    x_train = x_data[train_idx]
    test_idx = get_idxs(r_data, test_reps)
    x_test = x_data[test_idx]
    y_train = y_data[train_idx]
    y_test = y_data[test_idx]
    # y_train = to_categorical(y_train)
    return x_train, y_train, x_test, y_test


def balance_labels(emg, labels, max_samples_per_class):
    unique_labels = np.unique(labels)
    balanced_emg, balanced_labels = [], []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if label == 0:  # rest class, downsample it
            selected_indices = np.random.choice(indices, max_samples_per_class, replace=False)
        else:
            selected_indices = indices
        balanced_emg.append(emg[selected_indices])
        balanced_labels.append(labels[selected_indices])
    return np.vstack(balanced_emg), np.hstack(balanced_labels)


def segment_data(emg, labels, window_size=400, step_size=100):
    segmented_emg, segmented_labels = [], []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        label_emg = emg[indices]
        for start in range(0, len(label_emg) - window_size + 1, step_size):
            segmented_emg.append(label_emg[start:start + window_size])
            segmented_labels.append(label)
    return np.array(segmented_emg), np.array(segmented_labels)


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


def select_and_shuffle_samples(emg_train, label_train):
    unique_labels = np.unique(label_train)
    num_samples = emg_train.shape[0] // 10 // len(unique_labels)
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


def Hyser_provide(args):
    cut_length = 200
    sliding = 50

    target_path_1 = 'data/Hyser/subject'+str(args.target_parti)+'_session1.mat'
    target_path_2 = 'data/Hyser/subject'+str(args.target_parti)+'_session2.mat'
    target_mat_1 = scipy.io.loadmat(target_path_1)
    target_mat_2 = scipy.io.loadmat(target_path_2)
    data_1 = target_mat_1['data']
    label_1 = target_mat_1['label']
    data_2 = target_mat_2['data']
    label_2 = target_mat_2['label']

    emg_train_1, label_train_1, emg_val_1, label_val_1, emg_test_1, label_test_1 = hyser_read_mat(data_1, label_1, cut_length, sliding)
    emg_train_2, label_train_2, emg_val_2, label_val_2, emg_test_2, label_test_2 = hyser_read_mat(data_2, label_2, cut_length, sliding)
    emg_train = np.concatenate([emg_train_1, emg_train_2, emg_val_1, emg_val_2], axis=0)
    label_train = np.concatenate([label_train_1, label_train_2, label_val_1, label_val_2], axis=0)

    emg_test = np.concatenate([emg_test_1, emg_test_2], axis=0)
    label_test = np.concatenate([label_test_1, label_test_2], axis=0)
    emg_train, emg_test = normalize_data_2(emg_train, emg_test)
    emg_train, label_train = select_and_shuffle_samples(emg_train, label_train)
    train_data = {'emg': emg_train, 'label': label_train}
    test_data = {'emg': emg_test, 'label': label_test}
    total_data = [train_data, test_data]
    return total_data


def Ninapro_provide(args):
    window_len = 400
    window_inc = 100
    target_path = 'data/Ninapro/DB3_s' + str(args.target_parti)
    data_dict = import_db3(target_path, args.target_parti, rest_length_cap=5)
    reps = np.array(range(1, 7))
    train_reps = np.array([1,2,3,4,5])
    test_reps = np.array([6])
    movements = np.array(range(21))
    emg_data = normalise_emg(data_dict['emg'], data_dict['rep'], train_reps, data_dict['move'], movements)
    x_all, y_all, r_all = get_windows(reps, window_len, window_inc, emg_data, data_dict['move'], data_dict['rep'], which_moves=movements)
    x_train, y_train, x_test, y_test = split_sets(x_all, y_all, r_all, train_reps, test_reps)
    X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    X_train = np.squeeze(X_train, axis=-1)
    X_val = np.squeeze(X_val, axis=-1)
    x_test = np.squeeze(x_test, axis=-1)
    emg_train = [resample_emg(X_train[i], original_fs=2000, target_fs=1000) for i in range(X_train.shape[0])]
    emg_val = [resample_emg(X_val[i], original_fs=2000, target_fs=1000) for i in range(X_val.shape[0])]
    emg_test = [resample_emg(x_test[i], original_fs=2000, target_fs=1000) for i in range(x_test.shape[0])]
    emg_train = np.array(emg_train).transpose((0, 2, 1))
    emg_val = np.array(emg_val).transpose((0, 2, 1))
    emg_test = np.array(emg_test).transpose((0, 2, 1))
    emg_train = filter(emg_train, fs=1000, notch_freq=50)
    emg_val = filter(emg_val, fs=1000, notch_freq=50)
    emg_test = filter(emg_test, fs=1000, notch_freq=50)
    emg_train = np.concatenate([emg_train, emg_val], axis=0)
    label_train = np.concatenate([y_train, y_val], axis=0)
    emg_train, label_train = select_and_shuffle_samples(emg_train, label_train)

    idx_0 = np.where(y_test == 0)[0]
    idx_1 = np.where(y_test == 1)[0]
    idx_rest = np.where((y_test != 0) & (y_test != 1))[0]

    # 下采样 label=0，使其数量等于 label=1 的数量
    min_len = len(idx_1)
    np.random.seed(42)
    idx_0_sampled = np.random.choice(idx_0, min_len, replace=False)

    final_idx = np.concatenate([idx_0_sampled, idx_1, idx_rest])
    np.random.shuffle(final_idx)

    emg_test = emg_test[final_idx]
    y_test = y_test[final_idx]

    train_data = {'emg': emg_train, 'label': label_train}
    test_data = {'emg': emg_test, 'label': y_test}

    total_data = [train_data, test_data]
    return total_data


def Senic_provide(args):
    gesture_mapping = {"rest": 0, "eversion": 1, "fist": 2, "open_hand": 3,
                   "pinch_forefinger": 4, "pinch_middlefinger": 5, "two": 6, "varus": 7,}
    folder_path = "data/Senic/h"+str(args.target_parti)
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
                rest_emg = resample_emg(rest_emg, original_fs=200, target_fs=1000)
                rest_emg = bandpass_filter(rest_emg, 20, 100, 1000, 5)
                rest_emg = notch_filter(rest_emg, 50, 1000)

                if gesture_name in gesture_mapping:
                    gesture = gesture_mapping[gesture_name]
                    gesture_emg = emg_data[400:]  # 取剩余数据
                    gesture_emg = resample_emg(gesture_emg, original_fs=200, target_fs=1000)
                    gesture_emg = bandpass_filter(gesture_emg, 20, 100, 1000, 5)
                    gesture_emg = notch_filter(gesture_emg, 50, 1000)

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

    X_train, X_test = normalize_data_2(X_train, X_test)
    emg_train, label_train = select_and_shuffle_samples(X_train, y_train)

    train_data= {'emg': emg_train, 'label': label_train}
    test_data = {'emg': X_test, 'label': y_test}
    total_data = [train_data, test_data]
    return total_data

