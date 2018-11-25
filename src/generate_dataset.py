import datetime
import os
import wave
import numpy as np
import pandas as pd
import struct
import json
import sys

LAG_THRESHOLD = 3000
NUM_FFTS = 10
AUDIO_FREQ = 44000
RANDOM_STATE = 1234


def audio_fname_to_date(fname):
    date = fname.replace('.wav', '').replace('device1_channel1_', '')
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:8])
    hour = int(date[8:10])
    minute = int(date[10:12])
    second = int(date[12:14])
    date = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
    return date


def get_files(audio_dir, min_date, max_date):
    res = []
    for fname in os.listdir(audio_dir):
        date = audio_fname_to_date(fname)
        if date > max_date:
            continue
        if date < min_date and (min_date - date).seconds / 3600 > 0.33:
            continue
        res.append(fname)
    return res


def load_audio_data(path):
    f = wave.open(path)
    dur = f.getnframes()
    sampwidth = f.getsampwidth()
    fmt = '=' + 'h' * (dur)
    data = f.readframes(dur)
    data = np.array(struct.unpack(fmt, data))
    return data


def process_audio_data(fname):
    audio_data = load_audio_data(os.path.join(AUDIO_DIR, fname))
    cur_date = audio_fname_to_date(fname)
    total_seconds = len(audio_data) // AUDIO_FREQ
    times = []
    for i in range(total_seconds + 1):
        times.extend([cur_date + datetime.timedelta(seconds=i)] * AUDIO_FREQ)
    times = times[:len(audio_data)]
    df = pd.DataFrame({'timestamp': times, 'audio': audio_data})
    df['timestamp'] = df['timestamp'].dt.floor(freq='min')
    df = df.groupby('timestamp').agg(['max', 'min', 'mean', 'median',
                                      agg_quantile(0.25), agg_quantile(0.75), agg_power])
    df.columns = ['_'.join(col) for col in df.columns]
    return df


def get_audio_data(min_date, max_date):
    files = get_files(AUDIO_DIR, min_date, max_date)
    result_df = []
    for fname in files:
        result_df.append(process_audio_data(fname))
    pd.concat(result_df)
    return result_df


def load_motion_dataset(path):
    dataset = []
    for fname in os.listdir(path):
        df = get_motion_df(os.path.join(MOTION_DIR, fname))
        dataset.append(df)
    dataset = pd.concat(dataset)
    return dataset


def fname_to_date(fname):
    date = fname.replace('.json', '')
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:8])
    hour = int(date[9:11])
    date = datetime.datetime(year=year, month=month, day=day, hour=hour)
    return date


def timestamp_to_date(ts):
    return datetime.datetime.utcfromtimestamp(ts / 1000)


def agg_fft(series):
    skip_columns = ['timestamp', 'variable']

    result = []
    for column in series.columns:
        if column in skip_columns:
            continue

        ffts = np.abs(np.fft.fft(series[column]))[:NUM_FFTS]
        cols = ['%s_fft%d' % (column, i) for i in range(NUM_FFTS)]
        res = pd.Series(ffts, index=cols)
        result.append(res)
    result = pd.concat(result)
    return result


def agg_quantile(n):
    def quantile_(x):
        return np.quantile(x, n)

    quantile_.__name__ = 'quantile_%s' % n
    return quantile_


def agg_power(series):
    return (np.sum(series ** 2) / len(series)) ** 0.5


agg_power.__name__ = 'power'


def get_motion_df(path):
    data = json.load(open(path, 'r'))['data']

    acc_df = []
    quat_df = []
    for row in data:
        d_row = {}
        event = row['event']
        content = event['content']
        variable = event['variable']

        d_row['timestamp'] = timestamp_to_date(row['timestamp'])
        d_row['variable'] = variable.replace('.acceleration', '') \
            .replace('.quaternion', '') \
            .replace('.batter', '')

        if 'acceleration' in variable:
            if len(content) != 4:
                print('FUCK acceleration')
            d_row['x'] = content[0]
            d_row['y'] = content[1]
            d_row['z'] = content[2]
            # d_row['acc_shit'] = content[3]
            acc_df.append(d_row)
        elif 'quaternion' in variable:
            if len(content) != 5:
                print('FUCK quaternion')
            d_row['q1'] = content[0]
            d_row['q2'] = content[1]
            d_row['q3'] = content[2]
            d_row['q4'] = content[3]
            # d_row['q_shit'] = content[4]
            quat_df.append(d_row)
        elif 'battery' in variable:
            continue
        else:
            print("FUCK else")

    acc_df = pd.DataFrame.from_dict(acc_df)
    quat_df = pd.DataFrame.from_dict(quat_df)
    return acc_df, quat_df


def generate_features(df, freq='15S'):
    df['timestamp'] = df['timestamp'].dt.floor(freq=freq)
    fft_df = df.groupby(['variable', 'timestamp']).apply(agg_fft)
    agg_df = df.groupby(['variable', 'timestamp']).agg(['min', 'max', 'mean', 'std', 'median', agg_power,
                                                        agg_quantile(0.05),
                                                        agg_quantile(0.25),
                                                        agg_quantile(0.75),
                                                        agg_quantile(0.95),
                                                        agg_quantile(0.99)])
    agg_df = agg_df.join(fft_df)
    agg_df.columns = [column if type(column) is str else '_'.join(column) for column in agg_df.columns]
    return agg_df


def add_labels(df, label_df):
    df['status'] = 'unk'
    df = df.reset_index()
    min_t = df.timestamp.min()
    max_t = df.timestamp.max()
    for i, row in label_df.iterrows():
        df.loc[(df['timestamp'] >= row['start']) &
               (df['timestamp'] <= row['stop']), 'status'] = row['status']
    return df


def status_to_color(status):
    if status == 'Good':
        return 'green'
    if status == 'Stopped':
        return 'yellow'
    if status == 'Failure':
        return 'red'
    raise Exception("Unknown status")


def filter_unk_status(df):
    df = df[df['status'] != 'unk']
    return df.reset_index(drop=True)


def get_dataset(data_path, label_df, start_file_date=None, end_file_date=None, extract_features=True):
    result_acc = []
    result_quat = []
    for fname in os.listdir(data_path):
        date = fname_to_date(fname)
        if start_file_date is not None and date < start_file_date:
            continue
        if end_file_date is not None and date > end_file_date:
            break
        label_ind = None
        for i, row in label_df.iterrows():
            if date >= row['start'] and date <= row['stop']:
                label_ind = i
                break
        if label_ind == None:
            continue
        acc_df, quat_df = get_motion_df(os.path.join(data_path, fname))
        if extract_features:
            acc_df = generate_features(acc_df)
            quat_df = generate_features(quat_df)

        acc_df = add_labels(acc_df, label_df)
        quat_df = add_labels(quat_df, label_df)

        acc_df = filter_unk_status(acc_df)
        quat_df = filter_unk_status(quat_df)

        result_acc.append(acc_df)
        result_quat.append(quat_df)

    result_acc = pd.concat(result_acc)
    result_quat = pd.concat(result_quat)

    pepa_acc_df, salt_acc_df = split_by_device(result_acc)
    pepa_quat_df, salt_quat_df = split_by_device(result_quat)

    pepa_acc_df = pepa_acc_df.set_index(['timestamp', 'status'])
    pepa_quat_df = pepa_quat_df.set_index(['timestamp', 'status'])
    salt_acc_df = salt_acc_df.set_index(['timestamp', 'status'])
    salt_quat_df = salt_quat_df.set_index(['timestamp', 'status'])

    pepa_acc_df = pepa_acc_df.drop(columns='variable')
    pepa_quat_df = pepa_quat_df.drop(columns='variable')
    salt_acc_df = salt_acc_df.drop(columns='variable')
    salt_quat_df = salt_quat_df.drop(columns='variable')

    full_df = pepa_acc_df.join(pepa_quat_df, lsuffix='_pepa_acc', rsuffix='_pepa_quat') \
        .join(salt_acc_df, rsuffix='_salt_acc') \
        .join(salt_quat_df, rsuffix='_salt_quat')
    full_df = full_df.reset_index()
    full_df.loc[full_df['status'] == 'Stopped', 'status'] = 'Good'

    # TODO: AUDIO
    return full_df


def split_by_device(df):
    pepa_df = df[df['variable'] == 'NGIMU-PEPA']
    salt_df = df[df['variable'] == 'NGIMU-SALT']
    return pepa_df, salt_df


def get_labels(path):
    label_df = pd.read_excel(os.path.join(path, 'IHearVoicesDataPrep.xlsx'), header=1)
    label_df = label_df.iloc[:, :8]
    label_df = label_df[:29]
    label_df = label_df.drop(columns=['Date', 'Date.1', 'Time', 'Time.1'])
    label_df = label_df.reset_index(drop=True)
    label_df.columns = ['status', 'start', 'stop', 'hours']
    return label_df


if __name__ == "__main__":
    # check the number of arguments
    if len(sys.argv) != 3:
        print("Usage: generate_dataset <input_dir> <output_dir>", file=sys.stderr)
        exit(-1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    MOTION_DIR = os.path.join(input_folder, 'Motion')
    AUDIO_DIR = os.path.join(input_folder, 'Audio')

    start_date = datetime.datetime(year=2018, month=10, day=20, hour=23)
    end_date = datetime.datetime(year=2018, month=10, day=26, hour=10)
    label_df = get_labels(input_folder)
    full_df = get_dataset(MOTION_DIR, label_df, start_file_date=start_date, end_file_date=end_date)
    full_df.to_csv(os.path.join(output_folder, "full_df.csv"), index=False)
