import numpy as np
import os
import wfdb

from os.path import join
from sklearn import preprocessing
from utils import smooth_signal, baseline_correct, store_dataset

DATA_PATH = 'physionet.org/files/mitdb/1.0.0'

patients = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115', '116', '117', '118', '119', '121', '122', '123', '124',
            '200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234']
testset = ['101', '105', '114', '118', '124', '201', '210', '217']
trainset = [x for x in patients if x not in testset]

leads = ['MLII', 'V1', 'V2', 'V4', 'V5']
labels = ['N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'F', 'e', 'j', 'E', '/', 'f', 'Q']
aami_labels = ['N', 'S', 'V', 'F', 'Q']

# Conversion of MIT-BIH labels to AAMI labels
aami = {'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
        'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
        'V': 'V', 'E': 'V',
        'F': 'F',
        '/': 'Q', 'f': 'Q', 'Q': 'Q'}

'''
Store ECG samples and labels from each lead in dictionary.

{ lead_1 : [[sample_1, label_1],
            [sampel_2, label_2],
            ...
           ],

  lead_2 : [[sample_1, label_1],
            [sampel_2, label_2],
            ...
           ],
  ...           
}
'''

dic = {}
for lead in leads:
    dic[lead] = []

# Process ECG signal from each patient using wfdb library
for patient in patients:
    path = os.path.join(DATA_PATH, patient)

    record = wfdb.rdrecord(path)
    samp = wfdb.rdsamp(path)
    ann = wfdb.rdann(path, extension='atr')

    # Extract leads used to record ECG signal
    # 2 leads are used for each patient
    lead_1 = record.sig_name[0]
    lead_2 = record.sig_name[1]

    # Extract signals from the record and normalise
    signals_1 = preprocessing.scale(record.p_signal[:, 0])
    signals_2 = preprocessing.scale(record.p_signal[:, 1])

    # Extract QRS peaks and labels from the annotation
    # Each label corresponds to one QRS peak
    for i in range(len(ann.symbol)):
        label = ann.symbol[i]

        # Filter 85% of N label to balance data
        if label == 'N' and np.random.random() > 0.15:
            continue

        if label in labels:
            # One-hot encode label
            aami_label = [int(aami[label] == label) for label in aami_labels]
            peak = ann.sample[i]

            # Extract signal of window size 100 around peak
            raw_sample_1 = signals_1[peak - 50: peak + 50]
            raw_sample_2 = signals_2[peak - 50: peak + 50]

            # Smooth and baseline correct signal
            # Store signal with its label in dictionary
            if len(raw_sample_1) == 100:
                sample_1 = baseline_correct(smooth_signal(raw_sample_1))
                dic[lead_1].append([sample_1, aami_label])

            if len(raw_sample_2) == 100:
                sample_2 = baseline_correct(smooth_signal(raw_sample_2))
                dic[lead_2].append([sample_2, aami_label])

store_dataset(dic)