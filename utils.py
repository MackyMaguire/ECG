import bz2
import _pickle as cPickle

from BaselineRemoval import BaselineRemoval
from scipy.signal import savgol_filter
from sklearn.metrics import classification_report, confusion_matrix

def smooth_signal(signal):
    window_len = 11
    order = 3

    smoothed = savgol_filter(signal, window_len, order)

    #weights = np.hanning(window_len)
    #smoothed = np.convolve(weights/weights.sum(), signal, mode='valid')

    return smoothed


def baseline_correct(signal):
    degree = 2

    corrected = BaselineRemoval(signal).IModPoly(3)

    return corrected

def store_dataset(dataset):
    with bz2.BZ2File('dataset.pbz2', 'w') as f:
        cPickle.dump(dataset, f)

def load_dataset():
    with bz2.BZ2File('dataset.pbz2', 'rb') as f:
        dataset = cPickle.load(f)
    return dataset

def plot_confusion_matrix(y_true, y_pred):
    labels = ['N', 'S', 'V', 'F', 'Q']

    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=None,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

def print_results(y_true, y_pred):
    labels = ['N', 'S', 'V', 'F', 'Q']

    print('Classification Report')
    print(classification_report(y_true, y_pred, target_names=labels))