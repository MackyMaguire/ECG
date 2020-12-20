import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from models import build_cnn, build_seq2seq
from sklearn.model_selection import train_test_split
from utils import load_dataset, plot_confusion_matrix, print_results

# Using data from lead MLII
dataset = load_dataset()['MLII']

x = np.array([sample[0] for sample in dataset])
y = np.array([np.array(sample[1]) for sample in dataset])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = create_cnn()

callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.01, verbose=1),
        ModelCheckpoint('models/{}-latest.hdf5'.format(config.feature), monitor='val_loss', save_best_only=False, verbose=1, period=10)
]

model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          epochs=50,
          batch_size=256,
          callbacks=callbacks)

y_pred = model.predict(x_test)
plot_confusion_matrix(y_test, y_pred)
print_results(y_test, y_pred)