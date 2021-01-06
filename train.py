import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from models import build_cnn, build_seq2seq
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import load_dataset, plot_confusion_matrix, print_results

lead = 'MLII'

# Using data from chosen lead
train_dataset = load_dataset()['train_set'][lead]
test_dataset = load_dataset()['test_set'][lead]

train_x = np.array([np.expand_dims(sample[0], axis=1) for sample in train_dataset])
train_y = np.array([np.array(sample[1]) for sample in train_dataset])

x_train, x_val, y_train, y_val = train_test_split(train_x, train_y, 
                                                  test_size=0.2, shuffle=True)

test_x = np.array([np.expand_dims(sample[0], axis=1) for sample in test_dataset])
test_y = np.array([np.array(sample[1]) for sample in test_dataset])

x_test, y_test = shuffle(test_x, test_y)

model = build_cnn()

callbacks = [EarlyStopping(patience=10, verbose=1),
             ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.01, verbose=1),
             ModelCheckpoint('models/{}-latest.hdf5'.format(lead), 
                             monitor='val_loss', 
                             save_best_only=True, 
                             mode='min', 
                             verbose=1)]

model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          epochs=30,
          batch_size=256,
          callbacks=callbacks)

y_pred = model.predict(x_test)
plot_confusion_matrix(y_test, y_pred)
print_results(y_test, y_pred)