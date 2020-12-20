from keras import Input
from keras import backend as K
from keras.models import Model
from keras.layers import Conv1D, Dense, Add, Flatten, Dropout, MaxPooling1D, Activation, BatchNormalization, Lambda
from keras.optimizers import Adam

# 34-layer CNN from Stanford paper
def build_cnn():
    def Conv_1D(strides):
        return Conv1D(filters=32,
                      kernel_size=16,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal')

    def input_block(input):
        layer = Conv_1D(1)(input)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        shortcut = MaxPooling1D(pool_size=1, strides=1)(layer)

        layer = Conv_1D(1)(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(0.2)(layer)
        layer = Conv_1D(1)(layer)

        return Add()([shortcut, layer])

    def hidden_blocks(layer):
        def zeropad(x):
            y = K.zeros_like(x)
            return K.concatenate([x, y], axis=1)

        def zeropad_output_shape(input_shape):
            shape = list(input_shape)
            shape[1] *= 2
            return tuple(shape)

        filter_length = 32
        n_blocks = 15
        for block_index in range(n_blocks):
            subsample_length = 2 if block_index % 2 == 0 else 1

            shortcut = MaxPooling1D(pool_size=subsample_length)(layer)

            if block_index % 4 == 0 and block_index > 0:
                shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
                filter_length *= 2

            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Conv1D(filters=filter_length,
                           kernel_size=16,
                           padding='same',
                           strides=subsample_length,
                           kernel_initializer='he_normal')(layer)

            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(0.2)(layer)
            layer = Conv1D(filters=filter_length,
                           kernel_size=16,
                           padding='same',
                           strides=1,
                           kernel_initializer='he_normal')(layer)

            layer = Add()([shortcut, layer])

        return layer

    def output_block(layer):
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        outputs = Dense(len_classes, activation='softmax')(layer)
        model = Model(inputs=inputs, outputs=outputs)

        adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999,
                    epsilon=None, decay=0.0, amsgrad=False)

        model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()

        return model

    len_classes = 5

    input = Input(shape=(100, 1), name='input')
    layer = input_block(input)
    layer = hidden_blocks(layer)
    model = output_block(layer)

    return model
