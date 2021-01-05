from keras import Input
from keras import backend as K
from keras.models import Model
from keras.layers import Conv1D, Dense, Add, Flatten, Dropout, MaxPooling1D, Activation, BatchNormalization, Lambda
from keras.optimizers import Adam

# 34-layer CNN from Stanford paper
def build_cnn():
    def Conv_1D(filters, strides):
        return Conv1D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal')

    def first_conv_layer(input):
        layer = Conv_1D(num_filters, 1)(input)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        return layer

    def looped_residual_layers(layer, num_filters):
        def zeropad(x):
            y = K.zeros_like(x)
            return K.concatenate([x, y], axis=2)

        def zeropad_output_shape(input_shape):
            shape = list(input_shape)
            shape[1] *= 2
            return tuple(shape)

        for index in range(1, num_loops + 1):
            is_subsample = (index % subsample_freq == 0)
            strides = 2 if is_subsample else 1

            is_increase_filters = (index % increase_filters_freq == 0)
            num_filters *= 2 if is_increase_filters else 1

            shortcut = MaxPooling1D(pool_size=strides)(layer)
            if is_increase_filters:
                shortcut = Lambda(
                    zeropad, output_shape=zeropad_output_shape)(shortcut)

            if index > 1:
                layer = BatchNormalization()(layer)
                layer = Activation('relu')(layer)

            layer = Conv_1D(filters=num_filters,
                            strides=strides)(layer)

            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(dropout)(layer)
            layer = Conv_1D(filters=num_filters,
                            strides=1)(layer)

            layer = Add()([shortcut, layer])

        return layer

    def output_layer(layer):
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        layer = Flatten()(layer)
        #outputs = TimeDistributed(Dense(num_classes, activation='softmax'))(layer)
        outputs = Dense(num_classes, activation='softmax')(layer)
        print(outputs.shape)

        return outputs

    input_shape = (256, 1)

    num_classes = 5

    num_filters = 32
    kernel_size = 16
    dropout = 0.2

    # Number of residual block loops
    num_loops = 16

    # Subsample input by factor of 2 every alternate residual block
    subsample_freq = 2

    # Double number of filters every 4 residual block
    increase_filters_freq = 4

    inputs = Input(shape=input_shape, name='inputs')
    layer = first_conv_layer(inputs)
    layer = looped_residual_layers(layer, num_filters)
    outputs = output_layer(layer)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(lr=0.1, beta_1=0.9, beta_2=0.999,
                     epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.summary()

    return model
