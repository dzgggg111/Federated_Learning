
from custom_logger import CustomLogger
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, TimeDistributed, LSTM, BatchNormalization
from keras.layers import AveragePooling2D, Input
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
import time

import tensorflow as tf
import csv

class GlobalModel(object):
    """docstring for GlobalModel"""
    def __init__(self):
        self.model = self.build_model()
        self.current_weights = self.model.get_weights()
        self.logger = CustomLogger("info", "SERVER MODEL")
        
        self.prev_train_loss = []
        self.prev_valid_loss = []

        self.prev_train_acc = []
        self.prev_valid_acc = []

        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []

        self.training_start_time = int(round(time.time()))

        self.temp_weights = []
        self.Updates = []
        self.Norms = []
        self.median = []
        self.ClippedUpdates = []
        self.m = 0.0
        self.Sigma = 1.5
        self.eps = 8
        self.bound = 0.001
        self.delta_accountant = []

        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)
        
        self.graph = tf.get_default_graph()
        self.logger.debug("Initialized Server-side Model.")

    def build_model(self):
        raise NotImplementedError()

    def set_weights(self):
        with self.graph.as_default():
            self.model.set_weights(self.current_weights)

    def evaluate(self, x, y):
        self.logger.debug("Evaluating Local Model...")
        score = self.model.evaluate(x, y, verbose=0)
        self.logger.debug("Accuracy/Loss: {}/{}".format(score[1], score[0]))
        return score
    
    def predict(self, x):
        self.logger.debug("Predicting using local model...")
        predicted_y = self.model.predict(x)
        self.logger.debug("..done")
        return predicted_y

    # client_updates = [(w, n)..]
    def update_weights(self, client_weights, client_sizes):
        self.logger.debug("Updating weights from newly recieved client weights...")
        new_weights = [np.zeros(w.shape) for w in self.current_weights]
        total_size = np.sum(client_sizes)
        update_ok = True
        for c in range(len(client_weights)):
            for i in range(len(new_weights)):
                try:
                    new_weights[i] += client_weights[c][i] * client_sizes[c] / total_size
                except Exception as e:
                    self.logger.err("Problem while updating weights from client to local model")
                    self.logger.err(e)
                    self.logger.err("Ignoring this set of client weights and using the next set.")
                    continue
        self.current_weights = new_weights
        self.logger.debug("..done updating with new weights.")

    def aggregate_loss_accuracy(self, client_losses, client_accuracies, client_sizes):
        total_size = np.sum(client_sizes)
        aggr_loss = np.sum(client_losses[i] / total_size * client_sizes[i]
                for i in range(len(client_sizes)))
        aggr_accuraries = np.sum(client_accuracies[i] / total_size * client_sizes[i]
                for i in range(len(client_sizes)))
        return aggr_loss, aggr_accuraries

    def aggregate_train_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        return aggr_loss, aggr_accuraries

    # cur_round coule be None
    def aggregate_valid_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        return aggr_loss, aggr_accuraries

    def get_stats(self):
        return {
            "train_loss": self.train_losses,
            "valid_loss": self.valid_losses,
            "train_accuracy": self.train_accuracies,
            "valid_accuracy": self.valid_accuracies
        }


class GlobalModel_MNIST_MLP(GlobalModel):

    def __init__(self):
        super(GlobalModel_MNIST_MLP, self).__init__()

    def build_model(self):

        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(784,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='softmax'))

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                    optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
                    metrics=['accuracy'])
        return model


class GlobalModel_MNIST_CNN(GlobalModel):
    def __init__(self):
        super(GlobalModel_MNIST_CNN, self).__init__()

    def build_model(self):
        # ~5MB worth of parameters
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(28, 28, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model


class GlobalModel_CIFAR10_CNN(GlobalModel):
    def __init__(self):
        super(GlobalModel_CIFAR10_CNN, self).__init__()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=(32, 32, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        return model


class GlobalModel_MNIST_RNN(GlobalModel):

    def __init__(self):
        super(GlobalModel_MNIST_RNN, self).__init__()

    def build_model(self):
        x = Input(shape=(28, 28, 1))
        encoded_rows = TimeDistributed(LSTM(16))(x)
        encoded_columns = LSTM(16)(encoded_rows)
        prediction = Dense(10, activation='softmax')(encoded_columns)
        model = Model(x, prediction)
        model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])
        return model



# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
class GlobalModel_CIFAR10_RESNET20_V2(GlobalModel):

    def __init__(self):
        super(GlobalModel_CIFAR10_RESNET20_V2, self).__init__()

    def build_model(self):

        n = 3 # See chart above
        input_shape = (32, 32, 3) # Cifar10 Image size
        depth = n * 9 + 2

        def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
            """2D Convolution-Batch Normalization-Activation stack builder

            # Arguments
                inputs (tensor): input tensor from input image or previous layer
                num_filters (int): Conv2D number of filters
                kernel_size (int): Conv2D square kernel dimensions
                strides (int): Conv2D square stride dimensions
                activation (string): activation name
                batch_normalization (bool): whether to include batch normalization
                conv_first (bool): conv-bn-activation (True) or
                    bn-activation-conv (False)

            # Returns
                x (tensor): tensor as input to the next layer
            """
            conv = Conv2D(num_filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))

            x = inputs
            if conv_first:
                x = conv(x)
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
            else:
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
                x = conv(x)
            return x

        def resnet_v2(input_shape, depth, num_classes=10):
            """ResNet Version 2 Model builder [b]

            Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
            bottleneck layer
            First shortcut connection per layer is 1 x 1 Conv2D.
            Second and onwards shortcut connection is identity.
            At the beginning of each stage, the feature map size is halved (downsampled)
            by a convolutional layer with strides=2, while the number of filter maps is
            doubled. Within each stage, the layers have the same number filters and the
            same filter map sizes.
            Features maps sizes:
            conv1  : 32x32,  16
            stage 0: 32x32,  64
            stage 1: 16x16, 128
            stage 2:  8x8,  256

            # Arguments
                input_shape (tensor): shape of input image tensor
                depth (int): number of core convolutional layers
                num_classes (int): number of classes (CIFAR10 has 10)

            # Returns
                model (Model): Keras model instance
            """
            if (depth - 2) % 9 != 0:
                raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
            # Start model definition.
            num_filters_in = 16
            num_res_blocks = int((depth - 2) / 9)

            inputs = Input(shape=input_shape)
            # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
            x = resnet_layer(inputs=inputs,
                             num_filters=num_filters_in,
                             conv_first=True)

            # Instantiate the stack of residual units
            for stage in range(3):
                for res_block in range(num_res_blocks):
                    activation = 'relu'
                    batch_normalization = True
                    strides = 1
                    if stage == 0:
                        num_filters_out = num_filters_in * 4
                        if res_block == 0:  # first layer and first stage
                            activation = None
                            batch_normalization = False
                    else:
                        num_filters_out = num_filters_in * 2
                        if res_block == 0:  # first layer but not first stage
                            strides = 2    # downsample

                    # bottleneck residual unit
                    y = resnet_layer(inputs=x,
                                     num_filters=num_filters_in,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=activation,
                                     batch_normalization=batch_normalization,
                                     conv_first=False)
                    y = resnet_layer(inputs=y,
                                     num_filters=num_filters_in,
                                     conv_first=False)
                    y = resnet_layer(inputs=y,
                                     num_filters=num_filters_out,
                                     kernel_size=1,
                                     conv_first=False)
                    if res_block == 0:
                        # linear projection residual shortcut connection to match
                        # changed dims
                        x = resnet_layer(inputs=x,
                                         num_filters=num_filters_out,
                                         kernel_size=1,
                                         strides=strides,
                                         activation=None,
                                         batch_normalization=False)
                    x = keras.layers.add([x, y])

                num_filters_in = num_filters_out

            # Add classifier on top.
            # v2 has BN-ReLU before Pooling
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = AveragePooling2D(pool_size=8)(x)
            y = Flatten()(x)
            outputs = Dense(num_classes,
                            activation='softmax',
                            kernel_initializer='he_normal')(y)

            # Instantiate model.
            model = Model(inputs=inputs, outputs=outputs)
            
            # initiate RMSprop optimizer
            opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

            # Let's train the model using RMSprop
            model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])
            return model
        
        model = resnet_v2(input_shape, depth)
        return model


# All Available Models
global_models = {
    "cifar10_cnn": GlobalModel_CIFAR10_CNN,
    "mnist_rnn"  : GlobalModel_MNIST_RNN,
    "mnist_cnn"  : GlobalModel_MNIST_CNN,
    "mnist_mlp"  : GlobalModel_MNIST_MLP,
    "cifar10_resnet20_v2": GlobalModel_CIFAR10_RESNET20_V2
}
