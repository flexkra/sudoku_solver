from tensorflow.python.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.models import Sequential

class DigitClassifier:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()

        #first conv layer
        model.add(Conv2D(32, (3,3), input_shape=input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #second conv layer
        model.add(Conv2D(64, (3,3), input_shape=input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #first dense layer
        model.add(Flatten())
        model.add(Dropout(.5))
        model.add(Dense(classes, activation='softmax'))

        return model