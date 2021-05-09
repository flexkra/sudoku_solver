import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.np_utils import to_categorical

from model.DigitClassifierModel import DigitClassifier

def train_model(filename="digit_classifier", debug=False):
    num_classes = 10
    input_shape = (28,28,1)
    epochs = 10
    batch_size = 128

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model = DigitClassifier.build(input_shape, num_classes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=.1, verbose=int(debug))
    model.save(f"{filename}.h5", save_format='h5')

    if debug:
        model.evaluate(x_test, y_test, verbose=int(debug), batch_size=batch_size)

    return model