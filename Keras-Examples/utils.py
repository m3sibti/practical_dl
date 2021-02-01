from tensorflow.keras import models
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np


class KerasExperiment:

    def __init__(self):
        pass

    def build_model(self, layers):
        layers_list = []
        for layer in layers:
            layers_list.append(layer)
        model = models.Sequential(layers_list)
        return model

    def plot_loss(self, i=None, epochs=None, hist=None):
        plt.title(f'Model: {i + 1}')
        ep = [i for i in range(epochs)]
        plt.plot(ep, hist.history['loss'])
        plt.plot(ep, hist.history['val_loss'], 'o')
        plt.show()

    def compile_fit(self, i=0, model=None, c_params=None,
                    train_data=None,
                    val_data=None,
                    epochs=0, batch_size=10):
        model.compile(c_params[0], c_params[1], c_params[2])
        hist = model.fit(
            train_data[0],
            train_data[1],
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
        )
        self.plot_loss(i=i, epochs=epochs, hist=hist)

    def build_dense_models(self, param_lists, compile_params, train_data, val_data):
        assert len(param_lists) == len(compile_params), 'Compile Params must equal to Model params'
        models = []
        for param_list in param_lists:
            model_params = [Dense(pli[0], activation=pli[1]) for pli in param_list]
            models.append(self.build_model(model_params))
        for i, (m, c) in enumerate(zip(models, compile_params)):
            self.compile_fit(i=i,
                             model=m, c_params=c,
                             train_data=train_data, val_data=val_data,
                             epochs=10, batch_size=512
                             )
        # print(models[0].summary())


def plot_losses(hist, epochs=0):
    ep = range(epochs)
    plt.plot(ep, hist.history['loss'], label='Training Loss')
    plt.plot(ep, hist.history['val_loss'], 'o', label='Validation Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    xp_train, yp_train = np.random.normal(size=1000, ), np.zeros(shape=(1000,))
    x_val, y_val = np.random.normal(size=500, ), np.zeros(shape=(500,))

    kE = KerasExperiment()

    kE.build_dense_models(
        [
            [(16, 'relu'), (32, 'relu'), (1, 'sigmoid')],
            [(14, 'relu'), (1, 'sigmoid'), ],
            [(32, 'relu'), (10, 'tanh'), (1, 'sigmoid')],
        ],
        [
            ['rmsprop', 'binary_crossentropy', None],
            ['rmsprop', 'binary_crossentropy', None],
            ['rmsprop', 'binary_crossentropy', None],
        ],
        [xp_train, yp_train],
        (x_val, y_val),
    )
