import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential


class DataGenerator(keras.utils.Sequence):

    def __init__(self, dim, batch_size, labels, list_IDs,
                 n_channels, n_classes, shuffle):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        'updating indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates the batches of data'
        # X: (n_samples, dims, channels)

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size,), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            X[i,] = np.load(f'data/{ID}.npy')
            Y[i,] = self.labels[ID]

        return X, keras.utils.to_categorical(Y, num_classes=self.n_classes)

    def __len__(self):
        'Denotes the number of bathces per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y


if __name__ == '__main__':
    # Parameters
    params = {'dim': (32, 32, 32),
              'batch_size': 64,
              'n_classes': 6,
              'n_channels': 1,
              'shuffle': True}

    # Datasets
    partition = {'train': ['1', '2'], 'validation': ['3']}  # IDs
    labels = {'1': 0, '2': 0, '3': 1}  # Labels

    # Generators
    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)

    # Design model
    model = Sequential()
    [...]  # Architecture
    model.compile()

    # Train model on dataset
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=6)
