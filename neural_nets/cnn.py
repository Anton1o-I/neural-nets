from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from numpy import array


class ConvolutionalNN:
    def __init__(
        self,
        conv_layers: list = [64, 32],
        class_layers: list = [100, 50],
        pooling: dict = None,  # {"pool_size":(2,2), "stride":2},
        loss: str = "categorical_crossentropy",
        optimizer: str = "adam",
    ):
        """
        Initializes a ConvolutionalNN object.

        conv_layers: List of the number of nodes per convolution layer
        class_layers: List of number of nodes per layer in the classification phase
        pooling: Dictionary {"pool_size":typle, "stride":int} if provided adds a
        pooling layer with specified parameters after each convolution
        loss: Specifies the loss function to use
        optimizer: Specifies the optimizer to use
        """

        self.conv_layers = conv_layers
        self.class_layers = class_layers
        self.pooling = pooling
        self.loss = loss
        self.optimizer = optimizer

    def _build_nn(self, X):
        """
        Builds out the layers in the neural network, helper function,
        not supposed to be directly called
        """

        self.model = Sequential()
        for i, item in enumerate(self.conv_layers):
            if i == 0:
                self.model.add(
                    Conv2D(
                        filters=item,
                        kernel_size=3,
                        activation="relu",
                        input_shape=(28, 28, 1),
                    )
                )
            else:
                self.model.add(Conv2D(filters=item, kernel_size=3, activation="relu"))
            if self.pooling:
                self.model.add(
                    MaxPooling2D(
                        pool_size=self.pooling["pool_size"],
                        strides=self.pooling["stride"],
                    )
                )
        self.model.add(Flatten())
        for item in self.class_layers:
            self.model.add(Dense(units=item, activation="sigmoid"))

    def train(self, X: array, y: array, epochs: int = 2, batch_size: int = 32):
        """Trains the neural network

        Parameters:
        X (array): Array shape needs to match input_shape, change code above if necessary
        y (array): Dummied labels for classification
        epochs (int): Number of epochs to run
        batch_size (int): Size of batch to use to calculate gradient
        """

        self._build_nn(X)
        self.model.add(Dense(units=y.shape[1], activation="softmax"))
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=["categorical_accuracy"]
        )
        self.model.fit(x=X, y=y, epochs=epochs, batch_size=batch_size)

    def predict_proba(self, X: array):
        """Returns raw probability values per class"""
        return self.model.predict(x=X).round(4)

    def predict(self, X: array):
        """Takes argmax of predict_proba and returns 1 for most likely, 0 else"""
        preds = self.model.predict(X).round(4)
        for obs in preds:
            max_loc = obs.argmax()
            for i, _ in enumerate(obs):
                if i == max_loc:
                    obs[i] = 1
                else:
                    obs[i] = 0
        return preds

