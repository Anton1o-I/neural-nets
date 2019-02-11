from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from pandas import DataFrame, Series


class ConvolutionalNN:
    def __init__(
        self,
        conv_layers: list = [64, 32],
        class_layers: list = [100, 50],
        pooling: dict = None,  # {"pool_size":(2,2), "stride":2},
        loss: str = "categorical_crossentropy",
        optimizer: str = "adam",
    ):
        self.conv_layers = conv_layers
        self.class_layers = class_layers
        self.pooling = pooling
        self.loss = loss
        self.optimizer = optimizer

    def _build_nn(self, X):
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

    def train(self, X: DataFrame, y: DataFrame, epochs: int = 2, batch_size: int = 32):
        self._build_nn(X)
        self.model.add(Dense(units=y.shape[1], activation="softmax"))
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=["categorical_accuracy"]
        )
        self.model.fit(x=X, y=y, epochs=epochs, batch_size=batch_size)

    def predict_proba(self, X: DataFrame):
        return self.model.predict(x=X).round(4)

    def predict(self, X: DataFrame):
        preds = self.model.predict(X).round(4)
        for obs in preds:
            max_loc = obs.argmax()
            for i, _ in enumerate(obs):
                if i == max_loc:
                    obs[i] = 1
                else:
                    obs[i] = 0
        return preds

