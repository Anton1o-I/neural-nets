from keras.models import Sequential
from keras.layers import Dense, Dropout
from numpy import array


class ClassificationNN:
    def __init__(
        self,
        input_dim: int,
        layers: list,
        epochs: int = 100,
        batch_size: int = 32,
        dropout: float = None,
        optimizer: str = "rmsprop",
        activation_fn: str = "sigmoid",
        loss: str = "binary_crossentropy",
    ):
        self.input_dim = input_dim
        if all([type(item) != int for item in layers]):
            raise ValueError("Number of nodes in each layer must be integers")
        self.layers = layers
        self.optimizer = optimizer
        self.activation_fn = activation_fn
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout

    def _build(self):
        self.model = Sequential()
        for i, item in enumerate(self.layers):
            self.model.add(
                Dense(
                    item,
                    input_shape=(self.input_dim if i == 0 else None,),
                    activation=self.activation_fn,
                )
            )
            if self.dropout:
                self.model.add(Dropout(self.dropout))
        self.model.add(Dense(1, activation=self.activation_fn))

    def train(self, X, y):
        self._build()
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"]
        )
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)
        return self

    def predict(self, X, threshold: float = 0.5):
        if threshold <= 0 or threshold >= 1:
            raise ValueError(
                f"Threshold must be a value between 0 and 1, given value {threshold}"
            )
        pred = self.model.predict(X)
        values = array([1 if item > threshold else 0 for item in pred])
        return values

    def predict_proba(self, X):
        return self.model.predict(X)
