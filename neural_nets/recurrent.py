from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.models import Sequential
from numpy import unique, array


class GRUClassificationNN:
    def __init__(
        self,
        embed_units: int = 128,
        activation: str = "sigmoid",
        optimizer: str = "adam",
        loss="binary_crossentropy",
    ):
        self.embed_units = embed_units
        self.gru_units = embed_units
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss

    def _build(self, num_features, y):
        """Helper function to build out model, makes it easy to make adjustments as necessary"""
        self.model = Sequential()
        self.model.add(Embedding(input_dim=num_features, output_dim=self.embed_units))
        self.model.add(GRU(units=self.gru_units))
        self.model.add(Dense(units=1))

    def train(
        self,
        X: array,
        y: array,
        num_features: int,
        epochs: int = 10,
        batch_size: int = 32,
    ):
        """Compile and train model"""
        self._build(num_features, y)
        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=["accuracy"]
        )
        self.model.fit(x=X, y=y, epochs=epochs, batch_size=batch_size)

    def predict_proba(self, X):
        """returns probabilities of belonging to each class"""
        return self.model.predict(x=X)

    def predict(self, X, threshold):
        return array(
            [[1] if item > threshold else [0] for item in self.model.predict(x=X)]
        )
