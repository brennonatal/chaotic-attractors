from tensorflow.keras import Model, Sequential, layers
import os
import sys

sys.path.append(os.getcwd())
from GAN.utils import make_net


class Embedder(Model):
    def __init__(self, hidden_dim, sequence_length, net_type="GRU"):
        super(Embedder, self).__init__()
        self.model = self.build_model(hidden_dim, sequence_length, net_type)

    def build_model(self, hidden_dim, sequence_length, net_type):
        model = Sequential(name="Embedder")
        model.add(layers.InputLayer(shape=(sequence_length, 3)))
        model = make_net(
            model,
            n_layers=3,
            hidden_units=hidden_dim,
            output_units=hidden_dim,
            net_type=net_type,
        )
        return model

    def call(self, inputs):
        return self.model(inputs)
