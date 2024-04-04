from tensorflow.keras import Model, Sequential, layers
import os
import sys

sys.path.append(os.pardir)
from .utils import make_net


class Generator(Model):
    def __init__(self, hidden_dim, sequence_length, net_type="GRU"):
        super(Generator, self).__init__()
        self.model = self.build_model(hidden_dim, sequence_length, net_type)

    def build_model(self, hidden_dim, sequence_length, net_type):
        model = Sequential(name="Generator")
        model.add(layers.InputLayer(input_shape=(sequence_length, hidden_dim)))
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
