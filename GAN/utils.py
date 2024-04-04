import os
import sys

sys.path.append(os.getcwd())
from Attractors import (
    AizawaAttractor,
    ChenAttractor,
    DadrasAttractor,
    FourWingAttractor,
    HalvorsenAttractor,
    LorenzAttractor,
    Lorenz83Attractor,
    RabinovichFabrikantAttractor,
    RosslerAttractor,
    SprottAttractor,
    ThomasAttractor,
    ThreeScrollUnifiedAttractor,
)
import numpy as np


def load_chaotic_data(
    attractor_names,
    sequence_length=10000,
    perturbation=False,
    num_trajectories=2,
    perturbation_magnitude=0.05,
):
    """
    Generates or loads chaotic data from specified attractors.

    Args:
        attractor_names (list of str): Names of attractors to generate data from.
        sequence_length (int): Length of each trajectory.
        perturbation (bool): Whether to generate perturbed trajectories.
        num_trajectories (int): Number of trajectories to generate for each attractor.
                               This is used only if perturbation is True.
        perturbation_magnitude (float): Magnitude of perturbation for generating multiple trajectories.

    Returns:
        np.ndarray: Array of generated data with shape (n_samples, sequence_length, n_features).
    """
    attractors = {
        "AizawaAttractor": AizawaAttractor,
        "ChenAttractor": ChenAttractor,
        "DadrasAttractor": DadrasAttractor,
        "FourWingAttractor": FourWingAttractor,
        "HalvorsenAttractor": HalvorsenAttractor,
        "LorenzAttractor": LorenzAttractor,
        "Lorenz83Attractor": Lorenz83Attractor,
        "RabinovichFabrikantAttractor": RabinovichFabrikantAttractor,
        "RosslerAttractor": RosslerAttractor,
        "SprottAttractor": SprottAttractor,
        "ThomasAttractor": ThomasAttractor,
        "ThreeScrollUnifiedAttractor": ThreeScrollUnifiedAttractor,
    }

    data = []

    for name in attractor_names:
        attractor_class = attractors.get(name)
        if attractor_class is None:
            raise ValueError(f"Attractor {name} not recognized.")

        attractor = attractor_class()
        if perturbation:
            trajectories = attractor.generate_perturbed_trajectories(
                num_trajectories=num_trajectories,
                perturbation_magnitude=perturbation_magnitude,
                t_span=(0, sequence_length),
                t_steps=sequence_length,
            )
        else:
            trajectory = attractor.generate_trajectory(
                t_span=(0, sequence_length), t_steps=sequence_length
            )
            trajectories = [trajectory]

        data.extend(trajectories)

    data = np.array(data)
    data = np.transpose(data, (0, 2, 1))

    return data


from tensorflow.keras import layers


def make_net(model, n_layers, hidden_units, output_units, net_type="GRU"):
    """
    Adds recurrent layers to the provided model based on the specified parameters.

    Args:
        model (Sequential): The model to add layers to.
        n_layers (int): The number of recurrent layers to add.
        hidden_units (int): The number of units in each recurrent layer.
        output_units (int): The number of units in the output layer.
        net_type (str): The type of recurrent layer to add ('GRU' or 'LSTM').

    Returns:
        Sequential: The model with the added layers.
    """
    if net_type == "GRU":
        for i in range(n_layers):
            model.add(
                layers.GRU(
                    units=hidden_units, return_sequences=True, name=f"GRU_{i + 1}"
                )
            )
    else:
        for i in range(n_layers):
            model.add(
                layers.LSTM(
                    units=hidden_units, return_sequences=True, name=f"LSTM_{i + 1}"
                )
            )

    model.add(layers.Dense(units=output_units, activation="sigmoid", name="OUT"))
    return model
