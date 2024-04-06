import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
import logging

from models import Generator, Supervisor, Discriminator, Embedder, Recovery
from GAN.utils import load_chaotic_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_timegan(models, data, hyperparameters):
    """
    Train the TimeGAN framework with the provided data and models.

    Args:
        models (dict): A dictionary containing the models (Generator, Discriminator,
                       Embedder, Recovery, Supervisor) used in the TimeGAN framework.
        data (np.ndarray): The real sequences to be used for training, shaped as
                           (num_samples, sequence_length, num_variables).
        hyperparameters (dict): Training hyperparameters such as learning rate, batch size,
                                number of epochs, etc.

    """
    # Unpack the models
    generator, discriminator, embedder, recovery, supervisor = (
        models["generator"],
        models["discriminator"],
        models["embedder"],
        models["recovery"],
        models["supervisor"],
    )

    # Unpack hyperparameters
    learning_rate = hyperparameters.get("learning_rate", 0.001)
    batch_size = hyperparameters.get("batch_size", 128)
    epochs = hyperparameters.get("epochs", 1000)

    # Optimizers
    g_optimizer = Adam(learning_rate)
    d_optimizer = Adam(learning_rate)
    e_optimizer = Adam(learning_rate)
    r_optimizer = Adam(learning_rate)
    s_optimizer = Adam(learning_rate)

    # Loss function
    mse_loss = tf.keras.losses.MeanSquaredError()
    bce_loss = tf.keras.losses.BinaryCrossentropy()

    for epoch in range(epochs):
        # Shuffle and batch the data
        np.random.shuffle(data)
        num_batches = int(np.ceil(data.shape[0] / batch_size))

        for i in range(num_batches):
            batch_data = data[i * batch_size : (i + 1) * batch_size]
            # logging.info("batch_data shape:", batch_data.shape)

            # Noise vector for generator
            noise = np.random.normal(
                0,
                1,
                (
                    batch_size,
                    hyperparameters["sequence_length"],
                    hyperparameters["hidden_dim"],
                ),
            )

            # Train Embedder
            with tf.GradientTape() as tape:
                embedded_data = embedder(batch_data, training=True)
                # logging.info("embedded_data shape:", embedded_data.shape)

                recovery_data = recovery(embedded_data, training=True)
                # logging.info("recovery_data shape:", recovery_data.shape)
                e_loss = mse_loss(batch_data, recovery_data)
            e_grads = tape.gradient(
                e_loss, embedder.trainable_variables + recovery.trainable_variables
            )
            e_optimizer.apply_gradients(
                zip(
                    e_grads, embedder.trainable_variables + recovery.trainable_variables
                )
            )

            # Train Supervisor
            with tf.GradientTape() as tape:
                supervised_data = supervisor(embedded_data, training=True)
                # logging.info("supervised_data shape:", supervised_data.shape)
                s_loss = mse_loss(embedded_data, supervised_data)
            s_grads = tape.gradient(s_loss, supervisor.trainable_variables)
            s_optimizer.apply_gradients(zip(s_grads, supervisor.trainable_variables))

            # Generate synthetic data
            synthetic_data = generator(
                noise, training=False
            )  # Generator is trained through the adversarial process
            # logging.info("synthetic_data shape:", synthetic_data.shape)

            # Discriminator training
            with tf.GradientTape() as tape:
                d_real = discriminator(batch_data, training=True)
                d_fake = discriminator(synthetic_data, training=True)
                d_loss_real = bce_loss(tf.ones_like(d_real), d_real)
                d_loss_fake = bce_loss(tf.zeros_like(d_fake), d_fake)
                d_loss = (d_loss_real + d_loss_fake) / 2
            d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

            # Generator (Adversarial) training
            with tf.GradientTape() as tape:
                synthetic_data = generator(noise, training=True)
                d_fake = discriminator(synthetic_data, training=True)
                g_loss = bce_loss(tf.ones_like(d_fake), d_fake)
            g_grads = tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

        # Logging for monitoring
        logging.info(
            f"Epoch {epoch+1}/{epochs}, E Loss: {e_loss:.4f}, S Loss: {s_loss:.4f}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}"
        )


# Configuration and model initialization
input_dim = 100
sequence_length = 100
num_variables = 3
attractor_names = [
    "LorenzAttractor",
    "RosslerAttractor",
    "ChenAttractor",
]

# Load or generate real chaotic data
logging.info("Loading real data")
real_data = load_chaotic_data(
    attractor_names,
    sequence_length=sequence_length,
    perturbation=True,
    num_trajectories=1,
)
logging.info(f"Loaded data shape: {real_data.shape}")

# Normalize data
scaler = MinMaxScaler(feature_range=(-1, 1))
real_data_reshaped = real_data.reshape(-1, num_variables)
scaler.fit(real_data_reshaped)
normalized_data = scaler.transform(real_data_reshaped)
normalized_data = normalized_data.reshape(-1, sequence_length, num_variables)

# Define hyperparameters
hyperparameters = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 200,
    "hidden_dim": 24,
    "sequence_length": sequence_length,
    "num_variables": num_variables,
}

# Initialize models
hidden_dim = hyperparameters["hidden_dim"]
net_type = "GRU"  # Or "LSTM" based on preference

generator = Generator(
    hidden_dim=hidden_dim, sequence_length=sequence_length, net_type=net_type
)
discriminator = Discriminator(
    hidden_dim=hidden_dim, sequence_length=sequence_length, net_type=net_type
)
embedder = Embedder(
    hidden_dim=hidden_dim, sequence_length=sequence_length, net_type=net_type
)
recovery = Recovery(
    hidden_dim=hidden_dim, sequence_length=sequence_length, net_type=net_type
)
supervisor = Supervisor(
    hidden_dim=hidden_dim, sequence_length=sequence_length, net_type=net_type
)

models = {
    "generator": generator,
    "discriminator": discriminator,
    "embedder": embedder,
    "recovery": recovery,
    "supervisor": supervisor,
}

# Ready to train
logging.info("Starting training process")
train_timegan(models, normalized_data, hyperparameters)
