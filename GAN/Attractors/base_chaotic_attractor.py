import numpy as np
from scipy.integrate import solve_ivp


class BaseChaoticAttractor:
    """
    Base class for simulating chaotic attractors.

    Attributes:
        initial_state (np.ndarray): The initial state of the chaotic system.
        parameters (tuple): Parameters specific to the chaotic system.
    """

    def __init__(self, initial_state, parameters):
        """
        Initialize the chaotic attractor with an initial state and system parameters.

        Args:
            initial_state (list or np.ndarray): The initial state (conditions) of the system.
            parameters (tuple): System-specific parameters.
        """
        self.initial_state = np.array(initial_state)
        self.parameters = parameters

    def next_state(self, t, current_state, parameters):
        """
        Compute the next state of the system. This method should be overridden by subclasses
        to implement specific attractor equations.

        Args:
            t (float): Current time step.
            current_state (np.ndarray): Current state of the system.
            parameters (tuple): Parameters of the system.

        Returns:
            np.ndarray: The next state of the system.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate_trajectory(self, t_span, t_steps=10000):
        """
        Generate a trajectory for the chaotic system over a specified time span.

        Args:
            t_span (tuple): The start and end of the time interval.
            t_steps (int): Number of time steps to evaluate.

        Returns:
            np.ndarray: The system trajectory as an array of states.
        """
        t_eval = np.linspace(*t_span, t_steps)
        sol = solve_ivp(
            fun=self.next_state,
            t_span=t_span,
            y0=self.initial_state,
            t_eval=t_eval,
            args=(self.parameters,),
        )
        return sol.y

    def generate_perturbed_trajectories(
        self,
        num_trajectories=10,
        perturbation_magnitude=0.01,
        t_span=(0, 40),
        t_steps=10000,
    ):
        """
        Generate multiple trajectories from perturbed initial states.

        Args:
            num_trajectories (int): Number of trajectories to generate.
            perturbation_magnitude (float): Magnitude of the perturbation applied to the initial state.
            t_span (tuple): Time span for each trajectory.
            t_steps (int): Number of time steps to evaluate.

        Returns:
            list: A list of trajectories, each a np.ndarray of states.
        """
        trajectories = []
        for _ in range(num_trajectories):
            perturbed_initial_state = self.initial_state + np.random.uniform(
                -perturbation_magnitude,
                perturbation_magnitude,
                size=self.initial_state.shape,
            )
            self.initial_state = (
                perturbed_initial_state  # Update the initial state for each trajectory
            )
            trajectory = self.generate_trajectory(t_span, t_steps)
            trajectories.append(trajectory)
            self.initial_state -= (
                perturbed_initial_state - self.initial_state
            )  # Reset to original initial state after perturbation
        return trajectories
