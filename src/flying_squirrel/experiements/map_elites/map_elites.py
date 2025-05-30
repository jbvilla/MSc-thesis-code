from enum import Enum
from typing import List

import numpy as np
from dataclasses import dataclass

from src.flying_squirrel.environment.basic.mjc_env import FlyingSquirrelBasicMJCEnvironment
from src.flying_squirrel.experiements.shared.evaluate import evaluate_params, FitnessFunction


@dataclass
class MAPElitesState:
    """
    The state of the MAP-Elites algorithm
    """
    parameter_archive: np.ndarray
    """
    Parameters for each cell
    """

    fitness_archive: np.ndarray
    """
    Fitness for each cell that belongs to the parameters of the cell
    """

    descriptor_archive: np.ndarray
    """
    Descriptor for each cell that belongs to the parameters of the cell
    """

    filled_mask: np.ndarray
    """
    The mask that indicates which cells are filled
    """


class MapElites:
    """
    MAP-Elites algorithm
    """

    def __init__(
            self,
            noise_scale: float,
            descriptor_low: np.ndarray,
            descriptor_high: np.ndarray,
            parameter_low: np.ndarray,
            parameter_high: np.ndarray,
            number_of_steps: int,
            number_of_joints: int,
            number_parametric_control_points_spline: int,
            fitness_function: FitnessFunction,
            valid_sample_mask: np.ndarray
    ) -> None:
        self._dimensions = valid_sample_mask.shape
        self._noise_scale = noise_scale
        self._descriptor_low = descriptor_low
        self._descriptor_high = descriptor_high
        self._parameter_low = parameter_low
        self._parameter_high = parameter_high
        self._number_of_steps = number_of_steps
        self._number_of_joints = number_of_joints
        self._number_parametric_control_points_spline = number_parametric_control_points_spline
        self._fitness_function = fitness_function
        self._valid_sample_mask = valid_sample_mask

    def _select_random_cell(self, rng: np.random.RandomState) -> np.ndarray:
        """
        Returns a random cell index in the MAP-Elites grid that is valid
        :param rng: the random number generator
        :return: the random cell index np.ndarray of size len(self._dimensions) that is valid
        """
        # Get the indices of the valid cells
        valid_indices = np.argwhere(self._valid_sample_mask)
        random_index = rng.choice(len(valid_indices))
        return valid_indices[random_index]

    def _mutate_parameters(self, parameters: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """
        Mutate parameters by adding gaussian noise
        :param parameters: the parameters to mutate
        :param rng: the random number generator
        :return: the mutated parameters
        """
        # Normalize parameters to [0, 1]
        normalized_parameters = (parameters - self._parameter_low) / (self._parameter_high - self._parameter_low)

        # Generate noise
        noise = rng.normal(scale=self._noise_scale, size=len(parameters))

        # Add noise to parameters
        normalized_parameters = normalized_parameters + noise

        # Denormalize parameters
        parameters = (normalized_parameters * (self._parameter_high - self._parameter_low)) + self._parameter_low

        # Clip parameters to be within the bounds
        return np.clip(parameters, self._parameter_low, self._parameter_high)

    def _get_cell_index(self, descriptor: np.array) -> np.array:
        """
        Get the cell index for the given descriptor
        :param descriptor: the descriptor
        :return: cell index
        """
        descriptor = np.clip(descriptor, self._descriptor_low, self._descriptor_high)

        # normalize the descriptor to be between 0 and 1
        descriptor = (descriptor - self._descriptor_low) / (self._descriptor_high - self._descriptor_low)

        cell_index = (descriptor * self._dimensions).astype(np.int32)
        # return the cell index
        return np.clip(cell_index, 0, np.array(self._dimensions) - 1)

    def add_to_archive(
            self,
            state: MAPElitesState,
            descriptor: np.ndarray,
            fitness: float,
            parameters: np.ndarray
    ) -> MAPElitesState:
        """
        Add a new solution to the archive if it is better than the current solution or if the cell is empty
        :param state: the current state of the MAP-Elites algorithm
        :param descriptor: the descriptor of the solution
        :param fitness: the fitness of the solution
        :param parameters: the parameters of the solution
        :return: the new state of the MAP-Elites algorithm
        """
        cell_index = self._get_cell_index(descriptor)

        # Check if the cell is empty or if the new solution is better than the current solution
        if self._valid_sample_mask[*cell_index] and (not state.filled_mask[*cell_index] or fitness > state.fitness_archive[*cell_index]):
            # Update the archive
            state.filled_mask[*cell_index] = True
            state.fitness_archive[*cell_index] = fitness
            state.parameter_archive[*cell_index] = parameters
            state.descriptor_archive[*cell_index] = descriptor

        return state

    def step(
            self,
            state: MAPElitesState,
            env: FlyingSquirrelBasicMJCEnvironment,
            rng: np.random.RandomState,
            start_vel: List = None
    ) -> MAPElitesState:
        """
        Run one step of the MAP-Elites algorithm
        :param state: the current state of the MAP-Elites algorithm
        :param env: the environment to evaluate the parameters
        :param rng: the random number generator
        :param start_vel: the initial velocity of the flying squirrel
        :return: the new state of the MAP-Elites algorithm
        """
        if start_vel is None:
            start_vel = [0, 0, 0]

        # Split the random number generator
        selection_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
        mutation_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))

        # Select a random cell
        cell_index = self._select_random_cell(selection_rng)

        parameters = state.parameter_archive[*cell_index]
        parameters = self._mutate_parameters(parameters, mutation_rng)

        fitness, descriptor = evaluate_params(
            params=parameters,
            number_of_steps=self._number_of_steps,
            number_of_joints=self._number_of_joints,
            number_control_points_spline=self._number_parametric_control_points_spline,
            env=env,
            fitness_function=self._fitness_function,
            start_vel=start_vel,
            simplified=False
        )

        # 2D case
        descriptor = descriptor[:2]
        # Invert the fitness because MAP-Elites maximizes the fitness here
        fitness = -fitness

        state = self.add_to_archive(state=state, descriptor=descriptor, fitness=fitness, parameters=parameters)

        return state

    def reset(self, rng: np.random.RandomState) -> MAPElitesState:
        """
        Return the initial state of the MAP-Elites algorithm
        :param rng: the random number generator
        :return: the initial state of the MAP-Elites algorithm
        """
        parameters = rng.uniform(
            low=self._parameter_low,
            high=self._parameter_high,
            # parameters low size = number of joints * number of control points
            size=self._dimensions + (self._parameter_low.size,)
        ).astype(np.float32)

        return MAPElitesState(
            parameter_archive=parameters,
            fitness_archive=np.full(self._dimensions, -np.inf),
            descriptor_archive=np.zeros(self._dimensions + (self._descriptor_low.size,)),
            filled_mask=np.zeros(self._dimensions, dtype=bool)
        )


def merge_update_map_elite_states(states: List[MAPElitesState]) -> MAPElitesState:
    """
    Merge the states of the MAP-Elites algorithm
    :param states: the states to merge
    :return: the merged state
    """
    filled_mask = np.max([state.filled_mask for state in states], axis=0)
    fitness_archive = np.max([state.fitness_archive for state in states], axis=0)

    # Indices of the highest fitness values
    state_indices_of_highest_fitness_values = np.argmax([state.fitness_archive for state in states], axis=0)
    # Keep the parameters corresponding to the best fitness values
    parameter_archive = np.array([state.parameter_archive for state in states])[
        state_indices_of_highest_fitness_values,
        np.arange(states[0].parameter_archive.shape[0])[:, None],
        np.arange(states[0].parameter_archive.shape[1])
    ]
    # Keep the descriptors corresponding to the best fitness values
    descriptor_archive = np.array([state.descriptor_archive for state in states])[
        state_indices_of_highest_fitness_values,
        np.arange(states[0].descriptor_archive.shape[0])[:, None],
        np.arange(states[0].descriptor_archive.shape[1])
    ]

    return MAPElitesState(
        filled_mask=filled_mask,
        fitness_archive=fitness_archive,
        parameter_archive=parameter_archive,
        descriptor_archive=descriptor_archive
    )


class ValidMaskShape(Enum):
    """
    The shape of the valid mask
    """
    ALL = 0
    RIGHT_HALF = 1
    LOWER_THAN_SHAPE = 2
