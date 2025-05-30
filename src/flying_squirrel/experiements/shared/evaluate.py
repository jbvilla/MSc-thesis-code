from enum import Enum
from typing import Tuple

import numpy as np

from src.flying_squirrel.environment.basic.mjc_env import FlyingSquirrelBasicMJCEnvironment
from src.flying_squirrel.experiements.shared.flight_path import FlightPathWithPhases, FlightPath
from src.flying_squirrel.experiements.shared.interpolation import interpolate_params


class FitnessFunction(Enum):
    """
    Enum for the fitness function
    """
    DISTANCE = 0
    X_DISTANCE = 1
    DISTANCE_HEIGHT = 2
    ANGULAR_VELOCITY = 3
    TARGET = 4
    PHASES = 5
    PATH = 6


def _fitness_function(
        start_pos,
        end_pos,
        heights,
        angular_velocities,
        state,
        flight_path_with_phases,
        positions,
        flight_path: FlightPath,
        is_upside_down: bool,
        option: FitnessFunction = FitnessFunction.DISTANCE) -> float:
    """
    LOWER FITNESS IS BETTER for this fitness function

    Fitness function to evaluate the parameters
    :param start_pos: Start position
    :param end_pos: End position
    :param heights: Heights of the squirrel
    :param angular_velocities: Angular velocities of the squirrel
    :param state: State of the environment
    :param flight_path_with_phases: Flight path object for flight path experiment with phases
    :param positions: Position of the squirrel
    :param flight_path: Flight path object for flight path experiment without phases
    :param is_upside_down: Whether the squirrel is upside down
    :param option: Option to evaluate the fitness function
    :return: Fitness of the parameters
    """
    if option == FitnessFunction.DISTANCE:
        return float(-np.linalg.norm(end_pos[:2] - start_pos))
    if option == FitnessFunction.X_DISTANCE:
        return -(float(end_pos[0] - start_pos[0]))
    elif option == FitnessFunction.DISTANCE_HEIGHT:
        return float(-np.linalg.norm(end_pos[:2] - start_pos) - 2 * np.mean(heights))
    elif option == FitnessFunction.ANGULAR_VELOCITY:
        # Use of squared angular to avoid have the same effect for negative and positive angular velocity and
        # to punish high angular velocity
        return np.mean(np.sum(angular_velocities**2, axis=1))
    elif option == FitnessFunction.TARGET:
        # Positive because farther away is bigger number
        return FlyingSquirrelBasicMJCEnvironment.get_xyz_distance_to_target(state)
    elif option == FitnessFunction.PHASES:
        phase_values = []
        control_points_indexes = flight_path_with_phases.get_control_point_location_indexes()

        # phase 1
        position_direction = positions[control_points_indexes[0]] - positions[0]
        position_direction = position_direction / np.linalg.norm(position_direction)
        direction_path = flight_path_with_phases.directions[0]
        # abs makes the search space smaller
        phase_values.append(1 - np.dot(position_direction, direction_path))

        # phase 2
        position_direction = positions[control_points_indexes[2]] - positions[control_points_indexes[1]]
        position_direction = position_direction / np.linalg.norm(position_direction)
        direction_path = flight_path_with_phases.directions[1]
        phase_values.append(1 - np.dot(position_direction, direction_path))

        # phase 3
        position_direction = positions[len(positions) - 1] - positions[control_points_indexes[3]]
        position_direction = position_direction / np.linalg.norm(position_direction)
        direction_path = flight_path_with_phases.directions[2]
        phase_values.append(1 - np.dot(position_direction, direction_path))

        fitness = np.sum(phase_values)

        return fitness
    elif option == FitnessFunction.PATH:
        height_diff = positions[0][2] - positions[-1][2]

        # Create matrix of distances between the positions and the flight path
        positions = np.array(positions)[:, :2]
        dists = np.linalg.norm(positions[:, np.newaxis, :] - flight_path.interpolated_path[:, :2][np.newaxis, :, :], axis=2)

        # Calculate the progress of the path
        closest_indexes = np.argmin(dists, axis=1)

        diff = np.diff(closest_indexes)
        non_monotonic_penalty = np.sum((diff < 0).astype(int))
        normalized_non_monotonic_penalty = np.clip(non_monotonic_penalty / len(diff), 0, 1)

        progress = np.max(closest_indexes)
        normalized_progress = np.clip(progress / (len(flight_path.interpolated_path) - 1), 0, 1)

        distance_to_path = np.mean(np.min(dists, axis=1))

        normalized_distance = np.clip(distance_to_path / 10, 0, 1)

        normalized_height_diff = np.clip(height_diff / 10, 0, 1)

        return 4 * normalized_distance + normalized_non_monotonic_penalty + (1 - normalized_progress) + normalized_height_diff

    else:
        raise ValueError(f"Option {option} is not implemented.")


def evaluate_params(
        params: np.ndarray,
        number_of_steps: int,
        number_of_joints: int,
        number_control_points_spline: int,
        env: FlyingSquirrelBasicMJCEnvironment,
        fitness_function: FitnessFunction,
        start_vel: list,
        simplified: bool,
        flight_path_with_phases: FlightPathWithPhases = None,
        flight_path: FlightPath = None,
) -> Tuple[float, np.ndarray]:
    """
    Evaluate the parameters by interpolating them and simulating the environment
    :param params: Parameters to evaluate
    :param number_of_steps: Number of steps to simulate
    :param number_of_joints: Number of joints
    :param number_control_points_spline: Number of control points
    :param env: Environment to simulate
    :param fitness_function: Fitness function to evaluate the parameters
    :param start_vel: Initial velocity
    :param simplified: Whether the number of joints is simplified (so left and right limbs are mirrored)
    :param flight_path_with_phases: Flight path object for flight path experiment with phases
    :param flight_path: Flight path object for flight path experiment without phases
    :return: Fitness of the parameters and the end position
    """
    params = interpolate_params(params, number_of_steps, number_of_joints, number_control_points_spline, simplified, flight_path_with_phases)
    rng = np.random.RandomState(0)
    env_state = env.reset(rng=rng)

    # Set the initial velocity
    env_state.mj_data.qvel[0] = start_vel[0]
    env_state.mj_data.qvel[1] = start_vel[1]
    env_state.mj_data.qvel[2] = start_vel[2]

    if params is not None:
        # Found by looking at the joint ranges that are not [0-0]
        start_index_joint = 7
        for i in range(start_index_joint, np.shape(params)[1]):
            env_state.mj_data.qpos[i] = params[0][i]

    # copy otherwise it will be changed in the loop because it is a reference
    start_pos = env_state.observations["trunk_position"][:2].copy()
    heights = []
    angular_velocities = []
    positions = []
    is_upside_down = False
    i = 0
    while not (env_state.terminated | env_state.truncated):
        env_state = env.step(state=env_state, action=params[i])
        heights.append(env_state.observations["trunk_position"][2].copy())
        angular_velocities.append(env_state.observations["trunk_angular_velocity"].copy())
        positions.append(env_state.observations["trunk_position"].copy())

        # Check if the flying squirrel is upside down
        xmat = env_state.mj_data.xmat[env_state.mj_data.body("FlyingSquirrelMorphology/trunk").id].reshape(3, 3)
        z_axis_world = xmat[:, 2]
        z_up = z_axis_world[2]
        # Punish if the squirrel is upside down
        if z_up < -0.5:
            is_upside_down = True

        i += 1
    end_pos = env_state.observations["trunk_position"].copy()

    # Return negative distance between start and end position (fitness) - average height
    return _fitness_function(start_pos, end_pos, heights, np.array(angular_velocities), env_state, flight_path_with_phases, positions, flight_path, is_upside_down, fitness_function), end_pos
