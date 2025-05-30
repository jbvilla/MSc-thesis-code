import os

import numpy as np

from src.flying_squirrel.experiements.map_elites.map_elites_savable_object import MAPElitesSavableObject
from src.flying_squirrel.experiements.shared.interpolation import interpolate_params
from src.utils.load_write import load_pickle
from src.utils.visualise import render_environment


def get_indexes_of_all_directions(fitness_archive: np.ndarray, filled_mask: np.ndarray, use_DESCRIPTOR: bool = False):
    """
    Get the best indexes of all directions
    :param fitness_archive: The fitness archive
    :param filled_mask: The filled mask
    :param use_DESCRIPTOR: If True, use the descriptor instead of the fitness
    :return: A dictionary with the best indexes of all directions
    """
    # (x, y)
    # y points up so higher y is +1
    directions = {
        "N": (0, 1),
        "N2": (0, 1),
        "S": (0, -1),
        "S2": (0, -1),
        "E": (1, 0),
        "E2": (1, 0),
        "W": (-1, 0),
        "W2": (-1, 0),
        "NE": (1, 1),
        "NW": (-1, 1),
        "SE": (1, -1),
        "SW": (-1, -1)
    }

    middle = (fitness_archive.shape[0] // 2, fitness_archive.shape[1] // 2)
    middle_fitness = fitness_archive[middle[0], middle[1]]

    best_indexes = {}
    for direction, (dx, dy) in directions.items():
        best_fitness = middle_fitness
        best_index = middle
        x, y = middle

        # Middle is 4 squares so the middle is different for each direction
        if direction in ["SW", "SE", "E2", "W2"]:
            y -= 1
        if direction in ["NW", "SW", "N2", "S2"]:
            x -= 1

        while 0 <= x < fitness_archive.shape[0] and 0 <= y < fitness_archive.shape[1]:
            if not use_DESCRIPTOR:
                if fitness_archive[x, y] > best_fitness:
                    best_fitness = fitness_archive[x, y]
                    best_index = (x, y)
            else:
                if filled_mask[x, y]:
                    best_index = (x, y)
            x += dx
            y += dy
        best_indexes[direction] = best_index

    # Because the directions are 2 squares taking the best of the two
    if use_DESCRIPTOR:
        best_indexes["N"] = best_indexes["N"] if best_indexes["N"][1] > best_indexes["N2"][1] else best_indexes["N2"]
        best_indexes["S"] = best_indexes["S"] if best_indexes["S"][1] < best_indexes["S2"][1] else best_indexes["S2"]
        best_indexes["E"] = best_indexes["E"] if best_indexes["E"][0] > best_indexes["E2"][0] else best_indexes["E2"]
        best_indexes["W"] = best_indexes["W"] if best_indexes["W"][0] < best_indexes["W2"][0] else best_indexes["W2"]
    else:
        best_indexes["N"] = best_indexes["N"] if fitness_archive[best_indexes["N"][0], best_indexes["N"][1]] > \
            fitness_archive[best_indexes["N2"][0], best_indexes["N2"][1]] else best_indexes["N2"]
        best_indexes["S"] = best_indexes["S"] if fitness_archive[best_indexes["S"][0], best_indexes["S"][1]] < \
            fitness_archive[best_indexes["S2"][0], best_indexes["S2"][1]] else best_indexes["S2"]
        best_indexes["E"] = best_indexes["E"] if fitness_archive[best_indexes["E"][0], best_indexes["E"][1]] > \
            fitness_archive[best_indexes["E2"][0], best_indexes["E2"][1]] else best_indexes["E2"]
        best_indexes["W"] = best_indexes["W"] if fitness_archive[best_indexes["W"][0], best_indexes["W"][1]] < \
            fitness_archive[best_indexes["W2"][0], best_indexes["W2"][1]] else best_indexes["W2"]

    del best_indexes["N2"]
    del best_indexes["S2"]
    del best_indexes["E2"]
    del best_indexes["W2"]
    return best_indexes


def create_videos_of_all_directions(savable: MAPElitesSavableObject, use_DESCRIPTOR: bool = False):
    """
    Create videos of the best individuals in all directions
    :param savable: The savable object
    :param use_DESCRIPTOR: If True, use the descriptor instead of the fitness
    :return: None
    """
    # Get the best indexes of all directions
    directions = get_indexes_of_all_directions(savable.state.fitness_archive, savable.state.filled_mask, use_DESCRIPTOR)

    # Create a directory to save the visualisations
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "visualisations")
    os.makedirs(results_dir, exist_ok=True)

    # Get the environment
    env, env_config = savable.get_env()

    initial_vel = savable.start_vel if savable.start_vel is not None else [0, 0, 0]

    for direction, index in directions.items():
        result_name = os.path.join(results_dir, f"visualisation_{direction}")
        best_params = interpolate_params(savable.state.parameter_archive[index[0], index[1]], savable.number_of_steps, savable.number_of_joints, savable.number_parametric_control_points_spline, simplified=False)
        render_environment(env, env_config, actions=best_params, name_video=result_name, initial_vel=initial_vel)


def create_video_best_individual(savable: MAPElitesSavableObject):
    """
    Create a video of the best individual
    :param savable: The savable object
    :return: None
    """
    # Create a directory to save the visualisations
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "visualisations")
    os.makedirs(results_dir, exist_ok=True)

    # Get the environment
    env, env_config = savable.get_env()

    initial_vel = savable.start_vel if savable.start_vel is not None else [0, 0, 0]

    result_name = os.path.join(results_dir, "visualisation_best")
    index_best = np.unravel_index(np.argmax(savable.state.fitness_archive, axis=None), savable.state.fitness_archive.shape)

    best_params = interpolate_params(savable.state.parameter_archive[index_best[0], index_best[1]], savable.number_of_steps, savable.number_of_joints, savable.number_parametric_control_points_spline, simplified=False)

    render_environment(env, env_config, actions=best_params, name_video=result_name, initial_vel=initial_vel)

    env.close()


if __name__ == "__main__":
    savable = load_pickle("results/2025_05_18_16h02m04s_gen_2000_par_32_ANGULAR_VELOCITY_control_points_2_simpl_wings/params/map_elites_last.pkl")
    create_videos_of_all_directions(savable, use_DESCRIPTOR=False)
    create_video_best_individual(savable)
