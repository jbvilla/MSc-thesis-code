import io
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb

from src.flying_squirrel.experiements.map_elites.map_elites import MAPElitesState
from src.flying_squirrel.experiements.shared.evaluate import FitnessFunction
from src.utils.load_write import write_pickle
from src.utils.visualise import render_environment, get_video_buffer_webm


def _create_grid(
        map_size: Tuple[int, int],
        end_locations: List[np.ndarray],
        cels_per_map_size: int = 5,
) -> np.ndarray:
    """
    Create a grid with per cell the number of end locations that are in that cell
    :param map_size: The size of the map
    :param cels_per_map_size: The number of cells per map size
    :param end_locations: The end locations
    :return: The grid filled with counts of end locations
    """
    # Grid size is 2 times the map size because the map can be negative
    grid_size = (2 * map_size[0] * cels_per_map_size, 2 * map_size[1] * cels_per_map_size)
    grid = np.zeros(grid_size)
    for end_location in end_locations:
        x = int((end_location[0] + map_size[0]) * cels_per_map_size)
        y = int((end_location[1] + map_size[1]) * cels_per_map_size)
        if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
            grid[x, y] += 1
    return grid


def create_generation_end_location_heatmap(
        end_locations: List[np.ndarray],
        generation: int,
        map_size: Tuple[int, int],
        result_directory: str = None
) -> np.ndarray:
    """
    Create a heatmap of the end locations
    :param end_locations: The end locations
    :param generation: The generation
    :param map_size: The size of the map
    :param result_directory: The directory to save heatmap
    :return: The heatmap as an image
    """
    # Save the data to a pickle file for later use
    if result_directory is not None:
        result_directory = os.path.join(result_directory, "heatmaps")
        os.makedirs(result_directory, exist_ok=True)
        result_pkl_file = os.path.join(result_directory, f"heatmap_data_{generation}.pkl")
        write_pickle((end_locations, map_size), result_pkl_file)

    # Transpose so X is horizontal and Y is vertical
    grid = _create_grid(map_size, end_locations).T

    mask = (grid == 0)
    ax = sns.heatmap(grid, cmap="coolwarm", mask=mask)
    ax.invert_yaxis()
    plt.title(f"Generatie {generation}: heatmap van eindposities")
    plt.xlabel("X positie")
    plt.ylabel("Y positie")

    color_bar = ax.collections[0].colorbar
    color_bar.set_label("Aantal parameters met dezelfde eindposities")

    # Set the ticks
    ax.set_xticks([0, grid.shape[1] - 1])
    ax.set_xticklabels([-map_size[1], map_size[1]])
    ax.set_yticks([0, grid.shape[0] - 1])
    ax.set_yticklabels([-map_size[0], map_size[0]])

    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, format='png')
    image_buffer.seek(0)

    image = plt.imread(image_buffer)
    plt.close()
    image_buffer.close()

    return image


def get_metrics(fitness_values: List[float]) -> dict[str, np.array]:
    """
    Get the mean, std and best fitness
    :param fitness_values: The fitness values
    :return: The mean, std and best fitness
    """
    return {
        "mean_fitness": np.mean(fitness_values),
        "max_fitness": np.max(fitness_values),
        "min_fitness": np.min(fitness_values),
    }


def get_spline_plot(actuator: str, interpolated_params, control_points, x_location_control_points, generation: int):
    x = list(range(len(interpolated_params)))
    plt.plot(x, interpolated_params, linestyle='-', label='Geïnterpoleerde spline')
    plt.scatter(x_location_control_points, control_points, color='red', label='Controlepunten')
    plt.title(f"{actuator} spline voor generatie {generation}")
    plt.xlabel("Tijdsverloop")
    plt.ylabel("gewrichtspositie (rad)")
    plt.legend()

    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, format='png')
    image_buffer.seek(0)

    image = plt.imread(image_buffer)
    plt.close()
    image_buffer.close()

    return image


def create_spline_plots(
        interpolated_params: np.ndarray,
        control_points: np.ndarray,
        x_location_control_points: np.ndarray,
        generation: int,
        actuators: List[str],
        simplified: bool = False,
):
    """
    Create spline plots for each actuator
    :param interpolated_params: Interpolated parameters
    :param control_points: Control points
    :param x_location_control_points: X location of the control points
    :param generation: The generation
    :param actuators: The actuators
    :param simplified: Whether the number of joints is simplified
    :return:
    """

    if simplified:
        # Simplified number of joints so only the half of the libms are generated so mirror them
        control_points = [control_points[:6, :], control_points[:6, :], control_points[6:12, :], control_points[6:12, :], control_points[12:, :]]
        control_points = np.vstack(control_points)

    figures = []
    figure_names = []
    for i, actuator in enumerate(actuators):
        figure_name = actuator.split('/')[1].split('_p_')[0]
        figure = get_spline_plot(figure_name, interpolated_params[:, i], control_points[i], x_location_control_points,
                                 generation)
        figures.append(figure)
        figure_names.append(figure_name)

    wandb.log({"Best splines": [wandb.Image(figure) for figure in figures]}, step=generation)


def create_video(env, env_config, params, initial_vel, generation) -> None:
    """
    Create a video with the given parameters and log it to wandb
    :param env: the environment
    :param env_config: the configuration of the environment
    :param params: the parameters to use
    :param initial_vel: the initial velocity
    :param generation: current generation
    :return: None
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    result_name = os.path.join(current_dir, "temp")

    render_environment(env, env_config, image=False, write_xml=False, actions=params, initial_vel=initial_vel,
                       name_video=result_name, wandb_video=True)

    video_buffer = get_video_buffer_webm(result_name)

    wandb.log({"Best video": wandb.Video(video_buffer, format="webm")}, step=generation)

    os.remove(result_name + ".webm")


def create_fitness_archive_heatmap(
        state: MAPElitesState,
        generation: int,
        descriptor_low: np.ndarray,
        descriptor_high: np.ndarray,
        fitness_function: FitnessFunction,
) -> np.ndarray:
    """
    Create a heatmap of the fitness archive of the MAPElites state.
    """
    # Transposing so X is horizontal and Y is vertical
    data = np.array(state.fitness_archive).T
    mask = (data == -np.inf)

    ax = sns.heatmap(data, cmap="coolwarm", mask=mask)
    ax.invert_yaxis()
    plt.xlabel("X positie")
    plt.ylabel("Y positie")
    plt.title(f"Fitnessarchief bij generatie: {generation}")

    cbar = ax.collections[0].colorbar
    if fitness_function == FitnessFunction.DISTANCE:
        cbar.set_label("Afstand")
    elif fitness_function == FitnessFunction.X_DISTANCE:
        cbar.set_label("X afstand")
    elif fitness_function == FitnessFunction.DISTANCE_HEIGHT:
        cbar.set_label("Afstand + hoogte")
    elif fitness_function == FitnessFunction.ANGULAR_VELOCITY:
        cbar.set_label("Stabiliteit")
    plt.xticks(ticks=[0, data.shape[1]], labels=[str(descriptor_low[0]), str(descriptor_high[0])])
    plt.yticks(ticks=[0, data.shape[0]], labels=[str(descriptor_high[1]), str(descriptor_low[1])])

    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, format='png')
    image_buffer.seek(0)

    image = plt.imread(image_buffer)
    plt.close()
    image_buffer.close()

    return image


def get_scalar_metrics_map_elites(state: MAPElitesState) -> dict[str, np.array]:
    return {
        "mean_fitness": np.sum(np.where(state.filled_mask, state.fitness_archive, 0)) / np.sum(state.filled_mask),
        "maximum_fitness": np.max(np.where(state.filled_mask, state.fitness_archive, -np.inf)),
        "minimum_fitness": np.min(np.where(state.filled_mask, state.fitness_archive, np.inf)),
        "archive_occupancy": np.sum(state.filled_mask) / state.filled_mask.size,
        "furthest_distance": np.max(np.where(state.filled_mask, np.linalg.norm(state.descriptor_archive, axis=-1), -np.inf))
    }
