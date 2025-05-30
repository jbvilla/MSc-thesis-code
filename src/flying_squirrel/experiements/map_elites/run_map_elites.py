import os
import subprocess
import warnings
from datetime import datetime
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
import wandb

from src.flying_squirrel.experiements.flying_squirrel_environment_builder import get_flying_squirrel_environment
from src.flying_squirrel.experiements.map_elites.map_elites import MapElites, merge_update_map_elite_states, \
    ValidMaskShape
from src.flying_squirrel.experiements.map_elites.map_elites_savable_object import MAPElitesSavableObject
from src.flying_squirrel.experiements.shared.evaluate import FitnessFunction
from src.flying_squirrel.experiements.shared.wandb_utils import create_fitness_archive_heatmap, \
    get_scalar_metrics_map_elites
from src.utils.load_write import save_dict_data_to_csv, get_result_directory, write_pdf_with_url, write_result


def evaluate_step(args):
    map_elites, state, seed, start_vel = args
    env, _ = get_env()
    return map_elites.step(state, env, np.random.RandomState(seed), start_vel)


def run_map_elites(map_elites: MapElites, num_generations: int, num_parallel: int, fitness_function: FitnessFunction):
    rng = np.random.RandomState(0)
    state = map_elites.reset(rng=rng)

    with Pool(processes=num_parallel) as pool:
        for generation in tqdm(range(num_generations), desc="Generations"):
            # Generate seeds for the parallel evaluations
            seeds = rng.randint(0, 2 ** 31 - 1, num_parallel)
            results = pool.map(evaluate_step, [(map_elites, state, seed, start_vel) for seed in seeds])
            state = merge_update_map_elite_states(results)

            # Log scalar
            scalar = get_scalar_metrics_map_elites(state)
            wandb.log(data=scalar, step=generation)
            scalar["generation"] = generation
            save_dict_data_to_csv(scalar, "numeric_data", result_directory)

            savable_obj = MAPElitesSavableObject(
                state=state,
                number_of_steps=number_of_steps,
                number_of_joints=number_of_joints,
                number_parametric_control_points_spline=number_parametric_control_points_spline,
                euler_flying_squirrel=euler_flying_squirrel,
                position_flying_squirrel=flying_squirrel_position,
                simulation_time=simulation_time,
                arena_dim=arena_dim,
                simplified_wings=simplified_wings,
                start_vel=start_vel
            )

            # Always save the last result
            write_result(
                current_best=savable_obj,
                result_directory=result_directory,
                generation=generation,
                name="map_elites",
                last=True
            )

            if generation % 10 == 0 or generation == num_generations - 1:
                heatmap = create_fitness_archive_heatmap(
                    state=state,
                    generation=generation,
                    descriptor_low=descriptor_low,
                    descriptor_high=descriptor_high,
                    fitness_function=fitness_function
                )
                wandb.log({"heatmap": wandb.Image(heatmap)}, step=generation)

            # Save the best result every 50 generations
            if generation % 50 == 0 or generation == num_generations - 1:

                write_result(
                    current_best=savable_obj,
                    result_directory=result_directory,
                    generation=generation,
                    name="map_elites"
                )


"""
Global part because of the multiprocessing "get_env" function can not be inside the if __name__ == "__main__": block
"""

arena_dim = (15, 15)

number_parametric_control_points_spline = 2

descriptor_low = np.array(arena_dim) * -1
descriptor_high = np.array(arena_dim)

euler_flying_squirrel = [0, 0, 0]
simulation_time = 5
simplified_wings = True
flying_squirrel_position = [0, 0, 10]
start_vel = [3, 0, 0]


def get_env():
    return get_flying_squirrel_environment(
        size_arena=arena_dim,
        simulation_time=simulation_time,
        euler_flying_squirrel=euler_flying_squirrel,
        position_flying_squirrel=flying_squirrel_position,
        simplified_wings=simplified_wings,
    )


if __name__ == "__main__":
    # Remove all UserWarnings
    warnings.simplefilter("ignore", category=UserWarning)

    num_generations = 2000
    num_parallel = 32
    fitness_function = FitnessFunction.ANGULAR_VELOCITY
    valid_mask_shape = ValidMaskShape.LOWER_THAN_SHAPE

    timestamp = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")

    experiment_run_name_pdf = f"map_elites_{timestamp}_num_gen_{num_generations}_num_parallel_{num_parallel}_{fitness_function.name}_control_points_{number_parametric_control_points_spline}{'_simpl_wings' if simplified_wings else ''}"
    experiment_run_name = f"{timestamp}_gen_{num_generations}_par_{num_parallel}_{fitness_function.name}_control_points_{number_parametric_control_points_spline}{'_simpl_wings' if simplified_wings else ''}"

    env, env_config = get_env()

    number_of_steps = env_config.total_num_control_steps + 1

    number_of_joints = env.action_space.shape[0]
    # Lower and higher bound for every control point
    parameter_low = np.tile(env.action_space.low, number_parametric_control_points_spline)
    parameter_high = np.tile(env.action_space.high, number_parametric_control_points_spline)

    # So every square is 0.5mx0.5m
    dimensions = tuple(np.array(arena_dim) * 4)
    valid_mask = np.zeros(dimensions, dtype=bool)

    if valid_mask_shape == ValidMaskShape.ALL:
        valid_mask[:, :] = True
    elif valid_mask_shape == ValidMaskShape.RIGHT_HALF:
        # Set only the right half x values to True
        valid_mask[dimensions[0] // 2:, :] = True
    elif valid_mask_shape == ValidMaskShape.LOWER_THAN_SHAPE:
        # Create "<" mask (^ in the mask because x is indexed first than y)
        center_row, center_col = dimensions[0] // 2, dimensions[1] // 2
        slope = 1.0

        # fill the valid mask
        for i in range(dimensions[0] - center_row):
            r = center_row + i
            if dimensions[1] % 2 == 0:
                # Even number so 2 center columns
                col_min = center_col - int(slope * i) - 1
            else:
                col_min = center_col - int(slope * i)
            col_max = center_col + int(slope * i)

            # Clip within the bounds of mask
            col_min = max(col_min, 0)
            col_max = min(col_max, dimensions[0] - 1)

            valid_mask[r, col_min:col_max + 1] = True

    map_elites = MapElites(
        noise_scale=0.1,
        descriptor_low=descriptor_low,
        descriptor_high=descriptor_high,
        parameter_low=parameter_low,
        parameter_high=parameter_high,
        number_of_steps=number_of_steps,
        number_of_joints=number_of_joints,
        number_parametric_control_points_spline=number_parametric_control_points_spline,
        fitness_function=fitness_function,
        valid_sample_mask=valid_mask
    )

    result_directory = get_result_directory("map-elites", experiment_run_name, os.path.dirname(os.path.abspath(__file__)))
    # Create the directory if it does not exist
    os.makedirs(result_directory, exist_ok=True)

    # Initialize wandb
    wandb.init(project="Masterproef-map-elites", group="Map-Elites", name=experiment_run_name)

    commit_number = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
    wandb.run.notes = f"Commit: {commit_number}"
    wandb.run.summary["commit_number"] = commit_number

    # Write pdf with url and run name
    write_pdf_with_url(experiment_run_name_pdf, wandb.run.get_url(), result_directory)

    run_map_elites(map_elites, num_generations=num_generations, num_parallel=num_parallel, fitness_function=fitness_function)

    wandb.finish()
