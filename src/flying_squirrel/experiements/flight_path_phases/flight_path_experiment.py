import os
import subprocess
import warnings
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import wandb
from cmaes import CMA
from tqdm import tqdm

from src.flying_squirrel.experiements.flight_path_phases.flight_path_with_phases_savable_object import FlightPathWithPhasesSavableObject
from src.flying_squirrel.experiements.flying_squirrel_environment_builder import get_flying_squirrel_environment
from src.flying_squirrel.experiements.shared.evaluate import evaluate_params, FitnessFunction
from src.flying_squirrel.experiements.shared.flight_path import FlightPathWithPhases
from src.flying_squirrel.experiements.shared.wandb_utils import get_metrics
from src.utils.load_write import get_result_directory, write_pdf_with_url, save_dict_data_to_csv, write_result


def evaluate_step(args):
    (params, number_of_steps, number_of_joints,
     number_parametric_control_points_spline, fitness_function, straight_flight_path) = args
    env, _ = get_env()
    return evaluate_params(params, number_of_steps, number_of_joints,
                           number_parametric_control_points_spline, env, fitness_function,
                           [0, 0, 0], False, straight_flight_path)


def run_flight_path():
    current_best = None
    current_best_fitness = np.inf
    current_best_distance = np.inf
    current_best_height = 0

    with Pool(processes=population_size) as pool:
        for generation in tqdm(range(num_generations), desc="Generations"):
            params = [optimizer.ask() for _ in range(population_size)]
            results = pool.map(
                evaluate_step,
                [(param, number_of_steps, number_of_joints, number_parametric_control_points_spline,
                  fitness_function, flight_path_with_phases) for param in params]
            )
            solutions = []
            end_locations = []
            fitness_values = []
            for i, (fitness, end_pos) in enumerate(results):
                param = params[i]
                solutions.append((param, fitness))
                end_locations.append(end_pos[:2])
                fitness_values.append(fitness)
                if fitness < current_best_fitness:
                    current_best = param
                    current_best_fitness = fitness
                    current_best_distance = np.linalg.norm(end_pos[:2] - start_pos[:2])
                    current_best_height = end_pos[2]

            # Update the optimizer with the solutions
            optimizer.tell(solutions)
            best_fitness = min(s[1] for s in solutions)

            print(f"Generation {generation + 1}, best fitness: {best_fitness}")

            # Log the metrics to wandb
            dictionary = get_metrics(fitness_values)
            dictionary["best_params_distance"] = current_best_distance
            dictionary["best_params_height"] = current_best_height

            wandb.log(data=dictionary, step=generation)
            dictionary["generation"] = generation
            save_dict_data_to_csv(dictionary, "numeric_data", result_directory)

            savable_obj = FlightPathWithPhasesSavableObject(
                not_interpolated_params=current_best,
                position_flying_squirrel=start_pos,
                euler_flying_squirrel=euler_flying_squirrel,
                arena_dim=arena_dim,
                flight_path_with_phases=flight_path_with_phases,
            )

            write_result(
                current_best=savable_obj,
                result_directory=result_directory,
                generation=generation,
                name="flight_path_phases",
                last=True,
            )

            if generation % 100 == 0 or generation == num_generations - 1:
                write_result(
                    current_best=savable_obj,
                    result_directory=result_directory,
                    generation=generation,
                    name="flight_path_phases",
                )


"""
Global part because of the multiprocessing "get_env" function can not be inside the if __name__ == "__main__": block
"""

arena_dim = (40, 40)

descriptor_low = np.array(arena_dim) * -1
descriptor_high = np.array(arena_dim)

start_pos = np.array([0, 0, 100])
euler_flying_squirrel = np.array([0, np.pi / 2, 0])
simulation_time = 10


def get_env():
    return get_flying_squirrel_environment(
        size_arena=arena_dim,
        simulation_time=simulation_time,
        euler_flying_squirrel=euler_flying_squirrel,
        position_flying_squirrel=start_pos,
        simplified_wings=True,
    )


if __name__ == "__main__":
    # Remove all UserWarnings
    warnings.simplefilter("ignore", category=UserWarning)

    num_generations = 1000
    population_size = 32
    fitness_function = FitnessFunction.PHASES

    timestamp = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")

    experiment_run_name = f"flight_path{timestamp}_gen_{num_generations}_popsize_{population_size}_{fitness_function.name}"

    env, env_config = get_env()

    number_of_steps = env_config.total_num_control_steps + 1

    flight_path_with_phases = FlightPathWithPhases(
        start_pos=start_pos,
        freefall_angle=-90,
        stable_glide_angle_alpha=-20,
        landing_angle=5,
        simulation_time=simulation_time,
        number_of_steps=number_of_steps,
        alpha=15
    )

    number_parametric_control_points_spline = len(flight_path_with_phases.get_control_point_location_indexes()) - 2

    number_of_joints = env.action_space.shape[0]
    # Lower and higher bound for every control point
    lower_bound = np.tile(env.action_space.low, number_parametric_control_points_spline)
    higher_bound = np.tile(env.action_space.high, number_parametric_control_points_spline)

    bounds = np.column_stack((lower_bound, higher_bound))

    optimizer = CMA(mean=np.zeros(number_of_joints * number_parametric_control_points_spline),
                    sigma=0.5,
                    bounds=bounds,
                    population_size=population_size,
                    )

    result_directory = get_result_directory("flight_path", experiment_run_name,
                                            os.path.dirname(os.path.abspath(__file__)))
    # Create the directory if it does not exist
    os.makedirs(result_directory, exist_ok=True)

    # Initialize wandb
    wandb.init(project="Masterproef_flight_path", group="flight_path", name=experiment_run_name)

    commit_number = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
    wandb.run.notes = f"Commit: {commit_number}"
    wandb.run.summary["commit_number"] = commit_number

    # Write pdf with url and run name
    write_pdf_with_url(experiment_run_name, wandb.run.get_url(), result_directory)

    run_flight_path()

    env.close()

    wandb.finish()
