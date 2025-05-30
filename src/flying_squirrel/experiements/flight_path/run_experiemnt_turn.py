import os
import subprocess
import warnings
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import wandb
from cmaes import CMA
from tqdm import tqdm

from src.flying_squirrel.experiements.flight_path.flight_path_savable_object import FlightPathSavableObject
from src.flying_squirrel.experiements.flying_squirrel_environment_builder import get_flying_squirrel_environment
from src.flying_squirrel.experiements.shared.evaluate import evaluate_params, FitnessFunction
from src.flying_squirrel.experiements.shared.flight_path import FlightPath, FlightPathModes
from src.flying_squirrel.experiements.shared.wandb_utils import get_metrics
from src.utils.load_write import get_result_directory, write_pdf_with_url, save_dict_data_to_csv, write_result


def evaluate_step(args):
    (param, number_of_steps, number_of_joints,
     number_parametric_control_points_spline, fitness_function, flight_path_turn, start_vel) = args
    env, _ = get_env()
    return evaluate_params(param, number_of_steps, number_of_joints,
                           number_parametric_control_points_spline, env, fitness_function,
                           start_vel, False, None, flight_path_turn)


def run_flight_path_turn():
    current_best = None
    current_best_fitness = np.inf

    with Pool(processes=population_size) as pool:
        for generation in tqdm(range(num_generations), desc="Generations"):
            params = [optimizer.ask() for _ in range(population_size)]
            results = pool.map(
                evaluate_step,
                [(param, number_of_steps, number_of_joints, number_parametric_control_points_spline,
                  fitness_function, flight_path, start_vel) for param in params]
            )
            solutions = []
            fitness_values = []
            for i, (fitness, _) in enumerate(results):
                param = params[i]
                solutions.append((param, fitness))
                fitness_values.append(fitness)
                if fitness < current_best_fitness:
                    current_best = param
                    current_best_fitness = fitness

            # Update the optimizer with the solutions
            optimizer.tell(solutions)
            best_fitness = min(s[1] for s in solutions)

            print(f"Generation {generation + 1}, best fitness: {best_fitness}")

            # Log the metrics to wandb
            dictionary = get_metrics(fitness_values)

            wandb.log(data=dictionary, step=generation)
            dictionary["generation"] = generation
            save_dict_data_to_csv(dictionary, "numeric_data", result_directory)

            savable_obj = FlightPathSavableObject(
                not_interpolated_params=current_best,
                number_of_steps=number_of_steps,
                number_of_joints=number_of_joints,
                number_parametric_control_points_spline=number_parametric_control_points_spline,
                euler_flying_squirrel=euler_flying_squirrel,
                arena_dim=arena_dim,
                simulation_time=simulation_time,
                start_vel=start_vel,
                flight_path=flight_path,
            )

            write_result(
                current_best=savable_obj,
                result_directory=result_directory,
                generation=generation,
                name="flight_path",
                last=True,
            )

            if generation % 100 == 0 or generation == num_generations - 1:
                write_result(
                    current_best=savable_obj,
                    result_directory=result_directory,
                    generation=generation,
                    name="flight_path",
                )


"""
Global part because of the multiprocessing "get_env" function can not be inside the if __name__ == "__main__": block
"""

arena_dim = (40, 40)

descriptor_low = np.array(arena_dim) * -1
descriptor_high = np.array(arena_dim)

start_pos = np.array([0, 0, 35])
target_pos = np.array([7, 6, 28])
turn_distance_from_center = 2.0
flight_path_mode = FlightPathModes.LEFT_TURN
spacing = 0.005
euler_flying_squirrel = np.array([0, 0, 0])
start_vel = [3, 0, 0]
simulation_time = 5
number_parametric_control_points_spline = 2


def get_env():
    return get_flying_squirrel_environment(
        size_arena=arena_dim,
        simulation_time=simulation_time,
        euler_flying_squirrel=euler_flying_squirrel,
        attach_target=True,
        target_position=target_pos,
        simplified_wings=True,
    )


if __name__ == "__main__":
    # Remove all UserWarnings
    warnings.simplefilter("ignore", category=UserWarning)

    num_generations = 1000
    population_size = 32
    fitness_function = FitnessFunction.PATH

    timestamp = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")

    experiment_run_name = f"flight_turn{timestamp}_gen_{num_generations}_popsize_{population_size}_{str(flight_path_mode)}"

    env, env_config = get_env()

    number_of_steps = env_config.total_num_control_steps + 1

    flight_path = FlightPath(flight_path_mode, start_pos, target_pos, 20, spacing, turn_distance_from_center=turn_distance_from_center)

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

    result_directory = get_result_directory("flight_path_turn", experiment_run_name,
                                            os.path.dirname(os.path.abspath(__file__)))
    # Create the directory if it does not exist
    os.makedirs(result_directory, exist_ok=True)

    # Initialize wandb
    wandb.init(project="Masterproef_flight_path_turn", group="flight_path", name=experiment_run_name)

    commit_number = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
    wandb.run.notes = f"Commit: {commit_number}"
    wandb.run.summary["commit_number"] = commit_number

    # Write pdf with url and run name
    write_pdf_with_url(experiment_run_name, wandb.run.get_url(), result_directory)

    run_flight_path_turn()

    env.close()

    wandb.finish()
