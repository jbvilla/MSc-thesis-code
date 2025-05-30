import os
import subprocess
import warnings
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import wandb
from cmaes import CMA
from tqdm import tqdm

from src.flying_squirrel.experiements.cma_es.cma_es_savable_object import CMAESSavableObject
from src.flying_squirrel.experiements.flying_squirrel_environment_builder import get_flying_squirrel_environment
from src.flying_squirrel.experiements.shared.evaluate import evaluate_params, FitnessFunction
from src.flying_squirrel.experiements.shared.interpolation import interpolate_params, prepare_control_points, \
    get_time_points
from src.flying_squirrel.experiements.shared.wandb_utils import get_metrics, create_generation_end_location_heatmap, \
    create_spline_plots
from src.utils.load_write import get_result_directory, write_pdf_with_url, save_dict_data_to_csv, write_result


def evaluate_step(args):
    (params, number_of_steps, number_of_joints,
     number_parametric_control_points_spline, fitness_function, start_vel, simplified) = args
    env, _ = get_env()
    return evaluate_params(params, number_of_steps, number_of_joints,
                           number_parametric_control_points_spline, env, fitness_function, start_vel, simplified)


def run_cma_es():
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
                  fitness_function, start_vel, mirror_limbs) for param in params]
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
            # -best_fitness so it is again positive
            print(f"Generation {generation + 1}, best fitness: {-best_fitness}")

            # Log the metrics to wandb
            dictionary = get_metrics(fitness_values)
            dictionary["best_params_distance"] = current_best_distance
            dictionary["best_params_height"] = current_best_height

            wandb.log(data=dictionary, step=generation)
            dictionary["generation"] = generation
            save_dict_data_to_csv(dictionary, "numeric_data", result_directory)

            savable_obj = CMAESSavableObject(
                not_interpolated_params=current_best,
                number_of_steps=number_of_steps,
                number_of_joints=number_of_joints,
                number_parametric_control_points_spline=number_parametric_control_points_spline,
                euler_flying_squirrel=euler_flying_squirrel,
                position_flying_squirrel=start_pos,
                start_vel=start_vel,
                simulation_time=simulation_time,
                arena_dim=arena_dim,
                simplified_wings=simplified_wings,
                mirror_limbs=mirror_limbs,
            )

            write_result(
                current_best=savable_obj,
                result_directory=result_directory,
                generation=generation,
                name="cma-es",
                last=True,
            )

            if generation % 10 == 0 or generation == num_generations - 1:
                write_result(
                    current_best=savable_obj,
                    result_directory=result_directory,
                    generation=generation,
                    name="cma-es",
                )

                heatmap = create_generation_end_location_heatmap(end_locations, generation, arena_dim, result_directory)
                wandb.log({"heatmap": wandb.Image(heatmap)}, step=generation)

            if generation % 100 == 0 or generation == num_generations - 1:
                interpolated_params = interpolate_params(current_best, number_of_steps, number_of_joints,
                                                         number_parametric_control_points_spline, mirror_limbs)

                control_points = prepare_control_points(current_best, number_of_joints,
                                                        number_parametric_control_points_spline)

                # number_parametric_control_points_spline + 1 because we add control point 0 at the start
                create_spline_plots(interpolated_params,
                                    control_points,
                                    get_time_points(number_of_steps, number_parametric_control_points_spline + 1),
                                    generation,
                                    env.actuators, mirror_limbs)

                # Does not work on the HPC because of the display variable
                vsc_data_path = os.environ.get('VSC_DATA')
                # if not vsc_data_path:
                #     create_video(env, env_config, interpolated_params, start_vel, generation)


"""
Global part because of the multiprocessing "get_env" function can not be inside the if __name__ == "__main__": block
"""

arena_dim = (40, 40)

number_parametric_control_points_spline = 3

descriptor_low = np.array(arena_dim) * -1
descriptor_high = np.array(arena_dim)

euler_flying_squirrel = [0, np.pi / 2, 0]
start_pos = [0, 0, 20]
simulation_time = 15
simplified_wings = True
mirror_limbs = False


def get_env():
    return get_flying_squirrel_environment(
        size_arena=arena_dim,
        simulation_time=simulation_time,
        euler_flying_squirrel=euler_flying_squirrel,
        simplified_wings=simplified_wings,
    )


if __name__ == "__main__":
    # Remove all UserWarnings
    warnings.simplefilter("ignore", category=UserWarning)

    num_generations = 1000
    population_size = 32
    fitness_function = FitnessFunction.X_DISTANCE

    start_vel = np.array([0, 0, 0])

    if mirror_limbs and simplified_wings:
        raise ValueError("Mirror limbs can only be used for the full model (not simplified wings).")

    timestamp = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")

    experiment_run_name = f"CMA-ES_{timestamp}_gen_{num_generations}_start_vel_{'_'.join(map(str, start_vel))}_eul_{'_'.join(map(lambda x: str(round(x, 2)), euler_flying_squirrel))}_popsize_{population_size}_{fitness_function.name}{'_simpl_wings' if simplified_wings else ''}"

    env, env_config = get_env()

    number_of_steps = env_config.total_num_control_steps + 1

    if mirror_limbs:
        # Simplified number of joints
        # -12 because limbs are mirrored so only half of the joints are generated
        number_of_joints = env.action_space.shape[0] - 12

        # Lower and higher bound for every control point
        low = np.concatenate([env.action_space.low[:6], env.action_space.low[12:18], env.action_space.low[24:]])
        high = np.concatenate([env.action_space.high[:6], env.action_space.high[12:18], env.action_space.high[24:]])
        lower_bound = np.tile(low, number_parametric_control_points_spline)
        higher_bound = np.tile(high, number_parametric_control_points_spline)
    else:
        # Not simplified number of joints
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

    result_directory = get_result_directory("cma-es", experiment_run_name,
                                            os.path.dirname(os.path.abspath(__file__)))
    # Create the directory if it does not exist
    os.makedirs(result_directory, exist_ok=True)

    # Initialize wandb
    wandb.init(project="Masterproef", group="CMA-ES", name=experiment_run_name)

    commit_number = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
    wandb.run.notes = f"Commit: {commit_number}"
    wandb.run.summary["commit_number"] = commit_number

    # Write pdf with url and run name
    write_pdf_with_url(experiment_run_name, wandb.run.get_url(), result_directory)

    run_cma_es()

    env.close()

    wandb.finish()
