import os

from src.flying_squirrel.experiements.cma_es.cma_es_savable_object import CMAESSavableObject
from src.flying_squirrel.experiements.shared.interpolation import interpolate_params
from src.utils.load_write import load_pickle
from src.utils.visualise import render_environment


def create_video_best_individual(savable: CMAESSavableObject):
    # Create a directory to save the visualisations
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "visualisations")
    os.makedirs(results_dir, exist_ok=True)
    # Get the environment
    env, env_config = savable.get_env()

    result_name = os.path.join(results_dir, "visualisation")

    params = interpolate_params(
        savable.not_interpolated_params,
        savable.number_of_steps,
        savable.number_of_joints,
        savable.number_parametric_control_points_spline,
        simplified=savable.mirror_limbs
    )

    render_environment(env, env_config, actions=params, initial_vel=savable.start_vel, name_video=result_name)

    env.close()


if __name__ == "__main__":
    savable = load_pickle("../../../visualize_experiments/data/gliding_straight/cma-es_last_2.pkl")
    create_video_best_individual(savable)
