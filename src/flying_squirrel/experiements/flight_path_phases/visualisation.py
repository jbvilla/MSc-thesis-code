import os

from src.flying_squirrel.experiements.flight_path_phases.flight_path_with_phases_savable_object import FlightPathWithPhasesSavableObject
from src.utils.load_write import load_pickle
from src.utils.visualise import render_environment


def create_video_best_individual(savable: FlightPathWithPhasesSavableObject):
    # Create a directory to save the visualisations
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "visualisations")
    os.makedirs(results_dir, exist_ok=True)
    # Get the environment
    env, env_config = savable.get_env()

    result_name = os.path.join(results_dir, "visualisation")

    params = savable.get_interpolated_params()

    render_environment(env, env_config, actions=params, name_video=result_name)

    env.close()


if __name__ == "__main__":
    savable = load_pickle("results/flight_path2025_05_20_19h11m08s_gen_1000_popsize_32_PHASES/params/flight_path_phases_last.pkl")
    savable.position_flying_squirrel = [0, 0, 30]
    create_video_best_individual(savable)
