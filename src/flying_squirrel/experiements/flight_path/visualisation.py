import os

import numpy as np

from src.flying_squirrel.experiements.flight_path.flight_path_savable_object import FlightPathSavableObject
from src.utils.load_write import load_pickle
from src.utils.visualise import render_environment


def create_video_best_individual(savable: FlightPathSavableObject):
    # Create a directory to save the visualisations
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "visualisations")
    os.makedirs(results_dir, exist_ok=True)
    # Get the environment
    env, env_config = savable.get_env(True)

    result_name = os.path.join(results_dir, "visualisation")

    params = savable.get_interpolated_params()

    render_environment(env, env_config, actions=params, name_video=result_name, initial_vel=savable.start_vel)

    env.close()


if __name__ == "__main__":
    savable = load_pickle("../../../visualize_experiments/data/left_turn/left_turn.pkl")
    savable = load_pickle("results/flight_turn2025_05_19_01h19m09s_gen_1000_popsize_32_FlightPathModes.LEFT_TURN/params/flight_path_last.pkl")
    create_video_best_individual(savable)
