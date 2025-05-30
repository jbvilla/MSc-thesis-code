from typing import Tuple

import numpy as np

from src.flying_squirrel.experiements.flying_squirrel_environment_builder import get_flying_squirrel_environment
from src.flying_squirrel.experiements.shared.flight_path import FlightPathWithPhases
from src.flying_squirrel.experiements.shared.interpolation import interpolate_params


class FlightPathWithPhasesSavableObject:

    def __init__(
            self,
            not_interpolated_params: np.ndarray,
            position_flying_squirrel: np.ndarray,
            euler_flying_squirrel: np.ndarray,
            arena_dim: Tuple[int, int],
            flight_path_with_phases: FlightPathWithPhases,
    ):
        self.not_interpolated_params = not_interpolated_params
        self.position_flying_squirrel = position_flying_squirrel
        self.euler_flying_squirrel = euler_flying_squirrel
        self.arena_dim = arena_dim
        self.flight_path_with_phases = flight_path_with_phases

    def get_env(self):
        """
        Get flying squirrel environment
        """
        return get_flying_squirrel_environment(
            size_arena=self.arena_dim,
            simulation_time=self.flight_path_with_phases.simulation_time,
            euler_flying_squirrel=self.euler_flying_squirrel,
            simplified_wings=True,
            position_flying_squirrel=self.position_flying_squirrel,
        )

    def get_interpolated_params(self):
        """
        Interpolates the points between the control points for each joint
        """
        env, _ = self.get_env()
        number_of_joints = env.action_space.shape[0]

        return interpolate_params(
            self.not_interpolated_params,
            self.flight_path_with_phases.number_of_steps,
            number_of_joints,
            len(self.flight_path_with_phases.get_control_point_location_indexes()) - 2,
            False,
            self.flight_path_with_phases)

