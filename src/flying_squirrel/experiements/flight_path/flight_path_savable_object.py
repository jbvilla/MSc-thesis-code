from typing import Tuple

import numpy as np

from src.flying_squirrel.experiements.flying_squirrel_environment_builder import get_flying_squirrel_environment
from src.flying_squirrel.experiements.shared.flight_path import FlightPath
from src.flying_squirrel.experiements.shared.interpolation import interpolate_params


class FlightPathSavableObject:

    def __init__(
            self,
            not_interpolated_params: np.ndarray,
            number_of_steps: int,
            number_of_joints: int,
            number_parametric_control_points_spline: int,
            euler_flying_squirrel: np.ndarray,
            arena_dim: Tuple[int, int],
            simulation_time: int,
            start_vel: list,
            flight_path: FlightPath
    ):
        self.not_interpolated_params = not_interpolated_params
        self.number_of_steps = number_of_steps
        self.number_of_joints = number_of_joints
        self.number_parametric_control_points_spline = number_parametric_control_points_spline
        self.euler_flying_squirrel = euler_flying_squirrel
        self.arena_dim = arena_dim
        self.simulation_time = simulation_time
        self.start_vel = start_vel
        self.flight_path = flight_path

    def get_env(self, without_target=False):
        """
        Get flying squirrel environment
        """
        if without_target:
            return get_flying_squirrel_environment(
                size_arena=self.arena_dim,
                simulation_time=self.simulation_time,
                euler_flying_squirrel=self.euler_flying_squirrel,
                simplified_wings=True,
                position_flying_squirrel=self.flight_path.start_pos,
            )
        return get_flying_squirrel_environment(
            size_arena=self.arena_dim,
            simulation_time=self.simulation_time,
            euler_flying_squirrel=self.euler_flying_squirrel,
            simplified_wings=True,
            position_flying_squirrel=self.flight_path.start_pos,
            attach_target=True,
            target_position=self.flight_path.end_pos,
            flight_path=self.flight_path.get_interpolate_path(0.5, "quadratic"),
        )

    def get_interpolated_params(self):
        """
        Interpolates the points between the control points for each joint
        """
        env, _ = self.get_env()

        return interpolate_params(
            self.not_interpolated_params,
            self.number_of_steps,
            self.number_of_joints,
            self.number_parametric_control_points_spline,
            False,)
