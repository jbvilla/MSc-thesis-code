from typing import Tuple

import numpy as np

from src.flying_squirrel.experiements.flying_squirrel_environment_builder import get_flying_squirrel_environment
from src.flying_squirrel.experiements.shared.interpolation import interpolate_params


class CMAESSavableObject:

    def __init__(
            self,
            not_interpolated_params: np.ndarray,
            number_of_steps: int,
            number_of_joints: int,
            number_parametric_control_points_spline: int,
            euler_flying_squirrel: list,
            position_flying_squirrel: list,
            start_vel: np.ndarray,
            simulation_time: int,
            arena_dim: Tuple[int, int],
            simplified_wings: bool,
            mirror_limbs: bool,
    ):
        self.not_interpolated_params = not_interpolated_params
        self.number_of_steps = number_of_steps
        self.number_of_joints = number_of_joints
        self.number_parametric_control_points_spline = number_parametric_control_points_spline
        self.euler_flying_squirrel = euler_flying_squirrel
        self.position_flying_squirrel = position_flying_squirrel
        self.start_vel = start_vel
        self.simulation_time = simulation_time
        self.arena_dim = arena_dim
        self.simplified_wings = simplified_wings
        self.mirror_limbs = mirror_limbs

    def get_interpolated_params(self):
        """
        Get the interpolated parameters for the MAP-Elites algorithm
        """
        env, _ = self.get_env()

        return interpolate_params(
            self.not_interpolated_params,
            self.number_of_steps,
            self.number_of_joints,
            self.number_parametric_control_points_spline,
            self.mirror_limbs, )

    def get_env(self):
        """
        Get flying squirrel environment
        """
        return get_flying_squirrel_environment(
            size_arena=self.arena_dim,
            simulation_time=self.simulation_time,
            euler_flying_squirrel=self.euler_flying_squirrel,
            simplified_wings=self.simplified_wings,
            position_flying_squirrel=self.position_flying_squirrel,
        )
