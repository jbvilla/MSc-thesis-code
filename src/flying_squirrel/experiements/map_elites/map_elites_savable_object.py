from typing import Tuple, List

from src.flying_squirrel.experiements.flying_squirrel_environment_builder import get_flying_squirrel_environment
from src.flying_squirrel.experiements.map_elites.map_elites import MAPElitesState
from src.flying_squirrel.experiements.shared.interpolation import interpolate_params


class MAPElitesSavableObject:
    """
    The object that is saved for re-running the parameters of the MAP-Elites algorithm
    """

    def __init__(
            self,
            state: MAPElitesState,
            number_of_steps: int,
            number_of_joints: int,
            number_parametric_control_points_spline: int,
            euler_flying_squirrel: list,
            position_flying_squirrel: list,
            simulation_time: int,
            arena_dim: Tuple[int, int],
            simplified_wings: bool,
            start_vel: List,
    ):
        self.state = state
        self.number_of_steps = number_of_steps
        self.number_of_joints = number_of_joints
        self.number_parametric_control_points_spline = number_parametric_control_points_spline
        self.euler_flying_squirrel = euler_flying_squirrel
        self.position_flying_squirrel = position_flying_squirrel
        self.simulation_time = simulation_time
        self.arena_dim = arena_dim
        self.simplified_wings = simplified_wings
        self.start_vel = start_vel

    def get_interpolated_params(self, params):
        """
        Get the interpolated parameters for the MAP-Elites algorithm
        """
        env, _ = self.get_env()

        return interpolate_params(
            params,
            self.number_of_steps,
            self.number_of_joints,
            self.number_parametric_control_points_spline,
            False, )

    def get_env(self):
        """
        Get flying squirrel environment
        """
        return get_flying_squirrel_environment(
            size_arena=self.arena_dim,
            simulation_time=self.simulation_time,
            euler_flying_squirrel=self.euler_flying_squirrel,
            position_flying_squirrel=self.position_flying_squirrel,
            simplified_wings=self.simplified_wings,
        )
