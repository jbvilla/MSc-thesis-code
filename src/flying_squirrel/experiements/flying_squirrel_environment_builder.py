from typing import Tuple

import numpy as np

from src.flying_squirrel.environment.basic.mjc_env import FlyingSquirrelBasicMJCEnvironment
from src.flying_squirrel.environment.basic.shared import FlyingSquirrelBasicEnvironmentConfiguration
from src.flying_squirrel.mjcf.arena.forest import ForestArenaConfiguration, MJCFForestArena
from src.flying_squirrel.mjcf.morphology.morphology import MJCFFlyingSquirrelMorphology
from src.flying_squirrel.mjcf.morphology.specification.default import default_flying_squirrel_specification


def get_flying_squirrel_environment(
        size_arena: Tuple[int, int] = (10, 10),
        simulation_time: int = 5,
        num_physics_steps_per_control_step: int = 10,
        camera_ids=None,
        position_flying_squirrel: np.array = np.array([0, 0, 20.0]),
        euler_flying_squirrel: np.array = np.array([0, 0, 0]),
        attach_target: bool = False,
        target_position: np.array = None,
        target_birds_eye_distance: float = 5,
        simplified_wings: bool = False,
        flight_path: np.ndarray = None,
        num_cameras: int = 4,
) -> tuple[FlyingSquirrelBasicMJCEnvironment, FlyingSquirrelBasicEnvironmentConfiguration]:
    """
    Get the flying squirrel environment (MJC)
    :param size_arena: Size of the arena
    :param simulation_time: Simulation time
    :param num_physics_steps_per_control_step: Number of physics steps per control step
    :param camera_ids: Camera ids
    :param position_flying_squirrel: Position of the flying squirrel
    :param euler_flying_squirrel: Euler of the flying squirrel
    :param attach_target: Attach target
    :param target_position: Position of the target
    :param target_birds_eye_distance: Initial distance of the target
    :param simplified_wings: Simplified wings
    :param flight_path: Flight path
    :param num_cameras: Number of cameras around the flying squirrel
    :return: The flying squirrel environment + configuration (don't forget to close it if you don't need it anymore)
    """
    # So no mutable default arguments
    if camera_ids is None:
        camera_ids = [3]

    morphology_spec = default_flying_squirrel_specification()

    arena_config = ForestArenaConfiguration("forest_arena", size_arena, attach_target=attach_target, flight_path=flight_path)

    env_config = FlyingSquirrelBasicEnvironmentConfiguration(
        render_mode="rgb_array",
        simulation_time=simulation_time,
        num_physics_steps_per_control_step=num_physics_steps_per_control_step,
        time_scale=1,
        camera_ids=camera_ids,
        render_size=(480 * 4, 640 * 4),
        attach_target=attach_target,
        target_birds_eye_distance=target_birds_eye_distance,
        target_position=target_position
    )

    morphology = MJCFFlyingSquirrelMorphology(
        specification=morphology_spec,
        euler_flying_squirrel=euler_flying_squirrel,
        num_cameras=num_cameras,
        simplified_wings=simplified_wings,
    )

    arena = MJCFForestArena(configuration=arena_config)

    return FlyingSquirrelBasicMJCEnvironment.from_morphology_and_arena(
        morphology=morphology, arena=arena, configuration=env_config, position=position_flying_squirrel,
        euler=euler_flying_squirrel
    ), env_config
