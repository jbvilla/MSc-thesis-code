import numpy as np

from typing import Any, Dict, Union, List

from moojoco.environment.base import BaseEnvState
from moojoco.environment.mjc_env import MJCEnv, MJCEnvState, MJCObservable

from src.flying_squirrel.environment.basic.shared import (
    FlyingSquirrelBasicEnvironmentBase,
    FlyingSquirrelBasicEnvironmentConfiguration
)
from src.flying_squirrel.environment.shared.observables import get_base_flying_squirrel_observables

from src.flying_squirrel.mjcf.arena.forest import MJCFForestArena
from src.flying_squirrel.mjcf.morphology.morphology import MJCFFlyingSquirrelMorphology
from src.kite.mjcf.morphology.morphology import MJCFKiteMorphology


class FlyingSquirrelBasicMJCEnvironment(
    FlyingSquirrelBasicEnvironmentBase, MJCEnv
):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
            self,
            mjcf_str: str,
            mjcf_assets: Dict[str, Any],
            configuration: FlyingSquirrelBasicEnvironmentConfiguration,
    ) -> None:
        FlyingSquirrelBasicEnvironmentBase.__init__(self)
        MJCEnv.__init__(
            self,
            mjcf_str=mjcf_str,
            mjcf_assets=mjcf_assets,
            configuration=configuration,
        )
        self.mjcf_str = mjcf_str

    @property
    def environment_configuration(
            self,
    ) -> FlyingSquirrelBasicEnvironmentConfiguration:
        config = super(MJCEnv, self).environment_configuration

        if isinstance(config, FlyingSquirrelBasicEnvironmentConfiguration):
            return config
        else:
            raise TypeError("Configuration is not of type FlyingSquirrelBasicEnvironmentConfiguration")

    @classmethod
    def from_morphology_and_arena(
            cls,
            morphology: Union[MJCFKiteMorphology, MJCFFlyingSquirrelMorphology],
            arena: MJCFForestArena,
            configuration: FlyingSquirrelBasicEnvironmentConfiguration,
            position: np.array = np.array([0, 0, 20]),
            euler: np.array = np.array([0, 0, 0]),
    ) -> 'FlyingSquirrelBasicMJCEnvironment':
        arena.attach(other=morphology, free_joint=True, position=position, euler=euler)
        mjcf_str, mjcf_assets = arena.get_mjcf_str(), arena.get_mjcf_assets()
        return cls(
            mjcf_str=mjcf_str, mjcf_assets=mjcf_assets, configuration=configuration
        )

    @staticmethod
    def get_xyz_distance_to_target(state: MJCEnvState) -> float:
        target_position = state.mj_data.site("target_highlight").xpos
        trunk_position = state.mj_data.body("FlyingSquirrelMorphology/trunk").xpos
        distance = np.linalg.norm(target_position - trunk_position)
        return distance

    @staticmethod
    def get_xy_distance_to_target(state: MJCEnvState) -> float:
        target_position = state.mj_data.site("target_highlight").xpos
        trunk_position = state.mj_data.body("FlyingSquirrelMorphology/trunk").xpos
        distance = np.linalg.norm(target_position[:2] - trunk_position[:2])
        return distance

    def _create_observables(self) -> List[MJCObservable]:
        base_observables = get_base_flying_squirrel_observables(
            mj_model=self.frozen_mj_model, backend="mjc"
        )
        return base_observables

    @staticmethod
    def _get_time(state: MJCEnvState) -> MJCEnvState:
        return state.mj_data.time

    def _update_info(self, state: BaseEnvState) -> BaseEnvState:
        info = {
            "time": self._get_time(state=state),
        }

        # noinspection PyUnresolvedReferences
        return state.replace(info=info)

    def _update_reward(
            self, state: BaseEnvState, previous_state: BaseEnvState
    ) -> BaseEnvState:
        reward = 0

        # noinspection PyUnresolvedReferences
        return state.replace(reward=reward)

    def _update_truncated(self, state: BaseEnvState) -> BaseEnvState:
        """
        Update the truncated flag of the state based on the current time.
        """
        truncated = (
                self._get_time(state=state) > self.environment_configuration.simulation_time
        )
        # noinspection PyUnresolvedReferences
        return state.replace(truncated=truncated)

    def _update_terminated(self, state: MJCEnvState) -> MJCEnvState:
        terminated = False
        if self.environment_configuration.target_position is not None:
            # If the target position is not None, check if the distance to the target is less than a given value
            if self.get_xy_distance_to_target(state=state) < 0.05:
                terminated = True

        # noinspection PyUnresolvedReferences
        return state.replace(terminated=terminated)

    def _get_target_position(self, rng: np.random.RandomState) -> np.array:
        if self.environment_configuration.target_position is not None:
            return np.array(self.environment_configuration.target_position)
        else:
            angle = rng.uniform(0, 2 * np.pi)
            radius = self.environment_configuration.target_birds_eye_distance
            return np.array([radius * np.cos(angle), radius * np.sin(angle), 0])

    def reset(self, rng: np.random.RandomState, *args, **kwargs) -> MJCEnvState:
        mj_model, mj_data = self._prepare_reset()

        # Set the target position
        if self.environment_configuration.attach_target:
            target_position = self._get_target_position(rng)
            # If the height is 0, take a random height of the default target tree
            if target_position[2] == 0:
                height_target_highlight = rng.uniform(0, mj_model.geom("target_tree").size[1]*2)
                mj_model.body("target").pos = target_position
                mj_model.site("target_highlight").pos = [0, 0, height_target_highlight]
            # If the height is not 0, the tree is the same height as the target highlight so the target highlight is
            # the highest point on the tree
            else:
                # Size is half of the height
                mj_model.geom("target_tree").size = [mj_model.geom("target_tree").size[0], target_position[2] / 2, 0]
                mj_model.geom("target_tree").pos = [0, 0, target_position[2] / 2]
                mj_model.body("target").pos = [target_position[0], target_position[1], 0]
                mj_model.site("target_highlight").pos = [0, 0, target_position[2]]

        state = self._finish_reset(models_and_datas=(mj_model, mj_data), rng=rng)
        return state

    def write_to_xml(self) -> None:
        with open('test.xml', "w") as f:
            f.write(self.mjcf_str)
