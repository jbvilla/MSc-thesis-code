import numpy as np
from moojoco.mjcf.arena import ArenaConfiguration, MJCFArena
from typing import Tuple

from src.flying_squirrel.mjcf.arena.target import MJCFTarget


class ForestArenaConfiguration(ArenaConfiguration):
    def __init__(
            self,
            name: str,
            size: Tuple[int, int],
            attach_target: bool = False,
            flight_path: np.ndarray = None,
    ) -> None:
        super().__init__(name=name)
        self.size = size
        self.attach_target = attach_target
        self.flight_path = flight_path


class MJCFForestArena(MJCFArena):
    @property
    def arena_configuration(self) -> ForestArenaConfiguration:
        config = super().arena_configuration
        if isinstance(config, ForestArenaConfiguration):
            return config
        else:
            raise TypeError("Configuration is not of type BasicArenaConfiguration")

    def _build(self, *args, **kwargs) -> None:
        self._configure_cameras()
        self._build_ground()
        self._build_physics()
        self._configure_lights()
        self._build_target()
        self._build_flight_path()

    def _configure_cameras(self) -> None:
        self.mjcf_model.worldbody.add(
            "camera", name="top_camera", pos=[0, 0, 5], quat=[1, 0, 0, 0]
        )
        self.mjcf_model.worldbody.add(
            "camera", name="front_camera", pos=[2, -5, 5], euler=[np.deg2rad(60), 0, 0]
        )

        self.mjcf_model.worldbody.add(
            "camera", name="behind_camera", pos=[-1.5, 0, 6], euler=[0, np.deg2rad(-45), np.deg2rad(-90)]
        )

    def _configure_lights(self) -> None:
        # code from https://github.com/Co-Evolve/brt/blob/3049fd02d37cc457de791f5fcc4af0b750ef28ca/biorobot/brittle_star/mjcf/arena/aquarium.py#L47
        self.mjcf_model.worldbody.add(
            "light",
            pos=[-20, 0, 20],
            directional=True,
            dir=[0, 0, -0.5],
            diffuse=[0.1, 0.1, 0.1],
            castshadow=False,
        )
        self.mjcf_model.visual.headlight.set_attributes(
            ambient=[0.4, 0.4, 0.4], diffuse=[0.8, 0.8, 0.8], specular=[0.1, 0.1, 0.1]
        )

    def _build_ground(self) -> None:
        # code from https://github.com/Co-Evolve/brt/blob/3049fd02d37cc457de791f5fcc4af0b750ef28ca/biorobot/brittle_star/mjcf/arena/aquarium.py#L76
        ground_texture = self.mjcf_model.asset.add(
            "texture",
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
            type="2d",
            builtin="checker",
            name="groundplane",
            width=200,
            height=200,
            mark="edge",
            markrgb=[0.8, 0.8, 0.8],
        )
        ground_material = self.mjcf_model.asset.add(
            "material",
            name="groundplane",
            texrepeat=[2, 2],  # Makes white squares exactly 1x1 length units.
            texuniform=True,
            reflectance=0.2,
            texture=ground_texture,
        )

        # Build groundplane.
        self._ground_geom = self.mjcf_body.add(
            "geom",
            type="plane",
            name="groundplane",
            material=ground_material,
            rgba=None,
            size=list(self.arena_configuration.size) + [0.25],
            condim=6,
            contype=0,
            conaffinity=1,
        )

    def _build_physics(self) -> None:
        self.mjcf_model.option.gravity = [0, 0, -9.81]
        self.mjcf_model.option.density = 1.2
        self.mjcf_model.option.viscosity = 0.00002
        self.mjcf_model.option.wind = [0, 0, 0]

    def _build_target(self) -> None:
        if self.arena_configuration.attach_target:
            self._target = MJCFTarget(
                parent=self,
                name="target",
            )

    def _build_flight_path(self) -> None:
        if self.arena_configuration.flight_path is not None:
            for i, point in enumerate(self.arena_configuration.flight_path):
                self.mjcf_model.worldbody.add(
                    "site",
                    name=f"flight_path_{i}",
                    pos=point,
                    size=[0.03],
                    rgba=[0.5, 0.5, 0.5, 1],
                )


if __name__ == "__main__":
    MJCFForestArena(
        configuration=ForestArenaConfiguration("forest_arena", (10, 10))
    ).export_to_xml_with_assets("./xml")
