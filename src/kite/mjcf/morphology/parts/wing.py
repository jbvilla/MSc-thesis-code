from typing import Union
import numpy as np
import os
from moojoco.mjcf.morphology import MJCFMorphologyPart, MJCFMorphology

from src.kite.mjcf.morphology.specification.specification import KiteMorphologySpecification
from src.utils.generate_pin_ranges_obj import generate_pin_ranges_kite

current_directory = os.path.dirname(os.path.abspath(__file__))
wings_path = os.path.join(current_directory, "wings.obj")


class MJCFBKiteWing(MJCFMorphologyPart):
    def __init__(
            self,
            parent: Union[MJCFMorphology, MJCFMorphologyPart],
            name: str,
            pos: np.array,
            euler: np.array,
            width: float,
            length: float,
            *args,
            **kwargs,
    ) -> None:
        """
        Initialize the super class of the MJCFBKiteCentralBeam
        :param parent: parent morphology
        :param name: name of the MJCFBKiteCentralBeam
        :param pos: position of the body in the parent frame
        :param euler: euler angles of the body in the parent frame
        :param width: width of the wing (only used for the grid)
        :param length: length of the wing (only used for the grid)
        :param args:
        :param kwargs:
        """
        self.width = width
        self.length = length
        super().__init__(parent, name, pos, euler, *args, **kwargs)

    @property
    def morphology_specification(self) -> KiteMorphologySpecification:
        specification = super().morphology_specification
        if isinstance(specification, KiteMorphologySpecification):
            return specification
        else:
            raise TypeError("Specification is not of type KiteMorphologySpecification.")

    def _build(self) -> None:
        self._build_soft_tissue()

    def _build_soft_tissue(self) -> None:

        flexcomp = self.mjcf_body.add(
            "flexcomp",
            name=f"{self.base_name}_soft_tissue",
            type="mesh",
            file=wings_path,
            dim="2",
            mass=0.01,
            inertiabox="0.01",
            # z is the half of the height of the wing
            pos="0 0 0.005",
            rgba=[0.0, 0.0, 1.0, 1.0],
        )

        flexcomp.add("contact", contype=1, conaffinity=0, selfcollide="none")
        flexcomp.add("edge", equality="true")

        # Pin id for a mesh is the index of the point in the mesh
        generated_pin_ranges = generate_pin_ranges_kite(wings_path)
        for start, end in generated_pin_ranges:
            if start == end:
                flexcomp.add("pin", id=f"{start}")
            else:
                flexcomp.add("pin", range=f"{start} {end}")

    def _build_soft_tissue_grid(self) -> None:

        length_points = 11
        width_points = 11
        # - 1 because for example 6 points have 5 spaces between them
        spacing_x_length = self.length / (length_points - 1)
        spacing_y_width = self.width / (width_points - 1)

        flexcomp = self.mjcf_body.add(
            "flexcomp",
            name=f"{self.base_name}_soft_tissue",
            type="grid",
            dim="2",
            count=[length_points, width_points, 1],
            mass=1,
            inertiabox="0.01",
            spacing=[spacing_x_length, spacing_y_width, 0.01],
            # If fully connected on 0 0 0 it just flies up
            pos="0 0 0.005",
            rgba=[0.0, 0.0, 1.0, 1.0],
        )

        flexcomp.add("contact", contype=1, conaffinity=0, selfcollide="none")
        flexcomp.add("edge", equality="true")

        # The grid is defined as follows:
        # y ^
        #   |
        #   |
        #   ----> x
        # 0 0 is the left bottom corner
        # 0 1 is the one above it
        # n m is the top right corner with m being the number of rows (y) and n being the number of columns (x)

        for j in range(width_points):
            # Add the pins for the top and bottom
            flexcomp.add("pin", grid=f"{0} {j}")
            flexcomp.add("pin", grid=f"{length_points-1} {j}")
        for i in range(length_points):
            flexcomp.add("pin", grid=f"{i} {int(width_points/2)}")

        # TODO maybe not needed if everything is connected
        # self.mjcf_body.add(
        #     "inertial",
        #     pos="0 0 0",
        #     mass=0.2,
        #     # The inertia matrix: how bigger the value, how harder it is to rotate around that axis.
        #     diaginertia="0.01 0.015 0.005",
        # )
