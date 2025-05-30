import numpy as np
from moojoco.mjcf.morphology import MJCFMorphology
from transforms3d.euler import euler2quat

from src.kite.mjcf.morphology.parts.arm_beam import MJCFBKiteArmBeam
from src.kite.mjcf.morphology.parts.central_beam import MJCFBKiteCentralBeam

from src.kite.mjcf.morphology.parts.wing import MJCFBKiteWing
from src.kite.mjcf.morphology.specification.default import default_kite_specification
from src.kite.mjcf.morphology.specification.specification import KiteMorphologySpecification


class MJCFKiteMorphology(MJCFMorphology):
    def __init__(self, specification: KiteMorphologySpecification) -> None:
        super().__init__(specification, name="KiteMorphology")

    @property
    def morphology_specification(self) -> KiteMorphologySpecification:
        specification = super().morphology_specification
        if isinstance(specification, KiteMorphologySpecification):
            return specification
        else:
            raise TypeError("Specification is not of type KiteMorphologySpecification.")

    def _build(self, *args, **kwargs) -> None:
        self._configure_compiler()
        self._configure_defaults()
        self._build_central_beam()
        self._build_arms()
        self._build_wing()
        self._configure_camera()

    def _configure_compiler(self) -> None:
        self.mjcf_model.compiler.angle = "radian"

    def _configure_defaults(self) -> None:
        # condim 6 for the most realistic simulation when there is geom contact
        self.mjcf_model.default.geom.condim = 6
        # two geom objects will collide if (contype1 & conaffinity2) || (contype2 & conaffinity1) is not 0
        self.mjcf_model.default.geom.contype = 1
        self.mjcf_model.default.geom.conaffinity = 0

    def _build_central_beam(self) -> None:
        self._central_beam = MJCFBKiteCentralBeam(
            parent=self, name="central_beam", pos=np.zeros(3), euler=np.zeros(3)
        )

    def _build_arms(self) -> None:
        num_arms = len(self.morphology_specification.arm_beams_specification)
        self._arms = []

        # Length is the x-axis, width is the y-axis
        # First quadrant is left top, second quadrant is right top,
        # third quadrant is left bottom, fourth quadrant is right bottom
        positions = [
            [1, 1],  # Top left
            [1, -1],  # Top right
            [-1, 1],  # Bottom left
            [-1, -1]  # Bottom right
        ]

        for i in range(num_arms):
            width_central_beam = self.morphology_specification.central_beam_specification.width.value

            pos_x_length = positions[i][0] * self.morphology_specification.x_position_legs.value
            pos_y_width = positions[i][1] * (width_central_beam / 2 +
                                             self.morphology_specification.arm_beams_specification[i].length.value / 2)

            angle = np.pi / 2

            arm_beam = MJCFBKiteArmBeam(
                parent=self._central_beam,
                name=f"arm_{i}",
                pos=[pos_x_length, pos_y_width, 0],
                euler=[0, 0, angle],
                arm_index=i,
            )

            self._arms.append(arm_beam)

    def _build_wing(self) -> None:
        # The width of the wing is the width of the central beam plus 2 * length of the arm beam
        width = (self.morphology_specification.central_beam_specification.width.value +
                 2 * self.morphology_specification.arm_beams_specification[0].length.value)
        # The length of the wing
        length = self.morphology_specification.x_position_legs.value * 2
        self._wing = MJCFBKiteWing(
            parent=self,
            name="wings",
            pos=[0, 0, self.morphology_specification.central_beam_specification.height.value / 2],
            euler=np.zeros(3),
            width=width,
            length=length
        )

    def _configure_camera(self) -> None:
        self._central_beam.mjcf_body.add(
            "camera",
            name="side_camera",
            pos=[0.0, -1.0, 1.5],
            quat=euler2quat(40 / 180 * np.pi, 0, 0),
            mode="track",
        )


if __name__ == "__main__":
    spec = default_kite_specification()
    MJCFKiteMorphology(spec).export_to_xml_with_assets("./xml")
