from typing import Union
import numpy as np
from dm_control.mjcf.element import _ElementImpl
from moojoco.mjcf.morphology import MJCFMorphologyPart, MJCFMorphology

from src.flying_squirrel.mjcf.morphology.specification.specification import FlyingSquirrelMorphologySpecification, \
    FlyingSquirrelJointSpecification
from src.utils import colors


class MJCFFlingSquirrelTailSegment(MJCFMorphologyPart):
    def __init__(
            self,
            parent: Union[MJCFMorphology, MJCFMorphologyPart],
            name: str,
            pos: np.array,
            euler: np.array,
            *args,
            **kwargs,
    ) -> None:
        """
        Initialize the super class of the MJCFFlingSquirrelTailSegment
        :param parent: parent morphology
        :param name: name of part
        :param pos: position of the body in the parent frame
        :param euler: euler angles of the body in the parent frame
        :param args:
        :param kwargs:
        """
        super().__init__(parent, name, pos, euler, *args, **kwargs)

    @property
    def morphology_specification(self) -> FlyingSquirrelMorphologySpecification:
        specification = super().morphology_specification
        if isinstance(specification, FlyingSquirrelMorphologySpecification):
            return specification
        else:
            raise TypeError("Specification is not of type FlyingSquirrelMorphologySpecification.")

    def _build(self, index: int, *args, **kwargs) -> None:
        self._tail_segment_specification = self.morphology_specification.tail_specification.tail_segment_specifications[index]
        self._build_capsule()
        self._build_connector()
        self._build_fur_temp()
        self._configure_joints()
        self._configure_actuators()

    def _build_capsule(self) -> None:
        radius = self._tail_segment_specification.radius.value
        length = self._tail_segment_specification.length.value

        self._capsule = self.mjcf_body.add(
            "geom",
            name=f"{self.base_name}_capsule",
            type="capsule",
            pos=[radius / 2 + length / 2, 0, 0],
            euler=[0, np.pi / 2, 0],
            size=[radius / 2, length / 2],
            rgba=colors.rgba_green,
            density=1100,
            fluidshape="ellipsoid",
        )

    def _build_connector(self) -> None:
        radius = self._tail_segment_specification.radius.value
        self._connector = self.mjcf_body.add(
            "geom",
            name=f"{self.base_name}_connector",
            type="sphere",
            pos=np.zeros(3),
            size=[radius/2],
            rgba=colors.rgba_gray,
        )

    def _build_fur_temp(self):
        length = self._tail_segment_specification.length.value + self._tail_segment_specification.radius.value
        self.mjcf_body.add(
            "geom",
            name=f"{self.base_name}_fur",
            type="box",
            pos=[length / 2, 0, 0],
            size=[length/2, self._tail_segment_specification.fur_with.value / 2, self._tail_segment_specification.radius.value / 4],
            rgba=colors.rgba_green,
            density=300,
            fluidshape="ellipsoid",
        )

    def _build_fur(self) -> None:
        length_points = 11
        width_points = 11

        length = self._tail_segment_specification.length.value + self._tail_segment_specification.radius.value
        # - 1 because for example 6 points have 5 spaces between them
        spacing_x_length = length / (length_points - 1)
        spacing_y_width = self._tail_segment_specification.fur_with.value / (width_points - 1)

        flexcomp = self.mjcf_body.add(
            "flexcomp",
            name=f"{self.base_name}_soft_tissue",
            type="grid",
            dim="2",
            count=[length_points, width_points, 1],
            # mass=1,
            # inertiabox="0.01",
            spacing=[spacing_x_length, spacing_y_width, 0.01],
            pos=[length / 2, 0, 0],
            rgba=[0.0, 0.0, 1.0, 1.0],
        )

        flexcomp.add("contact", contype=1, conaffinity=0, selfcollide="none")
        flexcomp.add("edge", equality="true")
        # add pins to the middle of the grid
        for i in range(length_points):
            flexcomp.add("pin", grid=f"{i} {int(width_points / 2)}")

    def _configure_joint(
            self,
            name: str,
            axis: np.array,
            joint_specification: FlyingSquirrelJointSpecification,
    ) -> _ElementImpl:
        return self.mjcf_body.add(
            "joint",
            name=name,
            type="hinge",
            limited=True,
            range=[-joint_specification.range_min.value, joint_specification.range_max.value],
            axis=axis,
            stiffness=joint_specification.stiffness.value,
            damping=joint_specification.damping.value,
            armature=joint_specification.armature.value,
            pos=np.zeros(3),
        )

    def _configure_joints(self) -> None:
        self._in_plane_joint = self._configure_joint(
            name=f"{self.base_name}_in_plane_joint",
            axis=[0, 0, 1],
            joint_specification=self._tail_segment_specification.in_plane_joint_specification,
        )

        self._out_of_plane_joint = self._configure_joint(
            name=f"{self.base_name}_out_of_plane_joint",
            axis=[0, -1, 0],
            joint_specification=self._tail_segment_specification.out_of_plane_joint_specification,
        )

    def _get_strength(self) -> float:
        return (self.morphology_specification.head_specification.height.value
                * self.morphology_specification.actuation_specification.radius_to_strength_factor.value)

    def _configure_position_control_actuator(self, joint: _ElementImpl) -> _ElementImpl:
        return self.mjcf_model.actuator.add(
            "position",
            name=f"{joint.name}_p_control",
            joint=joint,
            kp=50,
            ctrllimited=True,
            ctrlrange=joint.range,
            forcelimited=True,
            forcerange=[-self._get_strength(), self._get_strength()],
        )

    def _configure_actuators(self) -> None:
        self._configure_position_control_actuator(self._in_plane_joint)
        self._configure_position_control_actuator(self._out_of_plane_joint)
