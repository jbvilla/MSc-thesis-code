from typing import Union
import numpy as np
from dm_control.mjcf.element import _ElementImpl
from moojoco.mjcf.morphology import MJCFMorphologyPart, MJCFMorphology

from src.flying_squirrel.mjcf.morphology.specification.specification import FlyingSquirrelMorphologySpecification, \
    FlyingSquirrelJointSpecification
from src.utils import colors


class MJCFFlyingSquirrelHead(MJCFMorphologyPart):
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
        Initialize the super class of the MJCFFlyingSquirrelHead
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

    def _build(self, *args, **kwargs) -> None:
        self._build_ellipsoid()
        self._build_connector()
        self._configure_joints()
        self._configure_actuators()

    def _build_ellipsoid(self) -> None:
        head_specification = self.morphology_specification.head_specification
        length = head_specification.length.value
        width = head_specification.width.value
        height = head_specification.height.value

        # self.mjcf_body.add(
        #    "geom",
        #    name=f"{self.base_name + '_neck'}",
        #    type="cylinder",
        #    size=[0.001, 0.0194/2],
        #    pos=[0.0194/2, 0, 0],
        #    euler=[0, np.pi/2, 0],
        #    rgba=colors.rgba_green,
        # )

        self.mjcf_body.add(
            "geom",
            type="ellipsoid",
            name=f"{self.base_name}_ellipsoid",
            pos=[length/2, 0, 0],
            size=[length/2, width/2, height/2],
            rgba=colors.rgba_green,
            contype=1,
            conaffinity=0,
            density=1100,
            fluidshape="ellipsoid",
        )

    def _build_connector(self) -> None:
        head_specification = self.morphology_specification.head_specification
        radius = head_specification.radius_connector.value

        self.mjcf_body.add(
            "geom",
            name=f"{self.base_name}_connector",
            type="sphere",
            pos=np.zeros(3),
            size=[0.5 * radius],
            rgba=colors.rgba_gray,
            contype=0,
            conaffinity=0,
        )

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
        head_specification = self.morphology_specification.head_specification

        self._in_plane_joint = self._configure_joint(
            name=f"{self.base_name}_in_plane_joint",
            axis=[0, 0, 1],
            joint_specification=head_specification.in_plane_joint_specification,
        )

        self._out_of_plane_joint = self._configure_joint(
            name=f"{self.base_name}_out_of_plane_joint",
            axis=[0, -1, 0],
            joint_specification=head_specification.out_of_plane_joint_specification,
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
