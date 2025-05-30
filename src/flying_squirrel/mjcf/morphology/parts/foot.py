from typing import Union
import numpy as np
from dm_control.mjcf.element import _ElementImpl
from moojoco.mjcf.morphology import MJCFMorphologyPart, MJCFMorphology

from src.flying_squirrel.mjcf.morphology.specification.specification import FlyingSquirrelMorphologySpecification, \
    FlyingSquirrelJointSpecification
from src.utils import colors


class MJCFFlyingSquirrelFoot(MJCFMorphologyPart):
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
        Initialize the super class of the MJCFFlyingSquirrelFoot
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

    def _build(self, in_plane_joint_axis, out_of_plane_joint_axis, *args, **kwargs) -> None:
        self._in_plane_joint_axis = in_plane_joint_axis
        self._out_of_plane_joint_axis = out_of_plane_joint_axis
        self._foot_specification = self.morphology_specification.hind_limb_specification.foot_specification
        self._build_box()
        self._build_connector()
        self._configure_joints()
        self._configure_actuators()

    def _build_box(self) -> None:
        length = self._foot_specification.length.value
        width = self._foot_specification.width.value
        height = self._foot_specification.height.value

        self._box = self.mjcf_body.add(
            "geom",
            name=f"{self.base_name}_box",
            type="box",
            size=[length / 2, width / 2, height / 2],
            pos=[length / 2, 0, -height / 2],
            euler=np.zeros(3),
            rgba=colors.rgba_green,
            density=1100,
            fluidshape="ellipsoid",
        )

    def _build_connector(self) -> None:
        radius = self._foot_specification.radius_connector.value
        self._connector = self.mjcf_body.add(
            "geom",
            name=f"{self.base_name}_connector",
            type="sphere",
            pos=[radius, 0, radius],
            size=[radius / 2],
            rgba=colors.rgba_gray,
        )

    def _configure_joint(
            self,
            name: str,
            axis: np.array,
            joint_specification: FlyingSquirrelJointSpecification,
    ) -> _ElementImpl:
        radius = self._foot_specification.radius_connector.value
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
            pos=[radius, 0, radius],
        )

    def _configure_joints(self) -> None:
        self._in_plane_joint = self._configure_joint(
            name=f"{self.base_name}_in_plane_joint",
            axis=self._in_plane_joint_axis,
            joint_specification=self._foot_specification.in_plane_joint_specification,
        )

        self._out_of_plane_joint = self._configure_joint(
            name=f"{self.base_name}_out_of_plane_joint",
            axis=self._out_of_plane_joint_axis,
            joint_specification=self._foot_specification.out_of_plane_joint_specification,
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