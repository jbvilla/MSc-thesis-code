import os
from typing import Union
import numpy as np
from dm_control.mjcf.element import _ElementImpl
from moojoco.mjcf.morphology import MJCFMorphologyPart, MJCFMorphology

from src.flying_squirrel.mjcf.morphology.specification.specification import FlyingSquirrelMorphologySpecification, \
    FlyingSquirrelJointSpecification
from src.utils import colors


class MJCFFlyingSquirrelSimplifiedPatagium(MJCFMorphologyPart):
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
        Initialize the super class of the MJCFFlyingSquirrelSimplifiedPatagium
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

    def _build(
            self,
            corner_points: np.ndarray,
            in_plane_joint_axis: np.ndarray,
            out_of_plane_joint_axis: np.ndarray,
            out_of_plane_pitch_joint_axis: np.ndarray,
            *args, **kwargs
    ) -> None:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        self._simplified_patagium_path = os.path.join(current_directory, "..", "..", "..", "3D_models", "temp",
                                                      f"simplified_patagium.obj")

        self._wing_length = np.abs(corner_points[0][1])
        self._in_plane_joint_axis = in_plane_joint_axis
        self._out_of_plane_joint_axis = out_of_plane_joint_axis
        self._out_of_plane_pitch_joint_axis = out_of_plane_pitch_joint_axis
        self._build_simplified_patagium(corner_points)
        self._configure_joints()
        self._configure_actuators()

    def _build_simplified_patagium(self, corner_points: np.ndarray) -> None:

        thickness = self.morphology_specification.patagium_specification.thickness.value

        top_right = corner_points[1]
        bottom_right = corner_points[2]
        # On the x-axis
        top_middle = [top_right[0], 0, top_right[2]]
        bottom_middle = [bottom_right[0], 0, bottom_right[2]]

        name = f"{self.base_name}_simplified_patagium"

        # Define the vertices of the mesh

        # This gives a good mesh but the physics is not correct
        vertex = f"{top_middle[0]} {top_middle[1]} {top_middle[2] + thickness}  " \
                 f"{top_right[0]} {top_right[1]} {top_right[2] + thickness} " \
                 f"{bottom_right[0]} {bottom_right[1]} {bottom_right[2] + thickness} " \
                 f"{bottom_middle[0]} {bottom_middle[1]} {bottom_middle[2] + thickness} " \
                 f"{top_middle[0]} {top_middle[1]} {top_middle[2] - thickness} " \
                 f"{top_right[0]} {top_right[1]} {top_right[2] - thickness} " \
                 f"{bottom_right[0]} {bottom_right[1]} {bottom_right[2] - thickness} " \
                 f"{bottom_middle[0]} {bottom_middle[1]} {bottom_middle[2] - thickness}"

        # To prevent qhull error add a small offset to the points in the middle
        temp = 1e-6
        #vertex = f"{top_middle[0]} {top_middle[1]} {top_middle[2] + temp}  " \
        #         f"{top_right[0]} {top_right[1]} {top_right[2]} " \
        #         f"{bottom_right[0]} {bottom_right[1]} {bottom_right[2]} " \
        #         f"{bottom_middle[0]} {bottom_middle[1]} {bottom_middle[2] + temp} " \

        self.mjcf_model.asset.add(
            "mesh",
            name=name,
            vertex=vertex,
        )

        self._simplified_patagium = self.mjcf_body.add(
            "geom",
            type="mesh",
            name=f"{self.base_name}_mesh",
            mesh=name,
            rgba=colors.rgba_green,
            contype=1,
            conaffinity=0,
            density=300,
            fluidshape="ellipsoid",
            fluidcoef="0.5 0.25 1.5 2.8 1.0",
            pos=np.zeros(3),
            euler=np.zeros(3),
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
        patagium_specification = self.morphology_specification.patagium_specification

        self._in_plane_joint = self._configure_joint(
            name=f"{self.base_name}_in_plane_joint",
            axis=self._in_plane_joint_axis,
            joint_specification=patagium_specification.in_plane_joint_specification,
        )

        self._out_of_plane_joint = self._configure_joint(
            name=f"{self.base_name}_out_of_plane_joint",
            axis=self._out_of_plane_joint_axis,
            joint_specification=patagium_specification.out_of_plane_joint_specification,
        )

        self._out_of_plane_pitch_joint = self._configure_joint(
            name=f"{self.base_name}_out_of_plane_pitch_joint",
            axis=self._out_of_plane_pitch_joint_axis,
            joint_specification=patagium_specification.out_of_plane_pitch_joint_specification,
        )

    def _get_strength(self) -> float:
        return (self._wing_length
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
        self._configure_position_control_actuator(self._out_of_plane_pitch_joint)
