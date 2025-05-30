from typing import Union
import numpy as np
from moojoco.mjcf.morphology import MJCFMorphologyPart, MJCFMorphology

from src.flying_squirrel.mjcf.morphology.parts.segments.bone_segment import MJCFFlyingSquirrelBoneSegment
from src.flying_squirrel.mjcf.morphology.parts.foot import MJCFFlyingSquirrelFoot
from src.flying_squirrel.mjcf.morphology.specification.specification import FlyingSquirrelMorphologySpecification


class MJCFFlyingSquirrelHindLimb(MJCFMorphologyPart):
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
        Initialize the super class of the MJCFFlyingSquirrelHindLimb
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
        self._build_femur()
        self._build_tibia()
        self._build_foot()

    def _build_femur(self) -> None:
        self._femur = MJCFFlyingSquirrelBoneSegment(
            parent=self,
            name=f"{self.base_name}_femur",
            pos=np.zeros(3),
            euler=np.zeros(3),
            in_plane_joint_axis=self._in_plane_joint_axis,
            out_of_plane_joint_axis=self._out_of_plane_joint_axis,
            bone_segment_specification=self.morphology_specification.hind_limb_specification.femur_specification,
            density=2600
        )

    def _build_tibia(self) -> None:
        femur_length = self._femur.morphology_specification.hind_limb_specification.femur_specification.length.value
        femur_radius = self._femur.morphology_specification.hind_limb_specification.femur_specification.radius.value
        self._tibia = MJCFFlyingSquirrelBoneSegment(
            parent=self._femur,
            name=f"{self.base_name}_tibia",
            pos=[femur_length + femur_radius, 0, 0],
            euler=np.zeros(3),
            in_plane_joint_axis=self._in_plane_joint_axis,
            out_of_plane_joint_axis=self._out_of_plane_joint_axis,
            bone_segment_specification=self.morphology_specification.hind_limb_specification.tibia_specification,
            density=2600
        )

    def _build_foot(self) -> None:
        # Position in parent (tibia)
        tibia_length = self._tibia.morphology_specification.hind_limb_specification.tibia_specification.length.value
        tibia_radius = self._tibia.morphology_specification.hind_limb_specification.tibia_specification.radius.value
        self._foot = MJCFFlyingSquirrelFoot(
            parent=self._tibia,
            name=f"{self.base_name}_foot",
            pos=[tibia_length, 0, -tibia_radius/2],
            euler=np.zeros(3),
            in_plane_joint_axis=self._in_plane_joint_axis,
            out_of_plane_joint_axis=self._out_of_plane_joint_axis
        )
