from typing import Union
import numpy as np
from moojoco.mjcf.morphology import MJCFMorphologyPart, MJCFMorphology

from src.flying_squirrel.mjcf.morphology.parts.segments.bone_segment import MJCFFlyingSquirrelBoneSegment
from src.flying_squirrel.mjcf.morphology.parts.hand import MJCFFlyingSquirrelHand
from src.flying_squirrel.mjcf.morphology.specification.specification import FlyingSquirrelMorphologySpecification
from src.utils import colors


class MJCFFlyingSquirrelForelimb(MJCFMorphologyPart):
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
        Initialize the super class of the MJCFFlyingSquirrelForelimb
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
        self.build_clavicle()
        self._build_humerus()
        self._build_radius()
        self._build_hand()

    def build_clavicle(self) -> None:
        clavicle_length = self.morphology_specification.forelimb_specification.clavicle_specification.length.value
        clavicle_radius = self.morphology_specification.forelimb_specification.clavicle_specification.radius.value
        self._clavicle = self.mjcf_body.add(
            "geom",
            name=f"{self.base_name}_clavicle",
            type="capsule",
            pos=[clavicle_radius / 2 + clavicle_length / 2, 0, 0],
            euler=[0, np.pi / 2, 0],
            size=[clavicle_radius / 2, clavicle_length / 2],
            rgba=colors.rgba_green,
            density=1100,
            fluidshape="ellipsoid",
        )

    def _build_humerus(self) -> None:
        clavicle_length = self.morphology_specification.forelimb_specification.clavicle_specification.length.value
        clavicle_radius = self.morphology_specification.forelimb_specification.clavicle_specification.radius.value
        self._humerus = MJCFFlyingSquirrelBoneSegment(
            parent=self,
            name=f"{self.base_name}_humerus",
            pos=[clavicle_length + clavicle_radius, 0, 0],
            euler=np.zeros(3),
            in_plane_joint_axis=self._in_plane_joint_axis,
            out_of_plane_joint_axis=self._out_of_plane_joint_axis,
            bone_segment_specification=self.morphology_specification.forelimb_specification.humerus_specification
        )

    def _build_radius(self) -> None:
        humerus_length = self._humerus.morphology_specification.forelimb_specification.humerus_specification.length.value
        humerus_radius = self._humerus.morphology_specification.forelimb_specification.humerus_specification.radius.value
        self._radius = MJCFFlyingSquirrelBoneSegment(
            parent=self._humerus,
            name=f"{self.base_name}_radius",
            pos=[humerus_length + humerus_radius, 0, 0],
            euler=np.zeros(3),
            in_plane_joint_axis=self._in_plane_joint_axis,
            out_of_plane_joint_axis=self._out_of_plane_joint_axis,
            bone_segment_specification=self.morphology_specification.forelimb_specification.radius_specification
        )

    def _build_hand(self) -> None:
        radius_length = self._radius.morphology_specification.forelimb_specification.radius_specification.length.value
        radius_radius = self._radius.morphology_specification.forelimb_specification.radius_specification.radius.value
        self._hand = MJCFFlyingSquirrelHand(
            parent=self._radius,
            name=f"{self.base_name}_hand",
            pos=[radius_length + radius_radius, 0, 0],
            euler=[0, 0, -np.deg2rad(145)],
            in_plane_joint_axis=self._in_plane_joint_axis,
            out_of_plane_joint_axis=self._out_of_plane_joint_axis
        )

