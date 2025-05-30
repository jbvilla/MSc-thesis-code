from typing import List

from fprs.parameters import FixedParameter
from fprs.specification import MorphologySpecification, Specification


class FlyingSquirrelJointSpecification(Specification):
    """
    Specification for the joint of the flying squirrel.
    """

    def __init__(self, range_min: float, range_max: float, stiffness: float, damping: float, armature: float) -> None:
        super().__init__()
        self.stiffness = FixedParameter(value=stiffness)
        self.damping = FixedParameter(value=damping)
        self.armature = FixedParameter(value=armature)
        self.range_min = FixedParameter(value=range_min)
        self.range_max = FixedParameter(value=range_max)


class FlyingSquirrelActuationSpecification(Specification):
    """
    Specification for the actuation of the flying squirrel.
    """

    def __init__(self, radius_to_strength_factor: float) -> None:
        super().__init__()
        self.radius_to_strength_factor = FixedParameter(value=radius_to_strength_factor)


class FlyingSquirrelTrunkSpecification(Specification):
    """
    Specification for the trunk of the flying squirrel.
    """

    def __init__(self, length: float, width: float, height: float) -> None:
        super().__init__()
        self.length = FixedParameter(length)
        self.width = FixedParameter(width)
        self.height = FixedParameter(height)


class FlyingSquirrelForelimbHumerusSpecification(Specification):
    """
    Specification for the humerus of the forelimb of the flying squirrel.
    """

    def __init__(self, length: float, radius: float,
                 in_plane_joint_specification: FlyingSquirrelJointSpecification,
                 out_of_plane_joint_specification: FlyingSquirrelJointSpecification) -> None:
        super().__init__()
        self.length = FixedParameter(length)
        self.radius = FixedParameter(radius)
        self.in_plane_joint_specification = in_plane_joint_specification
        self.out_of_plane_joint_specification = out_of_plane_joint_specification


class FlyingSquirrelForelimbRadiusSpecification(Specification):
    """
    Specification for the radius of the forelimb of the flying squirrel.
    """

    def __init__(self, length: float, radius: float,
                 in_plane_joint_specification: FlyingSquirrelJointSpecification,
                 out_of_plane_joint_specification: FlyingSquirrelJointSpecification) -> None:
        super().__init__()
        self.length = FixedParameter(length)
        self.radius = FixedParameter(radius)
        self.in_plane_joint_specification = in_plane_joint_specification
        self.out_of_plane_joint_specification = out_of_plane_joint_specification


class FlyingSquirrelForelimbHandSpecification(Specification):
    """
    Specification for the hand of the forelimb of the flying squirrel.
    """

    def __init__(
            self,
            length: float,
            width: float,
            height: float,
            radius_connector: float,
            wing_tip_length: float,
            wing_tip_radius: float,
            in_plane_joint_specification: FlyingSquirrelJointSpecification,
            out_of_plane_joint_specification: FlyingSquirrelJointSpecification) -> None:
        super().__init__()
        self.length = FixedParameter(length)
        self.width = FixedParameter(width)
        self.height = FixedParameter(height)
        self.radius_connector = FixedParameter(radius_connector)
        self.wing_tip_length = FixedParameter(wing_tip_length)
        self.wing_tip_radius = FixedParameter(wing_tip_radius)
        self.in_plane_joint_specification = in_plane_joint_specification
        self.out_of_plane_joint_specification = out_of_plane_joint_specification


class FlyingSquirrelForelimbClavicleSpecification(Specification):
    """
    Specification for the clavicle of the forelimb of the flying squirrel.
    """

    def __init__(self, length: float, radius: float) -> None:
        super().__init__()
        self.length = FixedParameter(length)
        self.radius = FixedParameter(radius)


class FlyingSquirrelForelimbSpecification(Specification):
    """
    Specification for the forelimb of the flying squirrel.
    """

    def __init__(
            self,
            humerus_specification: FlyingSquirrelForelimbHumerusSpecification,
            radius_specification: FlyingSquirrelForelimbRadiusSpecification,
            hand_specification: FlyingSquirrelForelimbHandSpecification,
            clavicle_specification: FlyingSquirrelForelimbClavicleSpecification) -> None:
        super().__init__()
        self.humerus_specification = humerus_specification
        self.radius_specification = radius_specification
        self.hand_specification = hand_specification
        self.clavicle_specification = clavicle_specification


class FlyingSquirrelHindLimbTibiaSpecification(Specification):
    """
    Specification for the tibia of the hind limb of the flying squirrel.
    """

    def __init__(self, length: float, radius: float,
                 in_plane_joint_specification: FlyingSquirrelJointSpecification,
                 out_of_plane_joint_specification: FlyingSquirrelJointSpecification) -> None:
        super().__init__()
        self.length = FixedParameter(length)
        self.radius = FixedParameter(radius)
        self.in_plane_joint_specification = in_plane_joint_specification
        self.out_of_plane_joint_specification = out_of_plane_joint_specification


class FlyingSquirrelHindLimbFemurSpecification(Specification):
    """
    Specification for the femur of the hind limb of the flying squirrel.
    """

    def __init__(self, length: float, radius: float,
                 in_plane_joint_specification: FlyingSquirrelJointSpecification,
                 out_of_plane_joint_specification: FlyingSquirrelJointSpecification) -> None:
        super().__init__()
        self.length = FixedParameter(length)
        self.radius = FixedParameter(radius)
        self.in_plane_joint_specification = in_plane_joint_specification
        self.out_of_plane_joint_specification = out_of_plane_joint_specification


class FlyingSquirrelHindLimbFootSpecification(Specification):
    """
    Specification for the foot of the hind limb of the flying squirrel.
    """

    def __init__(self, length: float, width: float, height: float, radius_connector: float,
                 in_plane_joint_specification: FlyingSquirrelJointSpecification,
                 out_of_plane_joint_specification: FlyingSquirrelJointSpecification) -> None:
        super().__init__()
        self.length = FixedParameter(length)
        self.width = FixedParameter(width)
        self.height = FixedParameter(height)
        self.radius_connector = FixedParameter(radius_connector)
        self.in_plane_joint_specification = in_plane_joint_specification
        self.out_of_plane_joint_specification = out_of_plane_joint_specification


class FlyingSquirrelHindLimbSpecification(Specification):
    """
    Specification for the hind limb of the flying squirrel.
    It consists of the tibia, femur, and foot.
    """

    def __init__(self,
                 tibia_specification: FlyingSquirrelHindLimbTibiaSpecification,
                 femur_specification: FlyingSquirrelHindLimbFemurSpecification,
                 foot_specification: FlyingSquirrelHindLimbFootSpecification) -> None:
        super().__init__()
        self.tibia_specification = tibia_specification
        self.femur_specification = femur_specification
        self.foot_specification = foot_specification


class FlyingSquirrelHeadSpecification(Specification):
    """
    Specification for the head of the flying squirrel.
    """

    def __init__(self,
                 length: float,
                 width: float,
                 height: float,
                 radius_connector,
                 in_plane_joint_specification: FlyingSquirrelJointSpecification,
                 out_of_plane_joint_specification: FlyingSquirrelJointSpecification) -> None:
        super().__init__()
        self.length = FixedParameter(length)
        self.width = FixedParameter(width)
        self.height = FixedParameter(height)
        self.radius_connector = FixedParameter(radius_connector)
        self.in_plane_joint_specification = in_plane_joint_specification
        self.out_of_plane_joint_specification = out_of_plane_joint_specification


class FlyingSquirrelTailSegmentSpecification(Specification):
    """
    Specification for the segment of the tail of the flying squirrel.
    """

    def __init__(
            self,
            length: float,
            radius: float,
            fur_with: float,
            in_plane_joint_specification: FlyingSquirrelJointSpecification,
            out_of_plane_joint_specification: FlyingSquirrelJointSpecification) -> None:
        super().__init__()
        self.length = FixedParameter(length)
        self.radius = FixedParameter(radius)
        self.fur_with = FixedParameter(fur_with)
        self.in_plane_joint_specification = in_plane_joint_specification
        self.out_of_plane_joint_specification = out_of_plane_joint_specification


class FlyingSquirrelTailSpecification(Specification):
    """
    Specification for the tail of the flying squirrel.
    """

    def __init__(
            self,
            tail_segment_specifications: List[FlyingSquirrelTailSegmentSpecification]) -> None:
        super().__init__()
        self.tail_segment_specifications = tail_segment_specifications


class FlyingSquirrelPatagiumSpecification(Specification):
    """
    Specification for the patagium of the flying squirrel.
    """

    def __init__(
            self,
            thickness: float,
            in_plane_joint_specification: FlyingSquirrelJointSpecification,
            out_of_plane_joint_specification: FlyingSquirrelJointSpecification,
            out_of_plane_pitch_joint_specification: FlyingSquirrelJointSpecification
    ) -> None:
        super().__init__()
        self.thickness = FixedParameter(thickness)
        self.in_plane_joint_specification = in_plane_joint_specification
        self.out_of_plane_joint_specification = out_of_plane_joint_specification
        self.out_of_plane_pitch_joint_specification = out_of_plane_pitch_joint_specification


class FlyingSquirrelMorphologySpecification(MorphologySpecification):
    """
    Specification for the morphology of the flying squirrel.
    """

    def __init__(self,
                 trunk_specification: FlyingSquirrelTrunkSpecification,
                 forelimb_specification: FlyingSquirrelForelimbSpecification,
                 hind_limb_specification: FlyingSquirrelHindLimbSpecification,
                 head_specification: FlyingSquirrelHeadSpecification,
                 tail_specification: FlyingSquirrelTailSpecification,
                 patagium_specification: FlyingSquirrelPatagiumSpecification,
                 actuation_specification: FlyingSquirrelActuationSpecification) -> None:
        super(FlyingSquirrelMorphologySpecification, self).__init__()
        self.trunk_specification = trunk_specification
        self.forelimb_specification = forelimb_specification
        self.hind_limb_specification = hind_limb_specification
        self.head_specification = head_specification
        self.tail_specification = tail_specification
        self.patagium_specification = patagium_specification
        self.actuation_specification = actuation_specification
