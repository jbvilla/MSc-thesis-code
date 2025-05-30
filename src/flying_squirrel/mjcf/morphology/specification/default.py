"""
Important if you change the default values for the flexcomp model:
- Make sure to remove the temp folder located at src/flying_squirrel/3D_models/temp (mesh for the flexcomp is cached there)
"""

import numpy as np

from src.flying_squirrel.mjcf.morphology.specification.specification import FlyingSquirrelMorphologySpecification, \
    FlyingSquirrelTrunkSpecification, FlyingSquirrelForelimbSpecification, FlyingSquirrelHindLimbSpecification, \
    FlyingSquirrelHeadSpecification, FlyingSquirrelJointSpecification, FlyingSquirrelActuationSpecification, \
    FlyingSquirrelHindLimbTibiaSpecification, FlyingSquirrelHindLimbFemurSpecification, \
    FlyingSquirrelHindLimbFootSpecification, FlyingSquirrelForelimbHumerusSpecification, \
    FlyingSquirrelForelimbRadiusSpecification, FlyingSquirrelForelimbHandSpecification, \
    FlyingSquirrelForelimbClavicleSpecification, FlyingSquirrelTailSegmentSpecification, \
    FlyingSquirrelTailSpecification, FlyingSquirrelPatagiumSpecification

DEFAULT_RADIUS = 0.002
DEFAULT_BODY_PARTS_RADIUS = DEFAULT_RADIUS
RADIUS_HIND_LIMB = 0.006

JOINT_DEGREE_IN_PLANE = 60
JOINT_DEGREE_OUT_OF_PLANE = 30

# Trunk
TRUNK_LENGTH = 0.0722  # 0.0722
TRUNK_WIDTH = DEFAULT_BODY_PARTS_RADIUS
TRUNK_HEIGHT = DEFAULT_BODY_PARTS_RADIUS
# Forelimb
FORELIMB_HUMERUS_LENGTH = 0.0262
FORELIMB_HUMERUS_RADIUS = DEFAULT_BODY_PARTS_RADIUS
FORELIMB_RADIUS_LENGTH = 0.0299
FORELIMB_RADIUS_RADIUS = DEFAULT_BODY_PARTS_RADIUS
FORELIMB_HAND_LENGTH = 0.010108
FORELIMB_HAND_WIDTH = DEFAULT_RADIUS
FORELIMB_HAND_HEIGHT = DEFAULT_RADIUS
FORELIMB_HAND_WING_TIP_LENGTH = FORELIMB_HAND_LENGTH
FORELIMB_HAND_WING_TIP_RADIUS = DEFAULT_RADIUS
CLAVICLE_LENGTH = 0.015
CLAVICLE_RADIUS = DEFAULT_BODY_PARTS_RADIUS
# Hind limb
HIND_LIMB_TIBIA_LENGTH = 0.0355
HIND_LIMB_TIBIA_RADIUS = RADIUS_HIND_LIMB
HIND_LIMB_FEMUR_LENGTH = 0.0307
HIND_LIMB_FEMUR_RADIUS = RADIUS_HIND_LIMB
HIND_LIMB_FOOT_LENGTH = 0.021
HIND_LIMB_FOOT_WIDTH = 0.004
HIND_LIMB_FOOT_HEIGHT = 0.002
# Head
HEAD_LENGTH = 0.032
HEAD_WIDTH = 0.0185
HEAD_HEIGHT = 0.0165
HEAD_RADIUS_CONNECTOR = DEFAULT_RADIUS
# Patagium (simplified because flexcomp could not change thickness when using equality constraint)
PATAGIUM_THICKNESS = 0.001  # this value has a lot of influence on the fluid coefficients


def degree_to_radian(degree: float) -> float:
    return degree / 180 * np.pi


def default_joint_specification(range_min: float, range_max: float) -> FlyingSquirrelJointSpecification:
    joint_specification = FlyingSquirrelJointSpecification(
        range_min=range_min, range_max=range_max, stiffness=0.01, damping=0.1, armature=0.001
    )

    return joint_specification


def default_head_specification() -> FlyingSquirrelHeadSpecification:
    head_specification = FlyingSquirrelHeadSpecification(
        length=HEAD_LENGTH, width=HEAD_WIDTH, height=HEAD_HEIGHT, radius_connector=HEAD_RADIUS_CONNECTOR,
        in_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(30), range_max=degree_to_radian(30)),
        out_of_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(30), range_max=degree_to_radian(30))
    )

    return head_specification


def default_hid_limb_specification() -> FlyingSquirrelHindLimbSpecification:
    tibia_specification = FlyingSquirrelHindLimbTibiaSpecification(length=HIND_LIMB_TIBIA_LENGTH, radius=HIND_LIMB_TIBIA_RADIUS,
            in_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(JOINT_DEGREE_IN_PLANE), range_max=degree_to_radian(0)),
            out_of_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(JOINT_DEGREE_OUT_OF_PLANE), range_max=degree_to_radian(JOINT_DEGREE_OUT_OF_PLANE)))
    femur_specification = FlyingSquirrelHindLimbFemurSpecification(length=HIND_LIMB_FEMUR_LENGTH, radius=HIND_LIMB_FEMUR_RADIUS,
            in_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(JOINT_DEGREE_IN_PLANE), range_max=degree_to_radian(JOINT_DEGREE_IN_PLANE)),
            out_of_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(JOINT_DEGREE_OUT_OF_PLANE), range_max=degree_to_radian(JOINT_DEGREE_OUT_OF_PLANE)))
    foot_specification = FlyingSquirrelHindLimbFootSpecification(length=HIND_LIMB_FOOT_LENGTH, width=HIND_LIMB_FOOT_WIDTH, height=HIND_LIMB_FOOT_HEIGHT, radius_connector=DEFAULT_RADIUS/2,
            in_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(10), range_max=degree_to_radian(100)),
            out_of_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(JOINT_DEGREE_OUT_OF_PLANE), range_max=degree_to_radian(JOINT_DEGREE_OUT_OF_PLANE)))

    hid_limb_specification = FlyingSquirrelHindLimbSpecification(
        tibia_specification=tibia_specification,
        femur_specification=femur_specification,
        foot_specification=foot_specification
    )

    return hid_limb_specification


def default_forelimb_specification() -> FlyingSquirrelForelimbSpecification:
    humerus_specification = FlyingSquirrelForelimbHumerusSpecification(length=FORELIMB_HUMERUS_LENGTH, radius=FORELIMB_HUMERUS_RADIUS,
            in_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(JOINT_DEGREE_IN_PLANE), range_max=degree_to_radian(JOINT_DEGREE_IN_PLANE)),
            out_of_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(JOINT_DEGREE_OUT_OF_PLANE), range_max=degree_to_radian(JOINT_DEGREE_OUT_OF_PLANE)))
    radius_specification = FlyingSquirrelForelimbRadiusSpecification(length=FORELIMB_RADIUS_LENGTH, radius=FORELIMB_RADIUS_RADIUS,
            in_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(0), range_max=degree_to_radian(JOINT_DEGREE_IN_PLANE)),
            out_of_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(JOINT_DEGREE_OUT_OF_PLANE), range_max=degree_to_radian(JOINT_DEGREE_OUT_OF_PLANE)))
    hand_specification = FlyingSquirrelForelimbHandSpecification(length=FORELIMB_HAND_LENGTH, width=FORELIMB_HAND_WIDTH, height=FORELIMB_HAND_HEIGHT, radius_connector=DEFAULT_RADIUS/2, wing_tip_length=FORELIMB_HAND_WING_TIP_LENGTH, wing_tip_radius=FORELIMB_HAND_WING_TIP_RADIUS,
            in_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(JOINT_DEGREE_IN_PLANE), range_max=degree_to_radian(JOINT_DEGREE_IN_PLANE)),
            out_of_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(JOINT_DEGREE_OUT_OF_PLANE), range_max=degree_to_radian(JOINT_DEGREE_OUT_OF_PLANE)))
    clavicle_specification = FlyingSquirrelForelimbClavicleSpecification(length=CLAVICLE_LENGTH, radius=CLAVICLE_RADIUS)

    forelimb_specification = FlyingSquirrelForelimbSpecification(
        humerus_specification=humerus_specification,
        radius_specification=radius_specification,
        hand_specification=hand_specification,
        clavicle_specification=clavicle_specification
    )

    return forelimb_specification


def default_tail_specification(
        segment_c1_c4_length: float = 0.0097,
        segment_c5_c13_length: float = 0.0617,
        segment_c14_c21_length: float = 0.0344) -> FlyingSquirrelTailSpecification:
    tail_segment_specifications = [
        FlyingSquirrelTailSegmentSpecification(
            length=segment_c1_c4_length,
            radius=DEFAULT_RADIUS,
            fur_with=0.018,
            in_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(30), range_max=degree_to_radian(30)),
            out_of_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(30), range_max=degree_to_radian(30))
        ),
        FlyingSquirrelTailSegmentSpecification(
            length=segment_c5_c13_length,
            radius=DEFAULT_RADIUS,
            fur_with=0.020,
            in_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(30), range_max=degree_to_radian(30)),
            out_of_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(30), range_max=degree_to_radian(30))
        ),
        FlyingSquirrelTailSegmentSpecification(
            length=segment_c14_c21_length,
            radius=DEFAULT_RADIUS,
            fur_with=0.018,
            in_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(30), range_max=degree_to_radian(30)),
            out_of_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(30), range_max=degree_to_radian(30))
        )
    ]

    return FlyingSquirrelTailSpecification(
        tail_segment_specifications=tail_segment_specifications
    )


def default_patagium_specification() -> FlyingSquirrelPatagiumSpecification:
    return FlyingSquirrelPatagiumSpecification(
        thickness=PATAGIUM_THICKNESS,
        in_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(5),
                                                                 range_max=degree_to_radian(5)),
        out_of_plane_joint_specification=default_joint_specification(range_min=degree_to_radian(30),
                                                                     range_max=degree_to_radian(30)),
        out_of_plane_pitch_joint_specification=default_joint_specification(range_min=degree_to_radian(5),
                                                                           range_max=degree_to_radian(5))
    )


def default_flying_squirrel_specification(radius_to_strength_factor: float = 200) -> FlyingSquirrelMorphologySpecification:
    """
    Default specification for the morphology of the flying squirrel.
    """
    trunk_specification = FlyingSquirrelTrunkSpecification(length=TRUNK_LENGTH, width=TRUNK_WIDTH, height=TRUNK_HEIGHT)
    forelimb_specification = default_forelimb_specification()
    hind_limb_specification = default_hid_limb_specification()
    head_specification = default_head_specification()
    tail_specification = default_tail_specification()
    actuation_specification = FlyingSquirrelActuationSpecification(radius_to_strength_factor=radius_to_strength_factor)

    return FlyingSquirrelMorphologySpecification(
        trunk_specification=trunk_specification,
        forelimb_specification=forelimb_specification,
        hind_limb_specification=hind_limb_specification,
        head_specification=head_specification,
        tail_specification=tail_specification,
        patagium_specification=default_patagium_specification(),
        actuation_specification=actuation_specification
    )
