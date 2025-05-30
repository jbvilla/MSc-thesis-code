# Default specification for the morphology of the kite.
from src.kite.mjcf.morphology.specification.specification import KiteBeamSpecification, KiteMorphologySpecification

CENTRAL_BEAM_LENGTH = 0.25
CENTRAL_BEAM_WIDTH = 0.005
CENTRAL_BEAM_HEIGHT = 0.005

BEAM_LENGTH = 0.075
BEAM_WIDTH = 0.005
BEAM_HEIGHT = 0.005

X_POSITION_LEGS = 0.1


def default_kite_specification() -> KiteMorphologySpecification:
    """
    Default specification for the morphology of the kite.
    :return: KiteMorphologySpecification
    """
    central_beam_specification = KiteBeamSpecification(
        length=CENTRAL_BEAM_LENGTH,
        width=CENTRAL_BEAM_WIDTH,
        height=CENTRAL_BEAM_HEIGHT
    )

    arm_beams_specification = KiteBeamSpecification(
        length=BEAM_LENGTH,
        width=BEAM_WIDTH,
        height=BEAM_HEIGHT
    )

    return KiteMorphologySpecification(
        central_beam_specification=central_beam_specification,
        arm_beams_specification=[
            arm_beams_specification, arm_beams_specification,
            arm_beams_specification, arm_beams_specification
        ],
        x_position_legs=X_POSITION_LEGS
    )
