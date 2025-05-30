from typing import List

from fprs.specification import MorphologySpecification, Specification
from fprs.parameters import FixedParameter


class KiteBeamSpecification(Specification):
    """
    Specification for the beams used in the kite.
    """

    def __init__(self, length: float, width: float, height: float) -> None:
        super().__init__()
        self.length = FixedParameter(length)
        self.width = FixedParameter(width)
        self.height = FixedParameter(height)


class KiteMorphologySpecification(MorphologySpecification):
    """
    Specification for the morphology of the kite.

    One central beam and two beams on top and bottom of the central beam, to get a rectangular shape.
    """

    def __init__(self,
                 central_beam_specification: KiteBeamSpecification,
                 arm_beams_specification: List[KiteBeamSpecification],
                 x_position_legs: float) -> None:
        """
        Initialize the KiteMorphologySpecification.
        :param central_beam_specification: specification for the central beam
        :param arm_beams_specification: specification for the arm beams
        :param x_position_legs: x position of the legs (where the legs are attached to the central beam)
        """
        super(KiteMorphologySpecification, self).__init__()
        self.central_beam_specification = central_beam_specification
        self.arm_beams_specification = arm_beams_specification
        self.x_position_legs = FixedParameter(x_position_legs)
