from typing import Union
import numpy as np
from moojoco.mjcf.morphology import MJCFMorphologyPart, MJCFMorphology

from src.kite.mjcf.morphology.specification.specification import KiteMorphologySpecification
from src.utils import colors


class MJCFBKiteCentralBeam(MJCFMorphologyPart):
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
        Initialize the super class of the MJCFBKiteCentralBeam
        :param parent: parent morphology
        :param name: name of the MJCFBKiteCentralBeam
        :param pos: position of the body in the parent frame
        :param euler: euler angles of the body in the parent frame
        :param args:
        :param kwargs:
        """
        super().__init__(parent, name, pos, euler, *args, **kwargs)

    @property
    def morphology_specification(self) -> KiteMorphologySpecification:
        specification = super().morphology_specification
        if isinstance(specification, KiteMorphologySpecification):
            return specification
        else:
            raise TypeError("Specification is not of type KiteMorphologySpecification.")

    def _build(self) -> None:
        self._beam_specification = self.morphology_specification.central_beam_specification
        self._build_beam()

    def _build_beam(self) -> None:
        length = self._beam_specification.length.value
        width = self._beam_specification.width.value
        height = self._beam_specification.height.value

        # X half-size; Y half-size; Z half-size.
        size = [length/2, width/2, height/2]

        self.mjcf_body.add(
            "geom",
            type="box",
            name=f"{self.base_name}_beam",
            # position of the beam in the body frame
            pos=np.zeros(3),
            size=size,
            rgba=colors.rgba_green,
            contype=1,
            conaffinity=0,
            density=1100,
            fluidshape="ellipsoid"
        )
