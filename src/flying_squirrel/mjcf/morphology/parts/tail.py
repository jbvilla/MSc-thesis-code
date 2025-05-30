from typing import Union
import numpy as np
from moojoco.mjcf.morphology import MJCFMorphologyPart, MJCFMorphology

from src.flying_squirrel.mjcf.morphology.parts.segments.tail_segment import MJCFFlingSquirrelTailSegment
from src.flying_squirrel.mjcf.morphology.specification.specification import FlyingSquirrelMorphologySpecification


class MJCFFlyingSquirrelTail(MJCFMorphologyPart):
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
        Initialize the super class of the MJCFFlyingSquirrelTail
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
        self._tail_specification = self.morphology_specification.tail_specification
        self._build_tail()

    def _build_tail(self) -> None:
        parent = self
        length_pos = 0
        for index, tail_segment_specification in enumerate(self._tail_specification.tail_segment_specifications):
            parent = MJCFFlingSquirrelTailSegment(
                parent=parent,
                name=f"tail_segment_{index}",
                pos=np.array([length_pos, 0, 0]),
                euler=np.zeros(3),
                index=index,
            )
            length_pos = tail_segment_specification.length.value + tail_segment_specification.radius.value
