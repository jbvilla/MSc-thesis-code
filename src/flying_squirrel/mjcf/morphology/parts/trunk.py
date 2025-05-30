from typing import Union
import numpy as np
from moojoco.mjcf.morphology import MJCFMorphologyPart, MJCFMorphology

from src.flying_squirrel.mjcf.morphology.specification.specification import FlyingSquirrelMorphologySpecification
from src.utils import colors


class MJCFFlyingSquirrelTrunk(MJCFMorphologyPart):
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
        Initialize the super class of the MJCFFlyingSquirrelTrunk
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
        self._build_trunk()
        self._configure_sensors()

    def _build_trunk(self) -> None:
        trunk_specification = self.morphology_specification.trunk_specification
        length = trunk_specification.length.value
        width = trunk_specification.width.value
        height = trunk_specification.height.value

        # X half-size; Y half-size; Z half-size.
        size = [length / 2, width / 2, height / 2]

        self.mjcf_body.add(
            "geom",
            type="box",
            name=f"{self.base_name}",
            pos=np.zeros(3),
            size=size,
            rgba=colors.rgba_green,
            contype=1,
            conaffinity=0,
            density=1100,
            fluidshape="ellipsoid"
        )

    def _configure_sensors(self) -> None:
        self.mjcf_model.sensor.add(
            "framepos",
            name=f"{self.base_name}_framepos_sensor",
            objtype="xbody",
            objname=self.mjcf_body.name,
        )
        self.mjcf_model.sensor.add(
            "framequat",
            name=f"{self.base_name}_framequat_sensor",
            objtype="xbody",
            objname=self.mjcf_body.name,
        )
        self.mjcf_model.sensor.add(
            "frameangvel",
            name=f"{self.base_name}_frameangvel_sensor",
            objtype="xbody",
            objname=self.mjcf_body.name,
        )
        self.mjcf_model.sensor.add(
            "framelinvel",
            name=f"{self.base_name}_framelinvel_sensor",
            objtype="xbody",
            objname=self.mjcf_body.name,
        )
