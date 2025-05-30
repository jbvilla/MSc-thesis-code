import numpy as np

from src.flying_squirrel.environment.shared.base import (
    FlyingSquirrelEnvironmentBaseConfiguration, FlyingSquirrelEnvironmentBase,
)


class FlyingSquirrelBasicEnvironmentConfiguration(FlyingSquirrelEnvironmentBaseConfiguration):
    def __init__(
            self,
            attach_target: bool = True,
            target_birds_eye_distance: float = 5,
            target_position: np.ndarray | None = None,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.attach_target = attach_target
        self.target_birds_eye_distance = target_birds_eye_distance
        self.target_position = target_position


class FlyingSquirrelBasicEnvironmentBase(FlyingSquirrelEnvironmentBase):
    def __init__(self):
        super().__init__()
