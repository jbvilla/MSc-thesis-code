from moojoco.environment.dual import DualMuJoCoEnvironment

from src.flying_squirrel.environment.basic.mjc_env import (
    FlyingSquirrelBasicMJCEnvironment,
)


class FlyingSquirrelEnvironment(DualMuJoCoEnvironment):
    MJC_ENV_CLASS = FlyingSquirrelBasicMJCEnvironment

    def __init__(
            self,
            env: (
                    FlyingSquirrelBasicMJCEnvironment
            ),

    ) -> None:
        # For this project we only support MJC env
        super().__init__(env=env, backend="MJC")
