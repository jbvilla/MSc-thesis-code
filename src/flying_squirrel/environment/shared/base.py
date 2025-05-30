from moojoco.environment.base import MuJoCoEnvironmentConfiguration


class FlyingSquirrelEnvironmentBaseConfiguration(MuJoCoEnvironmentConfiguration):
    def __init__(
            self,
            solver_iterations: int = 1,
            solver_ls_iterations: int = 5,
            *args,
            **kwargs
    ):
        super().__init__(
            disable_eulerdamp=True,
            solver_iterations=solver_iterations,
            solver_ls_iterations=solver_ls_iterations,
            *args,
            **kwargs,
        )


class FlyingSquirrelEnvironmentBase:
    def __init__(self):
        pass
