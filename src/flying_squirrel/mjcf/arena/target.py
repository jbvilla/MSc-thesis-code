from typing import Union

from moojoco.mjcf.component import MJCFSubComponent, MJCFRootComponent


class MJCFTarget(MJCFSubComponent):
    """
    Target for the flying squirrel to reach
    """

    def __init__(
            self,
            parent: Union[MJCFSubComponent, MJCFRootComponent],
            name: str,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(parent, name, *args, **kwargs)

    def _build(self, *args, **kwargs) -> None:
        self._target = self.mjcf_body.add(
            "geom",
            name=f"{self.base_name + '_tree'}",
            type="cylinder",
            size=[0.03, 0.5],
            pos=[0.0, 0.0, 0.5],
            rgba=[0.35, 0.16, 0.04, 1],
        )

        self._highlight = self.mjcf_body.add(
            "site",
            name=f"{self.base_name + '_highlight'}",
            size=[0.04],
            pos=[0.0, 0.0, 0.8],
            rgba=[1, 0, 0, 1],
        )
