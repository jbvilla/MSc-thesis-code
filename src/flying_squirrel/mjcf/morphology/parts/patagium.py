"""
Important: patagium.obj is made so te points are in the middle of the object, so it is easier to connect them to
the limbs
"""
import time
from typing import Union
import numpy as np
import os
import trimesh
from moojoco.mjcf.morphology import MJCFMorphologyPart, MJCFMorphology

from src.flying_squirrel.mjcf.morphology.specification.specification import FlyingSquirrelMorphologySpecification
from src.utils import colors
from src.utils.generate_pin_ranges_obj import generate_pin_ranges_flying_squirrel, \
    generate_connects_flying_squirrel_in_range
from src.utils.mesh_utils import subdivide_update_mesh, get_max_distance_edge


class MJCFFlyingSquirrelPatagium(MJCFMorphologyPart):
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
        Initialize the super class of the MJCFFlyingSquirrelPatagium
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

    def _build(self, corner_points, left_forelimb, right_forelimb, left_hind_limb, right_hind_limb, subdivisions_count: int = 3, *args, **kwargs) -> None:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        self._patagium_path = os.path.join(current_directory, "..", "..", "..", "3D_models", "temp", f"patagium_subdiv_{subdivisions_count}.obj")

        self._left_forelimb = left_forelimb
        self._right_forelimb = right_forelimb
        self._left_hind_limb = left_hind_limb
        self._right_hind_limb = right_hind_limb
        self._build_soft_tissue(corner_points, subdivisions_count)
        self._connect_forelimbs()
        self._connect_hind_limb()

    def _create_mesh(self, corner_points: np.ndarray, subdivisions_count: int) -> None:
        """
        Create simple mesh for the patagium that exists of 6 vertices and 4 faces (so there is a straight line in the
        middle of the patagium that can be pinned for the trunk)
        :param corner_points:
        :return:
        """
        top_left = corner_points[0]
        top_right = corner_points[1]
        bottom_right = corner_points[2]
        bottom_left = corner_points[3]
        # On the x-axis
        top_middle = [top_left[0], 0, top_left[2]]
        bottom_middle = [bottom_left[0], 0, bottom_left[2]]

        vertices = np.array([
            top_left, top_middle, top_right, bottom_right, bottom_middle, bottom_left
        ])
        faces = np.array([
            [0, 5, 1],
            [1, 5, 4],
            [1, 4, 3],
            [1, 3, 2]
        ])

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(self._patagium_path), exist_ok=True)
        mesh.export(self._patagium_path)
        # Subdivide the mesh to make it more realistic
        subdivide_update_mesh(self._patagium_path, count=subdivisions_count)

    def _build_soft_tissue(self, corner_points: np.ndarray, subdivisions_count: int) -> None:

        # Create the mesh if it does not already exist (for multiple experiments on the same time)
        # if not os.path.exists(self._patagium_path):
        #    self._create_mesh(corner_points, subdivisions_count)

        # Wait short time to make sure the file is created before it is used
        # otherwise it will error on the HPC
        time.sleep(0.5)

        self._create_mesh(corner_points, subdivisions_count)

        # Because the inertiabox starts at every vertex, so we need to divide by 2 (take the max distance, because
        # otherwise there will be a lot of holes in the whole inertiabox of the patagium)
        inertiabox = get_max_distance_edge(self._patagium_path) / 2

        self._flexcomp_name = f"{self.base_name}_soft_tissue"
        self._flexcomp = self.mjcf_body.add(
            "flexcomp",
            name=self._flexcomp_name,
            type="mesh",
            file=self._patagium_path,
            dim="2",
            mass=0.01,
            inertiabox=inertiabox,
            pos="0 0 0",
            rgba=colors.rgba_green,
        )

        self._flexcomp.add("contact", contype=1, conaffinity=0, selfcollide="none")
        self._flexcomp.add("edge", equality="true")

        # Pin id for a mesh is the index of the point in the mesh
        generated_pin_ranges = generate_pin_ranges_flying_squirrel(self._patagium_path)
        for start, end in generated_pin_ranges:
            if start == end:
                self._flexcomp.add("pin", id=f"{start}")
            else:
                self._flexcomp.add("pin", range=f"{start} {end}")

    def _connect_forelimbs(self):
        length_clavicle = self.morphology_specification.forelimb_specification.clavicle_specification.length.value
        radius_clavicle = self.morphology_specification.forelimb_specification.clavicle_specification.radius.value
        width_trunk = self.morphology_specification.trunk_specification.width.value
        length_humerus = self.morphology_specification.forelimb_specification.humerus_specification.length.value
        radius_humerus = self.morphology_specification.forelimb_specification.humerus_specification.radius.value
        length_radius = self.morphology_specification.forelimb_specification.radius_specification.length.value
        radius_radius = self.morphology_specification.forelimb_specification.radius_specification.radius.value

        x_pos = self.morphology_specification.trunk_specification.length.value / 2 - radius_clavicle / 2

        # Connect the clavicles
        clavicles_min_y_pos = width_trunk / 2
        clavicles_max_y_pos = width_trunk / 2 + length_clavicle + radius_clavicle
        self._connect_clavicles(x_pos=x_pos, min_y_pos=clavicles_min_y_pos, max_y_pos=clavicles_max_y_pos)

        # Connect the humerus
        humerus_min_y_pos = clavicles_max_y_pos
        humerus_max_y_pos = humerus_min_y_pos + length_humerus + radius_humerus
        self._connect(x_pos=x_pos, min_y_pos=humerus_min_y_pos, max_y_pos=humerus_max_y_pos, name_connect="humerus",
                      body1_left=self._left_forelimb._humerus.base_name,
                      body1_right=self._right_forelimb._humerus.base_name)

        # Connect the radius
        radius_min_y_pos = humerus_max_y_pos
        radius_max_y_pos = radius_min_y_pos + length_radius + radius_radius
        self._connect(x_pos=x_pos, min_y_pos=radius_min_y_pos, max_y_pos=radius_max_y_pos, name_connect="radius",
                      body1_left=self._left_forelimb._radius.base_name,
                      body1_right=self._right_forelimb._radius.base_name)

        # Connect the wing tip
        wing_tip_min_y_pos = radius_max_y_pos
        wing_tip_max_y_pos = float('inf')
        self._connect(x_pos=x_pos, min_y_pos=wing_tip_min_y_pos, max_y_pos=wing_tip_max_y_pos, name_connect="wing_tip",
                      body1_left=self._left_forelimb._hand._wing_tip_body_name,
                      body1_right=self._right_forelimb._hand._wing_tip_body_name)

    def _connect_hind_limb(self):
        width_trunk = self.morphology_specification.trunk_specification.width.value
        length_trunk = self.morphology_specification.trunk_specification.length.value
        length_femur = self.morphology_specification.hind_limb_specification.femur_specification.length.value
        radius_femur = self.morphology_specification.hind_limb_specification.femur_specification.radius.value

        x_pos = -length_trunk / 2 + radius_femur / 2

        # Connect the femur
        femur_min_y_pos = width_trunk / 2
        femur_max_y_pos = femur_min_y_pos + length_femur + radius_femur
        self._connect(x_pos=x_pos, min_y_pos=femur_min_y_pos, max_y_pos=femur_max_y_pos, name_connect="femur",
                      body1_left=self._left_hind_limb._femur.base_name,
                      body1_right=self._right_hind_limb._femur.base_name)

        # Connect the tibia
        tibia_min_y_pos = femur_max_y_pos
        tibia_max_y_pos = float('inf')
        self._connect(x_pos=x_pos, min_y_pos=tibia_min_y_pos, max_y_pos=tibia_max_y_pos, name_connect="tibia",
                      body1_left=self._left_hind_limb._tibia.base_name,
                      body1_right=self._right_hind_limb._tibia.base_name)

    def _connect_clavicles(self, x_pos, min_y_pos, max_y_pos):
        connects = (generate_connects_flying_squirrel_in_range(self._patagium_path, x_pos, min_y_pos, max_y_pos) +
                    generate_connects_flying_squirrel_in_range(self._patagium_path, x_pos, -max_y_pos, -min_y_pos))

        # clavicle can be pinned because it does not move in this simulation model
        for index, _ in connects:
            self._flexcomp.add("pin", id=f"{index}")

    def _connect(self, x_pos, min_y_pos, max_y_pos, name_connect, body1_left, body1_right):
        # Left
        connects = generate_connects_flying_squirrel_in_range(self._patagium_path, x_pos, min_y_pos, max_y_pos)
        for index, vertex in connects:
            self.mjcf_model.equality.add(
                "connect",
                name=f"{self.base_name}_{name_connect}_connect_{index}",
                body1=body1_left,
                body2=f"{self._flexcomp_name}_{index}",
                anchor=[vertex[1] - min_y_pos, 0, 0]  # y is x here because the forelimb is rotated
            )

        # Right
        connects = generate_connects_flying_squirrel_in_range(self._patagium_path, x_pos, -max_y_pos, -min_y_pos)
        for index, vertex in connects:
            self.mjcf_model.equality.add(
                "connect",
                name=f"{self.base_name}_{name_connect}_connect_{index}",
                body1=body1_right,
                body2=f"{self._flexcomp_name}_{index}",
                anchor=[abs(vertex[1]) - min_y_pos, 0, 0]  # y is x here because the forelimb is rotated
            )


