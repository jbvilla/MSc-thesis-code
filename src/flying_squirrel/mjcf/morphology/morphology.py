import numpy as np
from moojoco.mjcf.morphology import MJCFMorphology
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import qinverse

from src.flying_squirrel.mjcf.morphology.parts.forelimb import MJCFFlyingSquirrelForelimb
from src.flying_squirrel.mjcf.morphology.parts.head import MJCFFlyingSquirrelHead
from src.flying_squirrel.mjcf.morphology.parts.hind_limb import MJCFFlyingSquirrelHindLimb
from src.flying_squirrel.mjcf.morphology.parts.patagium import MJCFFlyingSquirrelPatagium
from src.flying_squirrel.mjcf.morphology.parts.simplified_patagium import MJCFFlyingSquirrelSimplifiedPatagium
from src.flying_squirrel.mjcf.morphology.parts.tail import MJCFFlyingSquirrelTail
from src.flying_squirrel.mjcf.morphology.parts.trunk import MJCFFlyingSquirrelTrunk
from src.flying_squirrel.mjcf.morphology.specification.default import default_flying_squirrel_specification
from src.flying_squirrel.mjcf.morphology.specification.specification import FlyingSquirrelMorphologySpecification, \
    FlyingSquirrelTrunkSpecification


def _inverse_euler(euler_angles):
    """
    Inverse the euler
    :param euler_angles: euler angles in radians
    :return:
    """
    # Convert to quaternion
    quat = euler2quat(*euler_angles, axes="rxyz")

    # Inverse quaternion
    inv_quat = qinverse(quat)

    # Convert back to euler angles
    inv_euler = quat2euler(inv_quat, axes="rxyz")

    return inv_euler


class MJCFFlyingSquirrelMorphology(MJCFMorphology):
    def __init__(self, specification: FlyingSquirrelMorphologySpecification, *args, **kwargs) -> None:
        super().__init__(specification, name="FlyingSquirrelMorphology", *args, **kwargs)

    @property
    def morphology_specification(self) -> FlyingSquirrelMorphologySpecification:
        specification = super().morphology_specification
        if isinstance(specification, FlyingSquirrelMorphologySpecification):
            return specification
        else:
            raise TypeError("Specification is not of type FlyingSquirrelMorphologySpecification.")

    def _build(self, euler_flying_squirrel=np.array([0, 0, 0]), num_cameras=4, simplified_wings=False, *args, **kwargs) -> None:
        if simplified_wings:
            # if simplified_wings is True the trunk needs to be adjusted
            trunk_specification = self.morphology_specification.trunk_specification
            self.morphology_specification.trunk_specification = FlyingSquirrelTrunkSpecification(
                length=trunk_specification.length.value,
                width=trunk_specification.width.value * 2.1,
                height=trunk_specification.height.value * 2.1,)
        self._configure_compiler()
        self._configure_defaults()
        self._build_trunk()
        # (keep this order because otherwise the already existing parameters will be mixed up)
        if not simplified_wings:
            self._build_forelimbs()
            self._build_hind_limbs()
        self._build_head()
        self._build_tail()
        if not simplified_wings:
            self._build_patagium()
        else:
            self._build_simplified_wings()
        self._configure_camera(euler_flying_squirrel, num_cameras)

    def _configure_compiler(self) -> None:
        self.mjcf_model.compiler.angle = "radian"

    def _configure_defaults(self) -> None:
        # condim 6 for the most realistic simulation when there is geom contact
        self.mjcf_model.default.geom.condim = 6
        # two geom objects will collide if (contype1 & conaffinity2) || (contype2 & conaffinity1) is not 0
        self.mjcf_model.default.geom.contype = 1
        self.mjcf_model.default.geom.conaffinity = 0

    def _build_trunk(self) -> None:
        self._trunk = MJCFFlyingSquirrelTrunk(
            parent=self,
            name="trunk",
            pos=np.zeros(3),
            euler=np.zeros(3),
        )

    def _build_forelimbs(self) -> None:
        angle = np.pi / 2

        trunk_length = self.morphology_specification.trunk_specification.length.value
        trunk_width = self.morphology_specification.trunk_specification.width.value

        # humerus radius for the right position of the forelimb
        humerus_radius = self.morphology_specification.forelimb_specification.humerus_specification.radius.value

        x_pos = trunk_length / 2 - humerus_radius / 2
        y_pos = trunk_width / 2

        self._left_forelimb = MJCFFlyingSquirrelForelimb(
            parent=self._trunk,
            name="left_forelimb",
            pos=[x_pos, y_pos, 0.0],
            euler=[0.0, 0.0, angle],
            in_plane_joint_axis=[0, 0, -1],
            out_of_plane_joint_axis=[0, -1, 0],
        )

        self._right_forelimb = MJCFFlyingSquirrelForelimb(
            parent=self._trunk,
            name="right_forelimb",
            pos=[x_pos, -y_pos, 0.0],
            euler=[0.0, np.pi, -angle],  # so it is mirrored from the left forelimb
            in_plane_joint_axis=[0, 0, -1],
            out_of_plane_joint_axis=[0, 1, 0],
        )

    def _build_hind_limbs(self) -> None:
        angle = np.pi / 2

        trunk_length = self.morphology_specification.trunk_specification.length.value
        trunk_width = self.morphology_specification.trunk_specification.width.value

        # Tibia radius for the right position of the hind limb
        tibia_radius = self.morphology_specification.hind_limb_specification.tibia_specification.radius.value

        x_pos = -trunk_length / 2 + tibia_radius / 2
        y_pos = trunk_width / 2

        self._left_hind_limb = MJCFFlyingSquirrelHindLimb(
            parent=self._trunk,
            name="left_hind_limb",
            pos=[x_pos, y_pos, 0.0],
            euler=[0.0, 0.0, angle],
            in_plane_joint_axis=[0, 0, -1],
            out_of_plane_joint_axis=[0, -1, 0],
        )

        self._right_hind_limb = MJCFFlyingSquirrelHindLimb(
            parent=self._trunk,
            name="right_hind_limb",
            pos=[x_pos, -y_pos, 0.0],
            euler=[0.0, 0.0, -angle],
            in_plane_joint_axis=[0, 0, 1],
            out_of_plane_joint_axis=[0, -1, 0],
        )

    def _build_head(self) -> None:
        trunk_length = self.morphology_specification.trunk_specification.length.value

        MJCFFlyingSquirrelHead(
            parent=self._trunk,
            name="head",
            pos=[trunk_length / 2, 0.0, 0.0],
            euler=[0.0, 0.0, 0.0],
        )

    def _build_tail(self) -> None:
        trunk_length = self.morphology_specification.trunk_specification.length.value

        MJCFFlyingSquirrelTail(
            parent=self._trunk,
            name="tail",
            pos=[-trunk_length / 2, 0.0, 0.0],
            euler=[0.0, 0.0, np.pi],
        )

    def _build_patagium(self) -> None:
        self._patagium = MJCFFlyingSquirrelPatagium(
            parent=self._trunk,
            name="patagium",
            pos=np.zeros(3),
            euler=np.zeros(3),
            left_forelimb=self._left_forelimb,
            right_forelimb=self._right_forelimb,
            left_hind_limb=self._left_hind_limb,
            right_hind_limb=self._right_hind_limb,
            corner_points=self.get_corner_points_wing(),
        )

    def _build_simplified_wings(self) -> None:
        self.simplified_wing_right = MJCFFlyingSquirrelSimplifiedPatagium(
            parent=self._trunk,
            name="simplified_wing_right",
            pos=np.zeros(3),
            euler=np.zeros(3),
            corner_points=self.get_corner_points_wing(),
            in_plane_joint_axis=[0, 0, 1],
            out_of_plane_joint_axis=[-1, 0, 0],
            out_of_plane_pitch_joint_axis=[0, -1, 0],
        )

        self.simplified_wing_left = MJCFFlyingSquirrelSimplifiedPatagium(
            parent=self._trunk,
            name="simplified_wing_left",
            pos=np.zeros(3),
            euler=[0, np.pi, np.pi],
            corner_points=self.get_corner_points_wing(),
            in_plane_joint_axis=[0, 0, 1],
            out_of_plane_joint_axis=[1, 0, 0],
            out_of_plane_pitch_joint_axis=[0, 1, 0],
        )

    def _configure_camera(self, euler_flying_squirrel, num_cameras=4) -> None:

        trunk_camera_body = self._trunk.mjcf_body.add(
            "body",
            name="trunk_camera_body",
            pos=[0.0, 0.0, 0.0],
            # inverse euler to make the camera horizontal with the ground again
            euler=_inverse_euler(euler_flying_squirrel),
        )

        trunk_camera_body.add(
            "camera",
            name="side_camera",
            pos=[0.0, -1.0, 1.4],
            quat=euler2quat(40 / 180 * np.pi, 0, 0),
            mode="track",
        )

        # Close up cameras
        radius = 0.3
        height = 0.3
        tilt_deg = 45
        tilt_rad = np.radians(tilt_deg)

        # Starting position is in front
        theta0 = -np.pi / 2

        # Add cameras around the trunk
        for i in range(num_cameras):
            theta = i * 2 * np.pi / num_cameras + theta0
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = height

            yaw = (theta + np.pi) - (theta0 + np.pi)
            quat = euler2quat(tilt_rad, 0, yaw)

            trunk_camera_body.add(
                "camera",
                name=f"camera_{i}",
                pos=[x, y, z],
                quat=quat,
                mode="track",
            )

    def get_corner_points_wing(self, debug: bool = False) -> np.ndarray:
        """
        Get the corner points of the wing
        :param debug: if True, print the corner points and draw them to the model
        :return: the corner points of the wing starting from the top left and going clockwise
        """
        length_trunk = self.morphology_specification.trunk_specification.length.value
        width_trunk = self.morphology_specification.trunk_specification.width.value
        length_humerus = self.morphology_specification.forelimb_specification.humerus_specification.length.value
        radius_humerus = self.morphology_specification.forelimb_specification.humerus_specification.radius.value
        length_radius = self.morphology_specification.forelimb_specification.radius_specification.length.value
        radius_radius = self.morphology_specification.forelimb_specification.radius_specification.radius.value
        length_clavicle = self.morphology_specification.forelimb_specification.clavicle_specification.length.value
        radius_clavicle = self.morphology_specification.forelimb_specification.clavicle_specification.radius.value
        length_wing_tip = self.morphology_specification.forelimb_specification.hand_specification.wing_tip_length.value

        length_tibia = self.morphology_specification.hind_limb_specification.tibia_specification.length.value
        radius_tibia = self.morphology_specification.hind_limb_specification.tibia_specification.radius.value
        length_femur = self.morphology_specification.hind_limb_specification.femur_specification.length.value
        radius_femur = self.morphology_specification.hind_limb_specification.femur_specification.radius.value

        y_pos = width_trunk / 2 + radius_clavicle + length_clavicle + radius_humerus + length_humerus + radius_radius + length_radius + length_wing_tip
        x_pos = length_trunk / 2 - radius_humerus / 2

        y_pos2 = width_trunk / 2 + radius_tibia + length_tibia + radius_femur / 2 + length_femur
        x_pos2 = -length_trunk / 2 + radius_femur / 2

        if debug:
            size = 0.001
            self.mjcf_body.add(
                "site",
                name="corner_point_1",
                pos=[x_pos, y_pos, 0.0],
                size=[size],
                rgba=[1, 0, 0, 1],
            )
            self.mjcf_body.add(
                "site",
                name="corner_point_2",
                pos=[x_pos, -y_pos, 0.0],
                size=[size],
                rgba=[1, 0, 0, 1],
            )
            self.mjcf_body.add(
                "site",
                name="corner_point_3",
                pos=[x_pos2, y_pos2, 0.0],
                size=[size],
                rgba=[1, 0, 0, 1],
            )
            self.mjcf_body.add(
                "site",
                name="corner_point_4",
                pos=[x_pos2, -y_pos2, 0.0],
                size=[size],
                rgba=[1, 0, 0, 1],
            )
            print("Corner points of the wing:")
            print(f"1: ({x_pos}, {y_pos}, 0.0)")
            print(f"2: ({x_pos}, {-y_pos}, 0.0)")
            print(f"3: ({x_pos2}, {-y_pos2}, 0.0)")
            print(f"4: ({x_pos2}, {y_pos2}, 0.0)")

        return np.array([
            [x_pos, y_pos, 0.0],
            [x_pos, -y_pos, 0.0],
            [x_pos2, -y_pos2, 0.0],
            [x_pos2, y_pos2, 0.0],
        ])


if __name__ == "__main__":
    spec = default_flying_squirrel_specification()
    morphology = MJCFFlyingSquirrelMorphology(specification=spec, simplified_wings=False)
    # morphology.print_corner_points_wing(debug=True)
    morphology.export_to_xml_with_assets("./xml")

