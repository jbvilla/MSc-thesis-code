from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns


class FlightPathModes(Enum):
    NORMAL = 0
    LEFT_TURN = 1
    RIGHT_TURN = 2


class FlightPathWithPhases:
    FREEFALL_PERCENTAGE = 0.25
    STABLE_GLIDE_PERCENTAGE = 0.9

    def __init__(
            self,
            start_pos: np.ndarray,
            freefall_angle: float,
            stable_glide_angle_alpha: float,
            landing_angle: float,
            simulation_time: int,
            number_of_steps: int,
            alpha: int = 10

    ):
        """
        Initialize the flight path parameters.
        :param start_pos: Start position of the flying squirrel in 3D space.
        :param simulation_time: Total simulation time in seconds.
        :param freefall_angle: Angle of freefall in degrees.
        :param stable_glide_angle_alpha: Angle of stable glide in degrees.
        :param landing_angle: Angle of landing in degrees.
        :param alpha: Number of steps to look ahead and behind for control points on a transition phase.

        """
        self.start_pos = start_pos
        self.freefall_angle = np.deg2rad(freefall_angle)
        self.stable_glide_angle_alpha = np.deg2rad(stable_glide_angle_alpha)
        self.landing_angle = np.deg2rad(landing_angle)
        self.freefall_phase = simulation_time * self.FREEFALL_PERCENTAGE
        self.stable_glide_phase = simulation_time * self.STABLE_GLIDE_PERCENTAGE
        self.simulation_time = simulation_time
        self.number_of_steps = number_of_steps
        self.alpha = alpha

        self.directions = self._get_directions()
        self.control_points_indexes, self.control_points_times = self.get_control_point_location_indexes(True)

    def _get_directions(self):
        """
        Get the direction vectors for freefall, stable glide, and landing phases.
        :return: Array of direction vectors.
        """
        freefall_direction_vec = np.array([
            np.cos(self.freefall_angle),
            0,
            np.sin(self.freefall_angle)
        ], dtype=float)

        stable_gliding_direction_vec = np.array([
            np.cos(self.stable_glide_angle_alpha),
            0,
            np.sin(self.stable_glide_angle_alpha)
        ], dtype=float)

        landing_direction_vec = np.array([
            np.cos(self.landing_angle),
            0,
            np.sin(self.landing_angle)
        ], dtype=float)

        return np.array([
            freefall_direction_vec / np.linalg.norm(freefall_direction_vec),
            stable_gliding_direction_vec / np.linalg.norm(stable_gliding_direction_vec),
            landing_direction_vec / np.linalg.norm(landing_direction_vec)
        ], dtype=float)

    def get_direction(self, time: float):
        """
        Get the direction vector based on the time in the simulation.
        :param time: Time in seconds.
        :return: Direction vector.
        """
        if time <= self.freefall_phase:
            return self.directions[0]
        elif time <= self.stable_glide_phase:
            return self.directions[1]
        else:
            return self.directions[2]

    def get_phase(self, time: float):
        """
        Get the phase of the flight based on the time in the simulation.
        :param time: Time in seconds.
        :return: Phase of the flight.
        """
        if time <= self.freefall_phase:
            return 0
        elif time <= self.stable_glide_phase:
            return 1
        else:
            return 2

    def get_control_point_location_indexes(self, return_times=False):
        """
        Get the control point location indexes based on the flight path.
        :param return_times: Whether to return the times of the control points.
        :return: Control point location indexes and optionally the times.
        """
        dt = self.simulation_time / self.number_of_steps
        transition_indexes = []

        index = 0
        time = 0.0
        prev_phase = self.get_phase(time)
        # Obtain the indexes of the transition points
        while time < self.simulation_time:
            phase = self.get_phase(time)
            if phase != prev_phase:
                transition_indexes.append((index - 1, time - dt))
            time += dt
            prev_phase = phase
            index += 1

        transition_indexes = np.array(transition_indexes, dtype=object)

        # Add the control points for the transition phases
        indexes = np.array([idx for idx, _ in transition_indexes])
        transition_control_points = np.column_stack((indexes - self.alpha, indexes + self.alpha)).flatten()

        times = np.array([time for _, time in transition_indexes])
        transition_control_points_times = np.column_stack((times - dt * self.alpha, times + dt * self.alpha)).flatten()

        control_point_indexes = np.array([
            *transition_control_points,
            # self.number_of_steps - 1
        ], dtype=int)

        if return_times:
            return control_point_indexes, [
                *transition_control_points_times,
                self.simulation_time - dt
            ]

        return control_point_indexes

    def plot_path(self, without_real_time=False):
        """
        Plot the path in the xz-plane.
        """
        control_points_indexes, control_points_times = self.get_control_point_location_indexes(True)
        control_points_indexes = control_points_indexes

        pos = self.start_pos.astype(float).copy()
        positions = [pos.copy()]
        times = [0]
        dt = self.simulation_time / self.number_of_steps

        # Start position is already added
        time = dt
        while time < self.simulation_time:
            direction = self.get_direction(time)
            pos += direction * dt
            positions.append(pos.copy())
            time += dt
            times.append(time)

        positions = np.array(positions)

        # Colors
        sns.set(style="white")
        line_color = "lightgray"
        transition_phase_color = "#0072B2"
        control_point_color = "red"
        start_position_color = "#005824"
        end_position_color = "#B22222"
        dashed_line_color = "#8DA0CB"

        plt.figure(figsize=(8, 6))
        plt.plot(positions[:, 0], positions[:, 2], color=line_color, linewidth=1.8)

        # limits
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
        arrow_y = z_min - 0.05 * (z_max - z_min)

        # Vertical lines control points
        for idx, pos in enumerate(control_points_indexes):
            plt.plot([positions[pos, 0], positions[pos, 0]], [arrow_y, positions[pos, 2]],
                     color=dashed_line_color, linestyle='--', linewidth=0.8, alpha=0.6)
            y_offset = idx % 2 * -0.35
            if without_real_time:
                plt.text(positions[pos, 0], arrow_y - 0.1 + y_offset,
                         fr'$t_{{{idx + 1}}}$', ha='center', va='top',
                         fontsize=12, color=dashed_line_color)
            else:
                plt.text(positions[pos, 0], arrow_y - 0.1 + y_offset,
                         f'{control_points_times[idx]:.2f} s', ha='center', va='top',
                         fontsize=12, color=dashed_line_color)

        # Transition phases
        i, j = control_points_indexes[0], control_points_indexes[1]
        plt.plot(positions[i:j + 1, 0], positions[i:j + 1, 2], color=transition_phase_color, linewidth=2, label='Overgangsfasen')
        i, j = control_points_indexes[2], control_points_indexes[3]
        plt.plot(positions[i:j + 1, 0], positions[i:j + 1, 2], color=transition_phase_color, linewidth=2)

        # Control points
        for i, idx in enumerate(control_points_indexes):
            plt.plot(positions[idx, 0], positions[idx, 2], 'ro', markersize=3, color=control_point_color,
                     label='Controlepunten' if i == 0 else "")

        # Start and end positions
        if without_real_time:
            start_label = fr'Startpositie ($t_{{start}}$)'
            end_label = fr'Eindpositie ($t_{{einde}}$)'
        else:
            start_label = f'Startpositie ({0} s)'
            end_label = f'Eindpositie ({self.simulation_time} s)'

        plt.plot(positions[0, 0], positions[0, 2], 's', color=start_position_color, markersize=3, label=start_label)
        plt.plot(positions[-1, 0], positions[-1, 2], 's', color=end_position_color, markersize=3, label=end_label)

        # Arrow
        x_pad = 0.2 * (x_max - x_min)
        z_pad = 0.2 * (z_max - z_min)
        plt.xlim(x_min - x_pad, x_max + x_pad)
        plt.ylim(z_min - z_pad, z_max + z_pad)

        plt.annotate('', xy=(x_max + x_pad * 0.8, arrow_y),
                     xytext=(x_min - x_pad * 0.8, arrow_y),
                     arrowprops=dict(arrowstyle="->", linewidth=1.4, color='black'))
        plt.text(x_max + x_pad * 0.9, arrow_y, 'Tijd (s)', ha='left', va='center',
                 fontsize=12, color='black')

        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_aspect('equal', adjustable='box')

        plt.title('Zweeffasen van de Vliegende Eekhoorn', fontsize=14)

        plt.legend(fontsize=11, loc='best', frameon=True)
        plt.tight_layout()

        plt.savefig(f'flight_path_{self.__class__.__name__}{"_real_values" if not without_real_time else ""}.png', dpi=300, bbox_inches='tight')
        plt.close()


class FlightPath:
    def __init__(
            self,
            mode: FlightPathModes,
            start_pos: np.ndarray,
            end_pos: np.ndarray,
            stable_glide_angle_alpha: float,
            spacing: float = 0.25,
            turn_distance_from_center: float = 2.0
    ):
        self.mode = mode
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.stable_glide_angle_alpha = stable_glide_angle_alpha
        self.turn_distance_from_center = turn_distance_from_center

        self.path = self._get_path()
        if self.mode == FlightPathModes.NORMAL:
            self.interpolated_path = self.get_interpolate_path(spacing, 'linear')
        else:
            self.interpolated_path = self.get_interpolate_path(spacing, 'quadratic')

    def _get_path(self):

        if self.mode == FlightPathModes.NORMAL:
            end_pos_free_fall_phase = np.array([
                (self.end_pos[0] - self.start_pos[0]) * 0.05,
                self.start_pos[1],
                self.start_pos[2] - (self.start_pos[2] - self.end_pos[2]) * 2 / 3
            ])

            end_pos_transition_phase = np.array([
                (self.end_pos[0] - self.start_pos[0]) * 0.15,
                self.start_pos[1],
                self.start_pos[2] - (self.start_pos[2] - self.end_pos[2]) * (2 / 3 + 0.05)
            ])

            end_pos_stable_glide_phase = np.array([
                (self.end_pos[0] - self.start_pos[0]) * 0.95,
                self.start_pos[1],
                0
            ])
            end_pos_stable_glide_phase[2] = end_pos_transition_phase[2] - np.tan(
                np.radians(self.stable_glide_angle_alpha)) * (
                                                    end_pos_stable_glide_phase[0] - end_pos_transition_phase[0])

            return np.array([
                self.start_pos,
                end_pos_free_fall_phase,
                end_pos_transition_phase,
                end_pos_stable_glide_phase,
                self.end_pos
            ])

        else:
            middle_pos = (self.start_pos + self.end_pos) / 2
            direction_vec = self.end_pos - self.start_pos
            orthogonal_vec = np.array([-direction_vec[1], direction_vec[0], 0], dtype=float)
            # normalize
            orthogonal_vec /= np.linalg.norm(orthogonal_vec)

            if self.mode == FlightPathModes.LEFT_TURN:
                # counter-clockwise rotation
                center_pos = middle_pos - self.turn_distance_from_center * orthogonal_vec
            else:
                # RIGHT_TURN
                # clockwise rotation
                center_pos = middle_pos + self.turn_distance_from_center * orthogonal_vec

            return np.array([
                self.start_pos,
                center_pos,
                self.end_pos
            ])

    def get_interpolate_path(self, spacing, type_of_interpolation='linear'):
        # Calculate cumulative distances between points
        distances = np.sqrt(np.sum(np.diff(self.path, axis=0) ** 2, axis=1))
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

        # Interpolate for each axis
        interpolated_points = []
        for i in range(self.path.shape[1]):
            interp_func = interp1d(cumulative_distances, self.path[:, i], kind=type_of_interpolation)
            new_distances = np.arange(0, cumulative_distances[-1], spacing)
            interpolated_points.append(interp_func(new_distances))

        return np.stack(interpolated_points, axis=1)

    def plot_path_xz(self):
        x_values = self.path[:, 0]
        z_values = self.path[:, 2]

        # Maak de plot
        plt.figure(figsize=(8, 6))
        plt.plot(x_values, z_values, marker='o', linestyle='-', color='b', label='hoogte over x afstand')
        plt.xlabel('Afstand x (m)')
        plt.ylabel('Hoogte (m)')
        plt.title('Zweeftraject van de Vliegende Eekhoorn')
        plt.legend()
        plt.savefig('flight_path_plot.png')

    def plot_path_xy(self, spacing=None):
        if spacing is None:
            x_values = self.path[:, 0]
            y_values = self.path[:, 1]
        else:
            interpolated_path = self.get_interpolate_path(spacing, 'quadratic')
            x_values = interpolated_path[:, 0]
            y_values = interpolated_path[:, 1]

        # Maak de plot
        plt.figure(figsize=(8, 6))
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label='y afstand over x afstand')
        plt.xlabel('Afstand x (m)')
        plt.ylabel('Afstand y (m)')
        plt.title('Zweeftraject van de Vliegende Eekhoorn')
        plt.legend()
        plt.axis('equal')
        plt.savefig('flight_path_plot_xy.png')


if __name__ == "__main__":

    straight_path = True

    if straight_path:
        start_pos = np.array([0, 0, 20])
        freefall_angle = -90
        stable_glide_angle_alpha = -20
        landing_angle = 5
        simulation_time = 10
        number_of_steps = 500
        alpha = 15

        flight_path = FlightPathWithPhases(start_pos, freefall_angle, stable_glide_angle_alpha, landing_angle,
                                           simulation_time, number_of_steps, alpha)

        flight_path.plot_path(False)

    else:
        start_pos = np.array([0, 0, 4])
        target_pos = np.array([4, 0, 0.5])
        stable_glide_angle_alpha = 20

        flight_path = FlightPath(FlightPathModes.NORMAL, start_pos, target_pos, stable_glide_angle_alpha)
        flight_path.plot_path_xz()
        print("Linear interpolation:")
        print(flight_path.get_interpolate_path(0.25))

        target_pos = np.array([7, 6, 0.5])
        turn_distance = 2

        flight_path = FlightPath(FlightPathModes.LEFT_TURN, start_pos, target_pos, stable_glide_angle_alpha, turn_distance_from_center=turn_distance)
        flight_path.plot_path_xy(0.25)
        print("Quadratic interpolation:")
        print(flight_path.get_interpolate_path(0.25, 'quadratic'))
