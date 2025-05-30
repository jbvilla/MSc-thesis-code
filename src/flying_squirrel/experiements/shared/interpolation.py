import numpy as np
from scipy.interpolate import CubicSpline

from src.flying_squirrel.experiements.shared.flight_path import FlightPathWithPhases


def get_time_points(number_steps, number_control_points_spline):
    """
    Gives specific positions for each control point between the number of steps
    :param number_steps: number of steps
    :param number_control_points_spline: number of control points
    :return: time points
    """
    return np.linspace(0, number_steps - 1, number_control_points_spline)


def prepare_control_points(params, number_of_joints, number_control_points_spline):
    """
    Prepare the control points for the interpolation so that every row is a joint with size number_control_points_spline
    + 0 at the start for a smooth visual start
    :param params:
    :param number_of_joints:
    :param number_control_points_spline:
    :return:
    """
    # reshape so every row is a joint with size number_control_points_spline
    params = np.array(params).reshape(number_of_joints, number_control_points_spline)
    # Add control point at the start that is 0 for a smooth visual start
    return np.insert(params, 0, 0, axis=1)


def interpolate_params(
        params: np.ndarray,
        number_steps: int,
        number_of_joints: int,
        number_control_points_spline: int,
        simplified: bool,
        flight_path_with_phases: FlightPathWithPhases = None,
) -> np.ndarray:
    """
    Interpolates the points between the control points for each joint
    :param params: Control points for each joint
    :param number_steps: Number of steps to interpolate
    :param number_of_joints: Number of joints
    :param number_control_points_spline: Number of control points
    :param simplified: Whether the number of joints is simplified (so left and right limbs are mirrored)
    :param flight_path_with_phases: Straight flight object for flight path experiment
    :return: Total interpolated points (Every row is a time step with the values for each joint)
    """
    # reshape so every row is a joint with size number_control_points_spline
    params = prepare_control_points(params, number_of_joints, number_control_points_spline)
    number_control_points_spline += 1

    if simplified:
        # Simplified number of joints so only the half of the libms are generated so mirror them
        number_of_joints = 32
        new_params = [params[:6, :], params[:6, :], params[6:12, :], params[6:12, :], params[12:, :]]
        params = np.vstack(new_params)

    # Gives specific positions for each control point between the number of steps
    if flight_path_with_phases is None:
        time_points = get_time_points(number_steps, number_control_points_spline)
    else:
        # For the flight path experiment, we need to use the control points of the flight path
        # Add 0 at the start for a smooth visual start
        params = np.delete(params, 0, axis=1)

        time_points = flight_path_with_phases.get_control_point_location_indexes()

        interpolated_params = np.zeros((number_of_joints, number_steps))

        range_1 = time_points[1] - time_points[0] + 1
        range_2 = time_points[3] - time_points[2] + 1

        for i in range(number_of_joints):

            transition_1 = np.linspace(0, params[i, 0], range_1)
            transition_2 = np.linspace(params[i, 0], params[i, 1], range_2)

            interpolated_params[i, time_points[0]:time_points[1] + 1] = transition_1
            for j in range(time_points[1]+1, time_points[2]+1):
                interpolated_params[i, j] = params[i, 0]
            interpolated_params[i, time_points[2]:time_points[3] + 1] = transition_2
            for j in range(time_points[3]+1, number_steps):
                interpolated_params[i, j] = params[i, 1]
        return interpolated_params.T

    # For each joint, initialize the values for each time step
    interpolated_params = np.zeros((number_of_joints, number_steps))

    for i in range(number_of_joints):
        # Interpolate the control points
        interp_func = CubicSpline(time_points, params[i], bc_type='natural')
        interpolated_params[i] = interp_func(np.arange(number_steps))

    # Transpose so every row is a time step with the values for each joint
    return interpolated_params.T
