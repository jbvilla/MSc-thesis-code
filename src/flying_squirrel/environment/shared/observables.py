import mujoco
import numpy as np
from moojoco.environment.mjc_env import MJCObservable
from transforms3d.euler import quat2euler


def get_base_flying_squirrel_observables(
        mj_model: mujoco.MjModel, backend: str
):
    if backend == "mjc":
        observable_class = MJCObservable
        bnp = np
        get_data = lambda state: state.mj_data
    else:
        raise ValueError(f"Backend {backend} not supported")

    # sensors = [mj_model.sensor(i) for i in range(mj_model.nsensor)]

    trunk_framepos_sensor = [
        mj_model.sensor(i) for i in range(mj_model.nsensor)
        if mj_model.sensor(i).type[0] == mujoco.mjtSensor.mjSENS_FRAMEPOS
        and "trunk" in mj_model.sensor(i).name
    ][0]
    trunk_position_observable = observable_class(
        name="trunk_position",
        low=-bnp.inf * bnp.ones(3),
        high=bnp.inf * bnp.ones(3),
        retriever=lambda state: get_data(state).sensordata[
            trunk_framepos_sensor.adr[0]: trunk_framepos_sensor.adr[0] + trunk_framepos_sensor.dim[0]
        ],
    )

    # Orientation [roll, pitch, yaw]
    # roll is around x-axis, pitch is around y-axis, yaw is around z-axis
    trunk_framequat_sensor = [
        mj_model.sensor(i) for i in range(mj_model.nsensor)
        if mj_model.sensor(i).type[0] == mujoco.mjtSensor.mjSENS_FRAMEQUAT
        and "trunk" in mj_model.sensor(i).name
    ][0]
    trunk_orientation_observable = observable_class(
        name="trunk_orientation",
        low=-bnp.pi * bnp.ones(3),
        high=bnp.pi * bnp.ones(3),
        retriever=lambda state: np.array(quat2euler(
            get_data(state).sensordata[
                trunk_framequat_sensor.adr[0]: trunk_framequat_sensor.adr[0] + trunk_framequat_sensor.dim[0]
            ],
            # Set the axes convention to be the same as in mujoco (Intrinsiek)
            axes='rxyz',
        )),
    )

    # Angle velocity
    trunk_frameangvel_sensor = [
        mj_model.sensor(i) for i in range(mj_model.nsensor)
        if mj_model.sensor(i).type[0] == mujoco.mjtSensor.mjSENS_FRAMEANGVEL
        and "trunk" in mj_model.sensor(i).name
    ][0]
    trunk_angular_velocity_observable = observable_class(
        name="trunk_angular_velocity",
        low=-bnp.inf * bnp.ones(3),
        high=bnp.inf * bnp.ones(3),
        retriever=lambda state: get_data(state).sensordata[
            trunk_frameangvel_sensor.adr[0]: trunk_frameangvel_sensor.adr[0] + trunk_frameangvel_sensor.dim[0]
        ],
    )

    # Linear velocity
    trunk_framevel_sensor = [
        mj_model.sensor(i) for i in range(mj_model.nsensor)
        if mj_model.sensor(i).type[0] == mujoco.mjtSensor.mjSENS_FRAMELINVEL
        and "trunk" in mj_model.sensor(i).name
    ][0]
    trunk_linear_velocity_observable = observable_class(
        name="trunk_linear_velocity",
        low=-bnp.inf * bnp.ones(3),
        high=bnp.inf * bnp.ones(3),
        retriever=lambda state: get_data(state).sensordata[
            trunk_framevel_sensor.adr[0]: trunk_framevel_sensor.adr[0] + trunk_framevel_sensor.dim[0]
        ],
    )

    return [
        trunk_position_observable,
        trunk_orientation_observable,
        trunk_angular_velocity_observable,
        trunk_linear_velocity_observable,
    ]
