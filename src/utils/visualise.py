"""
Code from SeLab3 for visualizing the MJCF model and creating a video from the images.

With additions for the flying squirrel model.
"""
import io

from moojoco.environment.base import MuJoCoEnvironmentConfiguration
from moojoco.mjcf.component import MJCFRootComponent
import mujoco
import cv2
from typing import List
import numpy as np
import logging


def visualize_mjcf(
        mjcf: MJCFRootComponent, name_image: str = "visualization"
) -> None:
    """
    Visualizes the given MJCF model and saves the image.
    :param mjcf: MJCFRootComponent to visualize
    :param name_image: name of the image file
    """
    model = mujoco.MjModel.from_xml_string(mjcf.get_mjcf_str())
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    image = renderer.render()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Save the image to a file
    cv2.imwrite(name_image + ".png", image)


def post_render(
        render_output: List[np.ndarray],
        environment_configuration: MuJoCoEnvironmentConfiguration
) -> np.ndarray:
    """
    Post-processes the render output to create a single image.
    :param render_output: the output of the render function
    :param environment_configuration: the environment configuration
    :return: np.ndarray containing the post-processed image
    """
    if render_output is None:
        # Temporary workaround until https://github.com/google-deepmind/mujoco/issues/1379 is fixed
        return None

    num_cameras = len(environment_configuration.camera_ids)
    num_envs = len(render_output) // num_cameras

    if num_cameras > 1:
        # Horizontally stack frames of the same environment
        frames_per_env = np.array_split(render_output, num_envs)
        render_output = [np.concatenate(env_frames, axis=1) for env_frames in frames_per_env]

    # Vertically stack frames of different environments
    render_output = np.concatenate(render_output, axis=0)

    return render_output[:, :, ::-1]  # RGB to BGR


def show_video(
        images: List[np.ndarray | None], name_video: str = "visualization", wandb_video: bool = False
) -> None:
    """
    Creates a video from the given images and saves it to a file
    :param images: the images to create the video from
    :param name_video: name of the video file
    :param wandb_video: When the video is for wandb (should be other format)
    """
    # Temporary workaround until https://github.com/google-deepmind/mujoco/issues/1379 is fixed
    filtered_images = [image for image in images if image is not None]
    num_nones = len(images) - len(filtered_images)
    if num_nones > 0:
        logging.warning(
            f"env.render produced {num_nones} None's. Resulting video might be a bit choppy (consquence of https://github.com/google-deepmind/mujoco/issues/1379)."
        )
    images_array = np.stack(filtered_images)

    height, width, _ = images_array[0].shape
    if wandb_video:
        fourcc = cv2.VideoWriter.fourcc(*'vp90')
        out = cv2.VideoWriter(name_video + ".webm", fourcc, 50.0, (width, height))
    else:
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(name_video + ".mp4", fourcc, 50.0, (width, height))

    for img in images_array:
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    out.release()


def get_video_buffer_webm(result_name):
    """
    Get the video buffer from the video file.
    :param result_name: the name of the video file
    :return: the video buffer
    """
    with open(result_name + ".webm", "rb") as f:
        return io.BytesIO(f.read())


def render_environment(env, env_config, image: bool = False, write_xml: bool = False, actions: np.ndarray = None, initial_vel: np.ndarray = [0, 0, 0], name_video: str = "visualization", wandb_video: bool = False) -> None:
    """
    Renders the environment and saves the image or creates a video.
    :param name_video: name of the video file
    :param actions: the actions to take each step
    :param initial_vel: the initial velocity of the squirrel
    :param env: the environment to render
    :param env_config: the environment configuration
    :param image: whether to save the image
    :param write_xml: whether to write the xml file
    :param name_video: name of the video file
    :param wandb_video: When the video is for wandb (should be other format)
    :return: None
    """
    rng = np.random.RandomState(0)
    state = env.reset(rng=rng)

    if write_xml:
        env.write_to_xml()
    if image:
        frame = env.render(state=state)
        image = post_render(render_output=frame, environment_configuration=env_config)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("environment_arena.png", image)
    else:
        frames = []
        env_state = env.reset(rng=rng)
        env_state.mj_data.qvel[0] = initial_vel[0]
        env_state.mj_data.qvel[1] = initial_vel[1]
        env_state.mj_data.qvel[2] = initial_vel[2]

        if actions is not None:
            # Found by looking at the joint ranges that are not [0-0]
            start_index_joint = 7
            for i in range(start_index_joint, np.shape(actions)[1]):
                env_state.mj_data.qpos[i] = actions[0][i]

        # Initial frame
        frame = post_render(env.render(state=env_state),
                            environment_configuration=env.environment_configuration)
        frames.append(frame)

        i = 0
        while not (env_state.terminated | env_state.truncated):
            if actions is not None:
                env_state = env.step(state=env_state, action=actions[i])
            else:
                env_state = env.step(state=env_state, action=False)
            frame = post_render(env.render(state=env_state),
                                environment_configuration=env.environment_configuration)
            frames.append(frame)
            i += 1

        show_video(images=frames, name_video=name_video, wandb_video=wandb_video)

