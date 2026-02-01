"""
Collect Demonstration with Language Instructions

Keyboard Controls:
- WASD: xy plane movement
- RF: z-axis up/down movement
- QE: tilt
- Arrow keys: rotation
- Space: toggle gripper state
- Z: reset environment (discard current episode data)
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import random
import numpy as np
import shutil
from PIL import Image

from mujoco_env.y_env2 import SimpleEnv2
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ========================
# Configuration Parameters
# ========================
# Random seed setting
SEED = 0      # Fixed seed: same object positions each time
# SEED = None  # None: randomize object positions

# Dataset configuration
REPO_NAME = 'omy_pnp_language'
NUM_DEMO = 20
ROOT = "./demo_data_language"

# Environment configuration
XML_PATH = './asset/example_scene_y2.xml'
CONTROL_HZ = 20
IMAGE_SIZE = (256, 256)


def create_dataset(root, repo_name):
    """Create or load dataset"""
    create_new = True

    if os.path.exists(root):
        print(f"Directory {root} already exists")
        ans = input("Delete it? (y/n) ")
        if ans.lower() == 'y':
            shutil.rmtree(root)
        else:
            create_new = False

    if create_new:
        print("Creating new dataset...")
        dataset = LeRobotDataset.create(
            repo_id=repo_name,
            root=root,
            robot_type="omy",
            fps=CONTROL_HZ,
            features={
                "observation.image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.wrist_image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["action"],
                },
                "obj_init": {
                    "dtype": "float32",
                    "shape": (9,),  # Initial positions of 3 objects
                    "names": ["obj_init"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )
    else:
        print("Loading existing dataset...")
        dataset = LeRobotDataset(repo_name, root=root)

    return dataset


def preprocess_image(image_array, size=IMAGE_SIZE):
    """Preprocess image"""
    image = Image.fromarray(image_array)
    image = image.resize(size)
    return np.array(image)


def collect_demonstrations(env, dataset, num_demo):
    """Collect demonstration data"""
    action = np.zeros(7)
    episode_id = 0
    record_flag = False

    print(f"Starting collection, target: {num_demo}")
    print("Move robot to start recording, press Z to reset current episode")

    while env.env.is_viewer_alive() and episode_id < num_demo:
        env.step_env()

        if env.env.loop_every(HZ=CONTROL_HZ):
            # Check if successful
            done = env.check_success()
            if done:
                dataset.save_episode()
                env.reset()
                episode_id += 1
                record_flag = False
                print(f"Episode {episode_id}/{num_demo} completed")
                continue

            # Teleoperate and get action
            action, reset = env.teleop_robot()

            # Detect if movement started
            if not record_flag and np.sum(np.abs(action)) > 0:
                record_flag = True
                print("Start recording...")

            # Reset environment
            if reset:
                env.reset()
                dataset.clear_episode_buffer()
                record_flag = False
                print("Environment reset")
                continue

            # Get images
            agent_image, wrist_image = env.grab_image()
            agent_image = preprocess_image(agent_image)
            wrist_image = preprocess_image(wrist_image)

            # Execute action
            joint_q = env.step(action)
            action = env.q[:7].astype(np.float32)

            # Record data
            if record_flag:
                dataset.add_frame(
                    {
                        "observation.image": agent_image,
                        "observation.wrist_image": wrist_image,
                        "observation.state": joint_q[:6],
                        "action": action,
                        "obj_init": env.obj_init_pose,
                    },
                    task=env.instruction  # Language instruction
                )

            env.render(teleop=True, idx=episode_id)

    return episode_id


def cleanup_images(dataset):
    """Clean up temporary images folder"""
    images_path = dataset.root / 'images'
    if images_path.exists():
        shutil.rmtree(images_path)
        print("Temporary images folder cleaned up")


def main():
    """Main function"""
    print("=" * 50)
    print("Data Collection with Language Instructions")
    print("=" * 50)

    # Create environment
    print(f"\nLoading environment: {XML_PATH}")
    env = SimpleEnv2(XML_PATH, seed=SEED, state_type='joint_angle')

    # Create dataset
    dataset = create_dataset(ROOT, REPO_NAME)

    # Collect data
    print("\n" + "=" * 50)
    collected = collect_demonstrations(env, dataset, NUM_DEMO)
    print("=" * 50)
    print(f"Collection completed! Total episodes: {collected}")

    # Close environment
    env.env.close_viewer()

    # Clean up temporary files
    try:
        cleanup_images(dataset)
    except FileNotFoundError:
        pass

    print("Done!")


if __name__ == "__main__":
    main()