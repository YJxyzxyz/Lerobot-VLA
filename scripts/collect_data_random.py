"""
Robot Teleoperation Data Collection Script (Random Reset Version)

Collect demonstration data by teleoperating a robot via keyboard for imitation learning.
This file is self-contained and does not import parameters/helpers from collect_data.py.

Keyboard Controls:
    W/A/S/D     - Move in XY plane
    R/F         - Move up/down (Z-axis)
    Q/E         - Tilt left/right
    Arrow Keys  - Rotation control
    Spacebar    - Toggle gripper open/close
    Z           - Reset environment and discard current episode
"""

import sys
import os
import shutil
import random
import numpy as np
from PIL import Image

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mujoco_env.y_env import SimpleEnv
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ==================== Configuration ====================
# Keep RESET_SEED=None so every reset gets randomized object placement.
RESET_SEED = None
BASE_SEED = None
REPO_NAME = "omy_pnp"
NUM_DEMO = 1
ROOT = "./demo_data_random"
TASK_NAME = "Put mug cup on the plate"
XML_PATH = "./asset/example_scene_y.xml"

# ==================== Dataset Feature Definition ====================
DATASET_FEATURES = {
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
        "shape": (6,),
        "names": ["obj_init"],
    },
}


def create_or_load_dataset(root: str, repo_name: str, features: dict) -> LeRobotDataset:
    """Create a new dataset or load an existing one."""
    if os.path.exists(root):
        print(f"Directory {root} already exists.")
        ans = input("Do you want to delete and recreate? (y/n) ").strip().lower()
        if ans == "y":
            shutil.rmtree(root)
            print(f"Deleted {root}")
        else:
            print("Loading existing dataset...")
            return LeRobotDataset(repo_name, root=root)

    print("Creating new dataset...")
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        root=root,
        robot_type="omy",
        fps=20,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )
    return dataset


def collect_demonstrations(
    env: SimpleEnv,
    dataset: LeRobotDataset,
    num_demos: int,
    reset_seed,
    task_name: str,
):
    """Main data collection loop."""
    episode_id = 0
    record_flag = False

    print("\n" + "=" * 50)
    print("Starting Data Collection")
    print(f"Target demonstrations: {num_demos}")
    print("=" * 50 + "\n")

    while env.env.is_viewer_alive() and episode_id < num_demos:
        env.step_env()

        if env.env.loop_every(HZ=20):
            done = env.check_success()
            if done:
                print(f"\n[OK] Episode {episode_id + 1}/{num_demos} completed!")
                dataset.save_episode()
                env.reset(seed=reset_seed)
                episode_id += 1
                record_flag = False
                if episode_id < num_demos:
                    print(f"Ready to collect episode {episode_id + 1}...\n")
                continue

            action, reset = env.teleop_robot()

            if not record_flag and sum(action) != 0:
                record_flag = True
                print("Recording started...")

            if reset:
                print("Resetting environment, clearing current episode data...")
                env.reset(seed=reset_seed)
                dataset.clear_episode_buffer()
                record_flag = False
                continue

            ee_pose = env.get_ee_pose()
            agent_image, wrist_image = env.grab_image()

            agent_image = np.array(Image.fromarray(agent_image).resize((256, 256)))
            wrist_image = np.array(Image.fromarray(wrist_image).resize((256, 256)))

            joint_q = env.step(action)

            if record_flag:
                dataset.add_frame(
                    {
                        "observation.image": agent_image,
                        "observation.wrist_image": wrist_image,
                        "observation.state": ee_pose,
                        "action": joint_q,
                        "obj_init": env.obj_init_pose,
                    },
                    task=task_name,
                )

            env.render(teleop=True)

    print("\n" + "=" * 50)
    print(f"Data collection finished! Collected {episode_id} demonstrations.")
    print("=" * 50)


def cleanup(env: SimpleEnv, dataset: LeRobotDataset):
    """Clean up resources after data collection."""
    env.env.close_viewer()

    images_path = dataset.root / "images"
    if images_path.exists():
        shutil.rmtree(images_path)
        print("Cleaned up temporary image files.")


def main():
    print("\n" + "=" * 50)
    print("Robot Teleoperation Data Collection System (Random Reset)")
    print("=" * 50)
    print("\nKeyboard Controls:")
    print("  W/A/S/D    - Move in XY plane")
    print("  R/F        - Move up/down (Z-axis)")
    print("  Q/E        - Tilt left/right")
    print("  Arrows     - Rotation control")
    print("  Spacebar   - Toggle gripper")
    print("  Z          - Reset environment")
    print("\nTask: Pick up the mug and place it on the plate")
    print("Success: Mug on plate + gripper open + end-effector above mug")
    print("\nRandom reset mode: enabled (seed=None on every reset)\n")

    if BASE_SEED is not None:
        np.random.seed(BASE_SEED)
        random.seed(BASE_SEED)
        print(f"Using base random seed: {BASE_SEED}")

    print("Initializing simulation environment...")
    env = SimpleEnv(XML_PATH, seed=RESET_SEED, state_type="joint_angle")

    dataset = create_or_load_dataset(ROOT, REPO_NAME, DATASET_FEATURES)

    try:
        collect_demonstrations(env, dataset, NUM_DEMO, RESET_SEED, TASK_NAME)
    except KeyboardInterrupt:
        print("\n\nData collection interrupted by user.")
    finally:
        cleanup(env, dataset)
        print("Program terminated.")


if __name__ == "__main__":
    main()
