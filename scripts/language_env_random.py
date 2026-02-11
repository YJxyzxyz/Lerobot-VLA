"""
Collect Language-Conditioned Demonstrations with Randomized Object Positions.

This script is self-contained and intentionally keeps reset seed as None,
so object initialization changes across episodes.

Keyboard Controls:
    W/A/S/D     - Move in XY plane
    R/F         - Move up/down (Z-axis)
    Q/E         - Tilt left/right
    Arrow Keys  - Rotation control
    Spacebar    - Toggle gripper open/close
    Z           - Reset environment and discard current episode

Usage:
    python scripts/language_env_random.py
"""

import sys
import os
import argparse
import random
import shutil

import numpy as np
from PIL import Image

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mujoco_env.y_env2 import SimpleEnv2
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


# Keep this as None to randomize object placement every reset.
RESET_SEED = None
# Optional: set to an int for reproducible random sequence across runs.
BASE_SEED = None
# Explicit scene randomization after each reset (wider than y_env2 defaults).
FORCE_SCENE_RANDOMIZATION = True
INSTRUCTION_TEMPLATES = [
    "Place the {color} mug on the plate.",
    "Put the {color} mug onto the plate.",
    "Move the {color} mug to the plate.",
    "Set the {color} mug down on the plate.",
    "Pick up the {color} mug and put it on the plate.",
    "Transfer the {color} mug onto the plate.",
    "Lift the {color} mug and place it on the plate.",
    "Take the {color} mug and set it on the plate.",
    "Position the {color} mug on the plate.",
    "Relocate the {color} mug to the plate.",
]

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
        "shape": (9,),
        "names": ["obj_init"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect language demonstrations with random object positions"
    )
    parser.add_argument("--repo_name", type=str, default="omy_pnp_language")
    parser.add_argument("--root", type=str, default="./demo_data_language_random")
    parser.add_argument("--num_demo", type=int, default=20)
    parser.add_argument("--xml_path", type=str, default="./asset/example_scene_y2.xml")
    parser.add_argument("--control_hz", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=256)
    return parser.parse_args()


def create_or_load_dataset(root: str, repo_name: str, control_hz: int) -> LeRobotDataset:
    """Create new dataset or load existing one."""
    if os.path.exists(root):
        print(f"Directory {root} already exists")
        ans = input("Delete it? (y/n) ").strip().lower()
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
        fps=control_hz,
        features=DATASET_FEATURES,
        image_writer_threads=10,
        image_writer_processes=5,
    )
    return dataset


def preprocess_image(image_array: np.ndarray, image_size: int) -> np.ndarray:
    """Resize image to square size."""
    image = Image.fromarray(image_array)
    image = image.resize((image_size, image_size))
    return np.array(image)


def _sample_xy(x_range, y_range):
    return np.array(
        [np.random.uniform(*x_range), np.random.uniform(*y_range)], dtype=np.float32
    )


def randomize_scene_objects(env: SimpleEnv2):
    """
    Randomize mug and plate positions with wider ranges so changes are visible.
    """
    plate_z = 0.82
    mug_z = 0.83

    plate_xy = _sample_xy((0.27, 0.34), (-0.30, -0.15))

    max_attempts = 300
    for _ in range(max_attempts):
        red_xy = _sample_xy((0.27, 0.36), (-0.05, 0.24))
        blue_xy = _sample_xy((0.27, 0.36), (-0.05, 0.24))

        if np.linalg.norm(red_xy - blue_xy) < 0.12:
            continue
        if np.linalg.norm(red_xy - plate_xy) < 0.14:
            continue
        if np.linalg.norm(blue_xy - plate_xy) < 0.14:
            continue

        p_mug_red = np.array([red_xy[0], red_xy[1], mug_z], dtype=np.float32)
        p_mug_blue = np.array([blue_xy[0], blue_xy[1], mug_z], dtype=np.float32)
        p_plate = np.array([plate_xy[0], plate_xy[1], plate_z], dtype=np.float32)

        env.set_obj_pose(p_mug_red, p_mug_blue, p_plate)
        env.obj_init_pose = np.concatenate([p_mug_red, p_mug_blue, p_plate], dtype=np.float32)
        return p_mug_red, p_mug_blue, p_plate

    raise RuntimeError("Failed to sample non-colliding randomized object positions")


def _get_target_color(env: SimpleEnv2) -> str:
    """Infer task color token from environment target object."""
    target = getattr(env, "obj_target", "")
    if target == "body_obj_mug_5":
        return "red"
    if target == "body_obj_mug_6":
        return "blue"

    instruction = getattr(env, "instruction", "").lower()
    if "red" in instruction:
        return "red"
    return "blue"


def randomize_instruction(env: SimpleEnv2) -> str:
    """
    Sample a paraphrased instruction while preserving target mug color.
    """
    color = _get_target_color(env)
    template = random.choice(INSTRUCTION_TEMPLATES)
    instruction = template.format(color=color)
    env.set_instruction(given=instruction)
    return instruction


def reset_env_with_randomization(env: SimpleEnv2, reset_seed, episode_id: int):
    env.reset(seed=reset_seed)
    p_red, p_blue, p_plate = env.get_obj_pose()
    if FORCE_SCENE_RANDOMIZATION:
        p_red, p_blue, p_plate = randomize_scene_objects(env)
    instruction = randomize_instruction(env)
    print(
        f"[Episode {episode_id}] init poses "
        f"red={np.round(p_red, 3)} blue={np.round(p_blue, 3)} plate={np.round(p_plate, 3)}"
    )
    print(f"[Episode {episode_id}] instruction: {instruction}")


def collect_demonstrations(
    env: SimpleEnv2,
    dataset: LeRobotDataset,
    num_demo: int,
    control_hz: int,
    image_size: int,
    reset_seed,
):
    """Run teleoperation loop and collect data."""
    episode_id = 0
    record_flag = False

    print("\n" + "=" * 50)
    print(f"Starting collection, target episodes: {num_demo}")
    print("Move robot to start recording. Press Z to discard current episode.")
    print("=" * 50 + "\n")

    while env.env.is_viewer_alive() and episode_id < num_demo:
        env.step_env()

        if env.env.loop_every(HZ=control_hz):
            if env.check_success():
                dataset.save_episode()
                episode_id += 1
                record_flag = False
                print(f"[OK] Episode {episode_id}/{num_demo} completed")
                if episode_id < num_demo:
                    reset_env_with_randomization(env, reset_seed, episode_id)
                    print(f"Next instruction: {env.instruction}")
                continue

            action, reset = env.teleop_robot()

            if not record_flag and np.sum(np.abs(action)) > 0:
                record_flag = True
                print("Recording started...")

            if reset:
                reset_env_with_randomization(env, reset_seed, episode_id)
                dataset.clear_episode_buffer()
                record_flag = False
                print("Episode cleared and environment reset")
                continue

            agent_image, wrist_image = env.grab_image()
            agent_image = preprocess_image(agent_image, image_size=image_size)
            wrist_image = preprocess_image(wrist_image, image_size=image_size)

            joint_q = env.step(action)
            action_to_save = env.q[:7].astype(np.float32)

            if record_flag:
                dataset.add_frame(
                    {
                        "observation.image": agent_image,
                        "observation.wrist_image": wrist_image,
                        "observation.state": joint_q[:6],
                        "action": action_to_save,
                        "obj_init": env.obj_init_pose,
                    },
                    task=env.instruction,
                )

            env.render(teleop=True, idx=episode_id)

    return episode_id


def cleanup(env: SimpleEnv2, dataset: LeRobotDataset):
    """Close viewer and remove temporary images cache."""
    env.env.close_viewer()
    images_path = dataset.root / "images"
    if images_path.exists():
        shutil.rmtree(images_path)
        print("Temporary images folder cleaned up")


def main():
    args = parse_args()

    print("=" * 60)
    print("Language Data Collection (Random Reset)")
    print("=" * 60)
    print(f"Dataset root: {args.root}")
    print(f"Dataset repo: {args.repo_name}")
    print(f"Episodes: {args.num_demo}")
    print(f"Reset seed: {RESET_SEED} (None means randomized per reset)")

    if BASE_SEED is not None:
        np.random.seed(BASE_SEED)
        random.seed(BASE_SEED)
        print(f"Base random seed enabled: {BASE_SEED}")

    print(f"\nLoading environment: {args.xml_path}")
    env = SimpleEnv2(args.xml_path, seed=RESET_SEED, state_type="joint_angle")
    reset_env_with_randomization(env, RESET_SEED, episode_id=0)

    dataset = create_or_load_dataset(args.root, args.repo_name, args.control_hz)

    try:
        collected = collect_demonstrations(
            env=env,
            dataset=dataset,
            num_demo=args.num_demo,
            control_hz=args.control_hz,
            image_size=args.image_size,
            reset_seed=RESET_SEED,
        )
        print("\n" + "=" * 50)
        print(f"Collection completed. Total episodes: {collected}")
        print("=" * 50)
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
    finally:
        cleanup(env, dataset)
        print("Done")


if __name__ == "__main__":
    main()
