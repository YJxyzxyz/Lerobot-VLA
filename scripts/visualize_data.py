"""
Dataset Visualization Script

Replay and visualize collected demonstration data in the MuJoCo simulation.

The main simulation window shows the robot replaying recorded actions.
Overlay images (top-right and bottom-right) display the original dataset images
for comparison with the simulation.

Usage:
    python visualize_data.py
    python visualize_data.py --episode 2
    python visualize_data.py --root ./demo_data_example
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import write_json, serialize_dict
from mujoco_env.y_env import SimpleEnv

# ==================== Configuration ====================
DEFAULT_REPO_NAME = 'omy_pnp'
DEFAULT_ROOT = './demo_data'
DEFAULT_EPISODE_INDEX = 0
XML_PATH = './asset/example_scene_y.xml'


class EpisodeSampler(Sampler):
    """
    Sampler that iterates through frames of a single episode.

    Args:
        dataset: LeRobotDataset instance
        episode_index: Index of the episode to sample from
    """

    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def load_dataset(repo_name: str, root: str) -> LeRobotDataset:
    """
    Load the LeRobot dataset.

    Args:
        repo_name: Name of the dataset repository
        root: Root directory of the dataset

    Returns:
        LeRobotDataset instance
    """
    print(f"Loading dataset from {root}...")
    dataset = LeRobotDataset(repo_name, root=root)
    print(f"Dataset loaded: {len(dataset)} frames")
    return dataset


def create_dataloader(dataset: LeRobotDataset, episode_index: int) -> tuple:
    """
    Create a DataLoader for a specific episode.

    Args:
        dataset: LeRobotDataset instance
        episode_index: Index of the episode to visualize

    Returns:
        Tuple of (DataLoader, EpisodeSampler)
    """
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = DataLoader(
        dataset,
        num_workers=1,
        batch_size=1,
        sampler=episode_sampler,
    )
    print(f"Episode {episode_index}: {len(episode_sampler)} frames")
    return dataloader, episode_sampler


def process_image(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert image tensor from dataset format to display format.

    Args:
        image_tensor: Image tensor of shape (C, H, W) with values in [0, 1]

    Returns:
        NumPy array of shape (H, W, C) with values in [0, 255]
    """
    # Convert to numpy and scale to [0, 255]
    image = image_tensor.numpy() * 255
    image = image.astype(np.uint8)
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    return image


def visualize_episode(env: SimpleEnv, dataloader: DataLoader,
                      episode_sampler: EpisodeSampler):
    """
    Main visualization loop that replays an episode in simulation.

    Args:
        env: MuJoCo simulation environment
        dataloader: DataLoader for the episode
        episode_sampler: Sampler for the episode (used to get length)
    """
    step = 0
    iter_dataloader = iter(dataloader)
    env.reset()

    print("\n" + "=" * 50)
    print("Starting Visualization")
    print("Close the viewer window to exit")
    print("=" * 50 + "\n")

    while env.env.is_viewer_alive():
        env.step_env()

        if env.env.loop_every(HZ=20):
            # Get data from dataset
            data = next(iter_dataloader)

            # Set object pose on first frame
            if step == 0:
                obj_init = data['obj_init'][0]
                env.set_obj_pose(obj_init[:3], obj_init[3:])

            # Execute action from dataset
            action = data['action'].numpy()[0]
            env.step(action)

            # Overlay dataset images for comparison
            env.rgb_agent = process_image(data['observation.image'][0])
            env.rgb_ego = process_image(data['observation.wrist_image'][0])
            env.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)

            # Render the scene
            env.render()
            step += 1

            # Loop back to beginning when episode ends
            if step == len(episode_sampler):
                print(f"Episode finished. Restarting...")
                iter_dataloader = iter(dataloader)
                env.reset()
                step = 0


def save_stats(dataset: LeRobotDataset):
    """
    Save dataset statistics to stats.json file.

    Args:
        dataset: LeRobotDataset instance
    """
    stats = dataset.meta.stats
    stats_path = dataset.root / 'meta' / 'stats.json'
    stats = serialize_dict(stats)
    write_json(stats, stats_path)
    print(f"Stats saved to {stats_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize collected demonstration data in simulation"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=DEFAULT_REPO_NAME,
        help=f"Dataset repository name (default: {DEFAULT_REPO_NAME})"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=DEFAULT_ROOT,
        help=f"Dataset root directory (default: {DEFAULT_ROOT})"
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=DEFAULT_EPISODE_INDEX,
        help=f"Episode index to visualize (default: {DEFAULT_EPISODE_INDEX})"
    )
    parser.add_argument(
        "--save-stats",
        action="store_true",
        help="Save dataset statistics to stats.json"
    )
    return parser.parse_args()


def main():
    """Main entry point for the visualization script."""
    args = parse_args()

    print("\n" + "=" * 50)
    print("Dataset Visualization Tool")
    print("=" * 50)
    print(f"\nDataset: {args.repo}")
    print(f"Root: {args.root}")
    print(f"Episode: {args.episode}\n")

    # Load dataset
    dataset = load_dataset(args.repo, args.root)

    # Create dataloader for specified episode
    dataloader, episode_sampler = create_dataloader(dataset, args.episode)

    # Initialize simulation environment
    print("\nInitializing simulation environment...")
    env = SimpleEnv(XML_PATH, action_type='joint_angle')

    try:
        # Run visualization loop
        visualize_episode(env, dataloader, episode_sampler)
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user.")
    except StopIteration:
        print("\nEnd of episode data.")
    finally:
        # Cleanup
        env.env.close_viewer()
        print("Viewer closed.")

    # Optionally save stats
    if args.save_stats:
        save_stats(dataset)

    print("Program terminated.")


if __name__ == "__main__":
    main()