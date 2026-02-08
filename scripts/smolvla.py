"""
Deploy Trained SmolVLA Policy in MuJoCo Simulation

Usage:
    python deploy_smolvla.py --checkpoint ./ckpt/smolvla_omy/checkpoints/last/pretrained_model
    python deploy_smolvla.py --from_hub Jeongeun/omy_pnp_smolvla
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
from PIL import Image
from torchvision import transforms

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import write_json, serialize_dict, dataset_to_policy_features
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType

from mujoco_env.y_env2 import SimpleEnv2


# ============================================================================
# [Refactor 1] Organize scattered code into functions for reusability and testing
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deploy SmolVLA policy in simulation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./ckpt/smolvla_omy/checkpoints/last/pretrained_model",
        help="Path to local checkpoint"
    )
    parser.add_argument(
        "--from_hub",
        type=str,
        default=None,
        help="Load model from HuggingFace Hub (e.g., Jeongeun/omy_pnp_smolvla)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./omy_pnp_language",
        help="Path to dataset for metadata"
    )
    parser.add_argument(
        "--xml_path",
        type=str,
        default="./asset/example_scene_y2.xml",
        help="Path to MuJoCo scene XML"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run model on (cuda/cpu)"
    )
    parser.add_argument(
        "--control_hz",
        type=int,
        default=20,
        help="Control frequency in Hz"
    )
    # [Refactor 2] Add seed parameter for reproducible experiments
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for environment reset"
    )
    return parser.parse_args()


def get_image_transform():
    """
    Return image preprocessing transform
    Converts PIL image [0-255] to FloatTensor [0.0-1.0]
    """
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def load_dataset_metadata(dataset_path: str):
    """
    Load dataset metadata

    [Refactor 3] Simplified try-except logic from original code
    """
    possible_paths = [
        "./demo_data_language",
        dataset_path,
        "./omy_pnp_language",
    ]

    for path in possible_paths:
        try:
            metadata = LeRobotDatasetMetadata("omy_pnp_language", root=path)
            print(f"Loaded dataset metadata from: {path}")
            return metadata
        except Exception:
            continue

    raise FileNotFoundError(f"Could not load dataset metadata from any of: {possible_paths}")


def load_policy(
        dataset_metadata: LeRobotDatasetMetadata,
        checkpoint_path: str = None,
        hub_repo: str = None,
        device: str = "cuda"
) -> SmolVLAPolicy:
    """
    Load SmolVLA policy model

    Args:
        dataset_metadata: Dataset metadata
        checkpoint_path: Local checkpoint path
        hub_repo: HuggingFace Hub repository ID
        device: Device to run on

    Returns:
        Loaded SmolVLAPolicy model
    """
    # Build feature configuration
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # Create policy configuration
    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=5,
        n_action_steps=5
    )

    # Load model
    if hub_repo:
        print(f"Loading model from HuggingFace Hub: {hub_repo}")
        policy = SmolVLAPolicy.from_pretrained(
            hub_repo,
            config=cfg,
            dataset_stats=dataset_metadata.stats
        )
    else:
        print(f"Loading model from local checkpoint: {checkpoint_path}")
        policy = SmolVLAPolicy.from_pretrained(
            checkpoint_path,
            dataset_stats=dataset_metadata.stats
        )

    policy.to(device)
    policy.eval()

    return policy


def prepare_observation(
        env: SimpleEnv2,
        transform: transforms.Compose,
        device: str
) -> dict:
    """
    Get observation data from environment and convert to model input format

    [Refactor 4] Extract observation preparation logic into standalone function
    [Refactor 5] Fix UserWarning from original code: use np.array() to avoid performance issues when creating tensor from list of numpy arrays
    """
    # Get joint states (first 6 joints)
    state = env.get_joint_state()[:6]

    # Get images
    image, wrist_image = env.grab_image()

    # Process main camera image
    image = Image.fromarray(image)
    image = image.resize((256, 256))
    image = transform(image)

    # Process wrist camera image
    wrist_image = Image.fromarray(wrist_image)
    wrist_image = wrist_image.resize((256, 256))
    wrist_image = transform(wrist_image)

    # [Refactor 5] Use np.array() conversion to avoid performance warning
    state_array = np.array(state, dtype=np.float32)

    # Build model input
    data = {
        'observation.state': torch.tensor(state_array).unsqueeze(0).to(device),
        'observation.image': image.unsqueeze(0).to(device),
        'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
        'task': [env.instruction],
    }

    return data


def run_deployment(
        policy: SmolVLAPolicy,
        env: SimpleEnv2,
        transform: transforms.Compose,
        device: str,
        control_hz: int = 20,
        seed: int = 0
):
    """
    Run policy deployment loop

    [Refactor 6] Encapsulate main loop logic into function with more state tracking
    """
    step = 0
    episode = 0

    env.reset(seed=seed)
    policy.reset()

    print(f"\n{'=' * 60}")
    print(f"Starting deployment - Episode {episode}")
    print(f"Instruction: {env.instruction}")
    print(f"{'=' * 60}\n")

    while env.env.is_viewer_alive():
        env.step_env()

        if env.env.loop_every(HZ=control_hz):
            # Check if task is completed
            success = env.check_success()

            if success:
                print(f"\n[Episode {episode}] Success at step {step}!")

                # Reset environment and policy
                policy.reset()
                env.reset()
                step = 0
                episode += 1

                print(f"\n{'=' * 60}")
                print(f"Starting deployment - Episode {episode}")
                print(f"Instruction: {env.instruction}")
                print(f"{'=' * 60}\n")
                continue

            # Prepare observation data
            observation = prepare_observation(env, transform, device)

            # Model inference
            with torch.no_grad():
                action = policy.select_action(observation)

            # Extract action (7-dim: 6 joints + 1 gripper)
            action = action[0, :7].cpu().numpy()

            # Execute action
            env.step(action)
            env.render(idx=episode)

            step += 1

            # [Refactor 7] Add progress printing
            if step % 50 == 0:
                print(f"  Step {step}...")

    print("\nViewer closed. Deployment ended.")


def main():
    """Main function"""
    args = parse_args()

    print("=" * 60)
    print("SmolVLA Policy Deployment")
    print("=" * 60)

    # 1. Load dataset metadata
    print("\n[1/4] Loading dataset metadata...")
    dataset_metadata = load_dataset_metadata(args.dataset_path)

    # 2. Load policy model
    print("\n[2/4] Loading SmolVLA policy model...")
    policy = load_policy(
        dataset_metadata=dataset_metadata,
        checkpoint_path=args.checkpoint,
        hub_repo=args.from_hub,
        device=args.device
    )

    # 3. Initialize environment
    print("\n[3/4] Initializing MuJoCo environment...")
    env = SimpleEnv2(args.xml_path, action_type='joint_angle')

    # 4. Run deployment
    print("\n[4/4] Starting deployment loop...")
    transform = get_image_transform()

    run_deployment(
        policy=policy,
        env=env,
        transform=transform,
        device=args.device,
        control_hz=args.control_hz,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
