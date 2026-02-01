"""
Deploy Trained ACT Policy in Simulation

Load a trained ACT policy model and deploy it in the MuJoCo simulation environment.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torchvision
import numpy as np
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType

from mujoco_env.y_env import SimpleEnv

# ========================
# Configuration Parameters
# ========================
DEVICE = 'cuda'

# Dataset and model paths
DATA_ROOT = './demo_data'
DATASET_NAME = 'omy_pnp'
CHECKPOINT_PATH = './ckpt/act_y'

# Simulation environment configuration
XML_PATH = './asset/example_scene_y.xml'
CONTROL_HZ = 20
IMAGE_SIZE = (256, 256)

# Policy configuration
CHUNK_SIZE = 10
N_ACTION_STEPS = 1
TEMPORAL_ENSEMBLE_COEFF = 0.9

# Random seed
SEED = 0


def load_policy():
    """Load trained ACT policy model"""
    dataset_metadata = LeRobotDatasetMetadata(DATASET_NAME, root=DATA_ROOT)

    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    input_features.pop("observation.wrist_image")

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        temporal_ensemble_coeff=TEMPORAL_ENSEMBLE_COEFF
    )

    policy = ACTPolicy.from_pretrained(
        CHECKPOINT_PATH,
        config=cfg,
        dataset_stats=dataset_metadata.stats
    )
    policy.to(DEVICE)
    policy.eval()

    return policy


def load_environment():
    """Load simulation environment"""
    env = SimpleEnv(XML_PATH, action_type='joint_angle')
    return env


def preprocess_image(image_array):
    """Preprocess image data"""
    img_transform = torchvision.transforms.ToTensor()
    image = Image.fromarray(image_array)
    image = image.resize(IMAGE_SIZE)
    image = img_transform(image)
    return image


def main():
    """Main function"""
    # Load policy and environment
    policy = load_policy()
    env = load_environment()

    # Initialize
    step = 0
    env.reset(seed=SEED)
    policy.reset()

    print("Starting deployment. Close window to exit...")

    # Main loop
    while env.env.is_viewer_alive():
        env.step_env()

        if env.env.loop_every(HZ=CONTROL_HZ):
            # Check if task is successful
            success = env.check_success()
            if success:
                print('Success')
                policy.reset()
                env.reset(seed=SEED)
                step = 0
                continue

            # Get state and images
            state = env.get_ee_pose()
            image, wrist_image = env.grab_image()

            # Preprocess images
            image = preprocess_image(image)
            wrist_image = preprocess_image(wrist_image)

            # Build observation data
            data = {
                'observation.state': torch.tensor([state]).to(DEVICE),
                'observation.image': image.unsqueeze(0).to(DEVICE),
                'observation.wrist_image': wrist_image.unsqueeze(0).to(DEVICE),
                'task': ['Put mug cup on the plate'],
                'timestamp': torch.tensor([step / CONTROL_HZ]).to(DEVICE)
            }

            # Select and execute action
            action = policy.select_action(data)
            action = action[0].cpu().detach().numpy()

            env.step(action)
            env.render()
            step += 1

            # Check success again
            success = env.check_success()
            if success:
                print('Success')
                break


if __name__ == "__main__":
    main()