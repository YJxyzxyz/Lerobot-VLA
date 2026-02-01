"""
Train Action-Chunking-Transformer (ACT) on Your Dataset

Train ACT model for robot imitation learning using collected demonstration data.
"""

import torch
import numpy as np
import random
from torchvision import transforms
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
import matplotlib.pyplot as plt

# ========================
# Configuration Parameters
# ========================
DEVICE = torch.device("cuda")
TRAINING_STEPS = 3000
LOG_FREQ = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
CHUNK_SIZE = 10
N_ACTION_STEPS = 10
SEED = 42  # Random seed, set to None for non-deterministic behavior

DATA_ROOT = './demo_data'
DATASET_NAME = 'omy_pnp'
CHECKPOINT_PATH = './ckpt/act_y'


# ========================
# Random Seed Setup
# ========================
def set_seed(seed=42):
    """
    Set all random seeds to ensure reproducible experiments
    
    Args:
        seed: Random seed value, if None then don't set
    """
    if seed is None:
        print("Random seed not set, results will vary between runs")
        return None, None
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Set CUDA deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create generator for DataLoader
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Create worker initialization function
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    print(f"✓ Random seed has been set to: {seed}")
    return g, seed_worker


# ========================
# Data Augmentation Classes (Must be defined as top-level classes to be picklable)
# ========================
class AddGaussianNoise:
    """Add Gaussian noise to image tensors"""

    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class ClampTransform:
    """Clamp tensor values to [0, 1] range (replaces lambda function, can be pickled)"""

    def __call__(self, tensor):
        return tensor.clamp(0, 1)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# ========================
# Episode Sampler
# ========================
class EpisodeSampler(torch.utils.data.Sampler):
    """Sample all frames from a single episode in the dataset"""

    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def create_policy(dataset_metadata):
    """Create and initialize ACT policy"""
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # Remove unnecessary features (e.g., wrist camera image)
    input_features.pop("observation.wrist_image", None)

    # Create configuration
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        device='cuda'
    )

    # Get delta_timestamps for action chunking
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

    # Create policy
    policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(DEVICE)

    return policy, delta_timestamps


def create_dataloader(dataset_name, delta_timestamps, image_transforms=None, generator=None, worker_init_fn=None):
    """Create training data loader"""
    dataset = LeRobotDataset(
        dataset_name,
        delta_timestamps=delta_timestamps,
        root=DATA_ROOT,
        image_transforms=image_transforms
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,  # Set to 0 on Windows to avoid multiprocessing issues
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=DEVICE.type != "cpu",
        drop_last=True,
        generator=generator,  # To fix shuffle order
        worker_init_fn=worker_init_fn,  # To fix worker random seeds
    )

    return dataset, dataloader


def train(policy, dataloader, training_steps=TRAINING_STEPS):
    """Train the model"""
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    step = 0
    done = False

    print("start train...")
    while not done:
        for batch in dataloader:
            # Move data to device
            inp_batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}

            # Forward pass
            loss, _ = policy.forward(inp_batch)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % LOG_FREQ == 0:
                print(f"step: {step} loss: {loss.item():.3f}")

            step += 1
            if step >= training_steps:
                done = True
                break

    print("finish!")
    return policy


def evaluate(policy, dataset, episode_index=0):
    """Evaluate model on specified episode"""
    policy.eval()

    episode_sampler = EpisodeSampler(dataset, episode_index)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,  # Set to 0 on Windows
        batch_size=1,
        shuffle=False,
        pin_memory=DEVICE.type != "cpu",
        sampler=episode_sampler,
    )

    actions = []
    gt_actions = []

    policy.reset()
    for batch in test_dataloader:
        inp_batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
        action = policy.select_action(inp_batch)
        actions.append(action)
        gt_actions.append(inp_batch["action"][:, 0, :])

    actions = torch.cat(actions, dim=0)
    gt_actions = torch.cat(gt_actions, dim=0)

    mean_error = torch.mean(torch.abs(actions - gt_actions)).item()
    print(f"Mean action error: {mean_error:.3f}")

    return actions, gt_actions


def plot_results(actions, gt_actions, action_dim=7):
    """Plot comparison between predicted and ground truth actions"""
    fig, axs = plt.subplots(action_dim, 1, figsize=(10, 10))

    for i in range(action_dim):
        axs[i].plot(actions[:, i].cpu().detach().numpy(), label="Prediction")
        axs[i].plot(gt_actions[:, i].cpu().detach().numpy(), label="Truth")
        axs[i].set_ylabel(f"dimension {i}")
        axs[i].legend()

    plt.xlabel("Time step")
    plt.suptitle("Action Prediction vs Truth")
    plt.tight_layout()
    plt.savefig("action_comparison.png", dpi=150)
    plt.show()
    print("Plot saved as action_comparison.png")


def main():
    # =============================
    # Step 1: Set random seed (Most important!)
    # =============================
    generator, worker_init_fn = set_seed(SEED)
    
    print(f"Using device: {DEVICE}")

    # 1. Load dataset metadata
    print("Loading dataset metadata...")
    dataset_metadata = LeRobotDatasetMetadata(DATASET_NAME, root=DATA_ROOT)

    # 2. Create policy
    print("Creating ACT policy...")
    policy, delta_timestamps = create_policy(dataset_metadata)

    # 3. Create data augmentation (using picklable classes instead of lambda)
    image_transforms = transforms.Compose([
        AddGaussianNoise(mean=0., std=0.02),
        ClampTransform()  # Replace transforms.Lambda(lambda x: x.clamp(0, 1))
    ])

    # 4. Create dataloader (pass generator and worker_init_fn)
    print("Creating dataloader...")
    dataset, dataloader = create_dataloader(
        DATASET_NAME,
        delta_timestamps,
        image_transforms=image_transforms,
        generator=generator,
        worker_init_fn=worker_init_fn
    )

    # 5. Train
    policy = train(policy, dataloader, training_steps=TRAINING_STEPS)

    # 6. Save model
    print(f"Saving model to {CHECKPOINT_PATH}...")
    policy.save_pretrained(CHECKPOINT_PATH)

    # 7. Evaluate
    print("Evaluating model...")
    actions, gt_actions = evaluate(policy, dataset, episode_index=0)

    # 8. Plot results
    plot_results(actions, gt_actions)


if __name__ == "__main__":
    main()