"""
Download HuggingFace Dataset to Local Directory
"""
from huggingface_hub import snapshot_download
import os

print("=" * 60)
print("Downloading dataset: Jeongeun/omy_pnp_language")
print("=" * 60)
print("\nThis may take a few minutes depending on your network speed...")

# Download entire dataset repository from HuggingFace
repo_id = "Jeongeun/omy_pnp_language"
local_dir = "./demo_data_language"

print(f"\nTarget directory: {local_dir}")
print("Starting download...\n")

try:
    # Download entire repository
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    
    print("\n" + "=" * 60)
    print("Download completed!")
    print("=" * 60)
    print(f"Dataset saved to: {local_dir}")
    
    # Display downloaded file structure
    print("\nFile structure:")
    for root, dirs, files in os.walk(local_dir):
        level = root.replace(local_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show only first 5 files
            print(f'{subindent}{file}')
        if len(files) > 5:
            print(f'{subindent}... and {len(files)-5} more files')
    
    print("=" * 60)
    
except Exception as e:
    print(f"\nDownload failed: {e}")
    print("\nAlternative solutions:")
    print("1. Check network connection")
    print("2. Ensure logged into HuggingFace: huggingface-cli login")
    print(f"3. Manually visit: https://huggingface.co/datasets/{repo_id}")
