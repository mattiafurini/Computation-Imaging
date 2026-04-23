import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import numpy as np

def populate_dataset(num_images=4000, target_size=(256, 256), seed=42):
    print("Loading LSUN Church dataset from HuggingFace...")
    # This dataset has 'train' split
    dataset = load_dataset("tglcourse/lsun_church_train", split="train", streaming=True)
    
    # Take the first num_images
    dataset_iter = iter(dataset)
    images = []
    
    print(f"Downloading and resizing {num_images} images...")
    
    # We will save them directly to avoid keeping all in RAM
    data_dir = Path(__file__).parent.parent / "data"
    raw_dir = data_dir / "lsun_church"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for item in tqdm(dataset_iter, total=num_images):
        if count >= num_images:
            break
        
        img = item['image']
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize to target size (256x256)
        # Using bicubic interpolation for resizing the ground truth
        img_resized = img.resize(target_size, Image.Resampling.BICUBIC)
        
        # Save to raw directory
        save_path = raw_dir / f"church_{count:04d}.png"
        img_resized.save(save_path)
        
        count += 1
        
    print("Dataset downloaded and resized.")
    print("Creating Train/Val/Test splits...")
    
    # Split configuration
    train_ratio = 0.8
    val_ratio = 0.1
    # test_ratio is the rest (0.1)
    
    num_train = int(num_images * train_ratio)
    num_val = int(num_images * val_ratio)
    
    # Randomize indices
    np.random.seed(seed)
    indices = np.random.permutation(num_images)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train+num_val]
    test_indices = indices[num_train+num_val:]
    
    def move_to_split(indices, split_name):
        split_dir = data_dir / split_name
        split_dir.mkdir(exist_ok=True)
        for idx in tqdm(indices, desc=f"Moving to {split_name}"):
            src = raw_dir / f"church_{idx:04d}.png"
            dst = split_dir / f"church_{idx:04d}.png"
            # Just copy or link, here we rename/move
            os.rename(src, dst)
            
    move_to_split(train_indices, "train")
    move_to_split(val_indices, "val")
    move_to_split(test_indices, "test")
    
    # Clean up empty raw dir
    if not os.listdir(raw_dir):
        os.rmdir(raw_dir)
        
    print(f"Done! Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

if __name__ == "__main__":
    populate_dataset()
