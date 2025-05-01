import os
import shutil
import random

# Define the paths
base_path = 'OCT2017/'  # Adjust this if your dataset is in a different directory
mini_dataset_path = 'mini_dataset/'

# Create the mini_dataset directory structure
os.makedirs(mini_dataset_path, exist_ok=True)
for split in ['train', 'test']:
    split_path = os.path.join(base_path, split)
    mini_split_path = os.path.join(mini_dataset_path, split)
    os.makedirs(mini_split_path, exist_ok=True)
    
    for class_folder in os.listdir(split_path):
        class_folder_path = os.path.join(split_path, class_folder)
        mini_class_folder_path = os.path.join(mini_split_path, class_folder)
        os.makedirs(mini_class_folder_path, exist_ok=True)

        # Get all images in the class folder and sample 10%
        images = os.listdir(class_folder_path)
        sample_size = max(1, int(len(images) * 0.1))  # Ensure at least 1 image is selected
        sampled_images = random.sample(images, sample_size)

        # Copy sampled images to the mini_dataset structure
        for image in sampled_images:
            src = os.path.join(class_folder_path, image)
            dst = os.path.join(mini_class_folder_path, image)
            shutil.copy(src, dst)

print("Mini dataset created successfully at:", os.path.abspath(mini_dataset_path))
