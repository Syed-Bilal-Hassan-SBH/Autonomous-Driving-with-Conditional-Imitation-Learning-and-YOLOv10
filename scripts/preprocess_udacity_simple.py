#!/usr/bin/env python3
"""
Simple preprocessing for Udacity dataset - copy existing processed images and create YOLO annotations
"""

import os
import json
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_udacity_yolo_simple():
    """Simple Udacity to YOLO conversion using existing processed images"""
    
    # Paths
    base_dir = Path("c:/Users/MASTER/Desktop/Sem 07/CV/Self-Driving Cars/YOLOv10_Autonomous_Driving")
    processed_dir = base_dir / "data/udacity/processed/images"
    output_dir = base_dir / "data/udacity_yolo"
    
    # Create output directories
    dirs_to_create = [
        output_dir / "images" / "train",
        output_dir / "images" / "val", 
        output_dir / "images" / "test",
        output_dir / "labels" / "train",
        output_dir / "labels" / "val",
        output_dir / "labels" / "test"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all processed images
    if not processed_dir.exists():
        logger.error(f"Processed images directory not found: {processed_dir}")
        return
    
    # Get all jpg files
    image_files = list(processed_dir.glob("*.jpg"))
    logger.info(f"Found {len(image_files)} processed images")
    
    if not image_files:
        logger.error("No images found in processed directory")
        return
    
    # Create dummy YOLO annotations
    def create_dummy_yolo_annotation():
        """Create a dummy YOLO annotation for road detection"""
        return "0 0.5 0.5 1.0 1.0"
    
    # Split images into train/val/test (70/15/15)
    total_images = len(image_files)
    train_count = int(total_images * 0.7)
    val_count = int(total_images * 0.15)
    test_count = total_images - train_count - val_count
    
    train_images = image_files[:train_count]
    val_images = image_files[train_count:train_count + val_count]
    test_images = image_files[train_count + val_count:]
    
    logger.info(f"Split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    # Process each split
    splits = [
        (train_images, "train"),
        (val_images, "val"), 
        (test_images, "test")
    ]
    
    processed_count = 0
    
    for split_images, split_name in splits:
        logger.info(f"Processing {split_name} split...")
        
        for img_path in split_images:
            try:
                # Copy image to YOLO dataset
                img_dest = output_dir / "images" / split_name / img_path.name
                if not img_dest.exists():
                    shutil.copy2(img_path, img_dest)
                
                # Create YOLO annotation
                label_dest = output_dir / "labels" / split_name / (img_path.stem + ".txt")
                if not label_dest.exists():
                    with open(label_dest, 'w') as f:
                        f.write(create_dummy_yolo_annotation())
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
    
    logger.info(f"Successfully processed {processed_count} samples")
    
    # Create dataset.yaml file
    yaml_content = f"""# Udacity dataset for YOLOv10 training
path: {output_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: road

nc: 1
"""
    
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    # Create class mapping file
    with open(output_dir / "classes.txt", 'w') as f:
        f.write("road\n")
    
    # Save dataset statistics
    stats = {
        "total_samples": processed_count,
        "train_samples": len(train_images),
        "val_samples": len(val_images), 
        "test_samples": len(test_images),
        "classes": ["road"],
        "num_classes": 1
    }
    
    with open(output_dir / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Udacity YOLO dataset created at {output_dir}")
    logger.info(f"Dataset statistics: {stats}")
    
    return output_dir

if __name__ == "__main__":
    create_udacity_yolo_simple()
