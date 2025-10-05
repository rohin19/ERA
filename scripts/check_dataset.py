#!/usr/bin/env python3
"""Dataset validation script for YOLO training.
Checks if dataset structure matches cr_data.yaml configuration.
"""
import os
import sys
from pathlib import Path
import yaml


def main():
    project_root = Path(__file__).parent.parent
    yaml_path = project_root / "model" / "cr_data.yaml"
    
    print(f"🔍 Checking dataset from: {yaml_path}")
    
    if not yaml_path.exists():
        print(f"❌ YAML config not found: {yaml_path}")
        return False
        
    # Load YAML config
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build paths correctly (from project root, not from model/)
    dataset_root = project_root / config['path']  # data/dataset_yolov11
    train_imgs = dataset_root / config['train']   # train/images  
    val_imgs = dataset_root / config['val']       # valid/images
    train_labels = dataset_root / config['train'].replace('images', 'labels')
    val_labels = dataset_root / config['val'].replace('images', 'labels')
    
    print(f"📂 Dataset root: {dataset_root}")
    print(f"📂 Train images: {train_imgs}")
    print(f"📂 Val images: {val_imgs}")
    
    # Check existence
    success = True
    
    if not train_imgs.exists():
        print(f"❌ Missing train images: {train_imgs}")
        success = False
    else:
        train_count = len(list(train_imgs.glob("*.jpg"))) + len(list(train_imgs.glob("*.png")))
        print(f"✅ Train images: {train_count} files")
        
    if not val_imgs.exists():
        print(f"❌ Missing val images: {val_imgs}")
        success = False
    else:
        val_count = len(list(val_imgs.glob("*.jpg"))) + len(list(val_imgs.glob("*.png")))
        print(f"✅ Val images: {val_count} files")
        
    if not train_labels.exists():
        print(f"❌ Missing train labels: {train_labels}")
        success = False
    else:
        label_count = len(list(train_labels.glob("*.txt")))
        print(f"✅ Train labels: {label_count} files")
        
    if not val_labels.exists():
        print(f"❌ Missing val labels: {val_labels}")
        success = False
    else:
        label_count = len(list(val_labels.glob("*.txt")))
        print(f"✅ Val labels: {label_count} files")
    
    # Check classes
    classes_file = project_root / "data" / "classes.txt"
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            file_classes = [line.strip() for line in f if line.strip()]
        yaml_classes = config['names']
        
        if file_classes == yaml_classes:
            print(f"✅ Classes match: {len(yaml_classes)} classes")
        else:
            print(f"⚠️  Classes mismatch between classes.txt and YAML")
            print(f"   classes.txt: {file_classes}")
            print(f"   YAML: {yaml_classes}")
    
    if success:
        print("🎉 Dataset validation passed! Ready to train.")
        return True
    else:
        print("❌ Dataset validation failed. Fix paths before training.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)