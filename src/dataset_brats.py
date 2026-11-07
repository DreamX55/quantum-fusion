import os
import random
import torch
from torch.utils.data import Dataset
import monai.transforms as transforms
import nibabel as nib
import numpy as np

MODALITY_FILENAMES = {
    "flair": "flair.nii.gz",
    "t1": "t1.nii.gz",
    "t1ce": "t1ce.nii.gz",
    "t2": "t2.nii.gz",
    "seg": "seg.nii.gz"
}

class BraTS2021Dataset(Dataset):
    def __init__(self, data_dir="data/brats2021", modalities=["flair"], split="train", val_ratio=0.20, seed=42, transform=None):
        self.data_dir = data_dir
        self.modalities = [m.lower() for m in modalities]
        self.transform = transform
        self.split = split
        
        all_subjects = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d)) and not d.startswith(".")
        ])
        
        valid_subjects = []
        for s in all_subjects:
            s_dir = os.path.join(self.data_dir, s)
            files = os.listdir(s_dir)
            has_all = all(any(f"{s}_{MODALITY_FILENAMES[m]}" in f for f in files) for m in (self.modalities + ["seg"]))
            if has_all:
                valid_subjects.append(s)
                
        rng = random.Random(seed)
        shuffled = list(valid_subjects)
        rng.shuffle(shuffled)
        
        num_val = int(len(shuffled) * val_ratio)
        if split in ["val", "test"]:
            self.subjects = sorted(shuffled[:num_val])
        else:
            self.subjects = sorted(shuffled[num_val:])
            
    def __len__(self):
        return len(self.subjects)
        
    def __getitem__(self, idx):
        sub_id = self.subjects[idx]
        sub_dir = os.path.join(self.data_dir, sub_id)
        
        mod_arrays = []
        for mod in self.modalities:
            fname = os.path.join(sub_dir, f"{sub_id}_{MODALITY_FILENAMES[mod]}")
            img_data = nib.load(fname).get_fdata(dtype=np.float32)
            mask = img_data > 0
            if mask.sum() > 0:
                mean = img_data[mask].mean()
                std = img_data[mask].std() + 1e-8
                img_data = (img_data - mean) / std
                img_data[~mask] = 0.0
            mod_arrays.append(img_data)
            
        stacked_images = np.stack(mod_arrays, axis=0)
        
        seg_path = os.path.join(sub_dir, f"{sub_id}_{MODALITY_FILENAMES['seg']}")
        seg_data = nib.load(seg_path).get_fdata(dtype=np.float32)
        wt_mask = (seg_data > 0).astype(np.float32)[np.newaxis, ...]
        
        data = {
            "image": torch.from_numpy(stacked_images),
            "mask": torch.from_numpy(wt_mask),
            "subject_id": sub_id
        }
        
        if self.transform:
            data = self.transform(data)
            
        return data

def get_brats_transforms():
    return transforms.Compose([
        transforms.Resized(keys=["image", "mask"], spatial_size=(64, 64, 64), mode=["trilinear", "nearest"]),
        transforms.EnsureTyped(keys=["image", "mask"])
    ])
