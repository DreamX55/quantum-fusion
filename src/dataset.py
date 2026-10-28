import os
from torch.utils.data import Dataset
import monai.transforms as transforms
import nibabel as nib
import torch

class SynthRADDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None, seed=42, val_ratio=0.2):
        """
        Args:
            data_dir (str): Path to the extracted SynthRAD dataset.
            split (str): 'train', 'val', or 'all'.
            transform: MONAI transforms.
            seed (int): Deterministic split seed.
            val_ratio (float): Ratio of held-out validation patients.
        """
        self.data_dir = os.path.join(data_dir, "Task1", "brain")
        self.transform = transform
        self.split = split
        
        self.patient_folders = []
        if os.path.exists(self.data_dir):
            all_folders = sorted([os.path.join(self.data_dir, p) for p in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, p))])
            if split == "all" or len(all_folders) == 0:
                self.patient_folders = all_folders
            else:
                import random
                rng = random.Random(seed)
                shuffled = list(all_folders)
                rng.shuffle(shuffled)
                num_val = int(len(shuffled) * val_ratio)
                if split in ["val", "test"]:
                    self.patient_folders = sorted(shuffled[:num_val])
                else: # "train"
                    self.patient_folders = sorted(shuffled[num_val:])
            
        if len(self.patient_folders) == 0:
            print("WARNING: No data found. Falling back to dummy dataset mode.")
            self.patient_folders = ["dummy_patient_1", "dummy_patient_2", "dummy_patient_3", "dummy_patient_4"]
            
    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, idx):
        patient_path = self.patient_folders[idx]
        
        # Expected filenames (adjust these once we see the exact downloaded dataset structure)
        mri_path = os.path.join(patient_path, 'mr.nii.gz')
        ct_path = os.path.join(patient_path, 'ct.nii.gz')
        mask_path = os.path.join(patient_path, 'mask.nii.gz') # Target segmentations
        
        # For Phase 1 Baseline without actual data, we return dummy tensors if data isn't there yet
        if not os.path.exists(mri_path):
            # Dummy data for debugging the pipeline before download completes
            return {
                "mri": torch.randn((1, 64, 64, 64)),
                "ct": torch.randn((1, 64, 64, 64)),
                "mask": torch.randint(0, 2, (1, 64, 64, 64)).float()
            }
            
        data = {
            "mri": mri_path,
            "ct": ct_path,
            "mask": mask_path
        }
        
        if self.transform:
            data = self.transform(data)
            
        return data

# Default MONAI transform pipeline for 3D Medical Images
def get_transforms():
    return transforms.Compose([
        transforms.LoadImaged(keys=["mri", "ct", "mask"]),
        transforms.EnsureChannelFirstd(keys=["mri", "ct", "mask"]),
        transforms.ScaleIntensityd(keys=["mri", "ct"]),
        transforms.Resized(keys=["mri", "ct", "mask"], spatial_size=(64, 64, 64)), # Downsampled for CPU compatibility
        transforms.EnsureTyped(keys=["mri", "ct", "mask"])
    ])
