import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from data_utils import extract_3d_patch, load_patient_volume
from config import cfg

class data_generator_brats(Sequence):
    def __init__(self, patient_dirs, batch_size=cfg.BATCH_SIZE, patch_size=cfg.PATCH_SIZE, is_training=True):
        """
        patient_dirs: paths to patient folders.
        batch_size: number of blocks per batch.
        patch_size: (128, 128, 128).
        is_training: If True, forces 50% of crops to center on the tumor.
        """
        self.patient_dirs = patient_dirs
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.is_training = is_training
        
    def __len__(self):
        return len(self.patient_dirs) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_paths = self.patient_dirs[start:end]
        
        X = []
        y = []
        
        for path in batch_paths:
            img_vol, mask_vol = load_patient_volume(path)
            
            tumor_prob = 0.5 if self.is_training else 0.0
            patch_img, patch_mask = extract_3d_patch(
                img_vol, mask_vol, 
                patch_size=self.patch_size, 
                tumor_prob=tumor_prob
            )
            
            patch_mask = np.expand_dims(patch_mask, axis=-1)
            
            X.append(patch_img)
            y.append(patch_mask)
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)