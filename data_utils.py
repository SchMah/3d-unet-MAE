import numpy as np
import random
import os
import nibabel as nib
import glob

def normalize_volume(volume):
    """
    apply Z-score normalization to each MRI channel.
    """
    normalized = np.zeros_like(volume, dtype=np.float32)
    
    for c in range(volume.shape[-1]):
        channel_data = volume[..., c]

        brain_mask = channel_data > 0 
        
        if np.any(brain_mask):
            mean = channel_data[brain_mask].mean()
            std = channel_data[brain_mask].std()
            
            normalized[..., c] = np.where(brain_mask, (channel_data - mean) / (std + 1e-8), 0)
            
    return normalized

def load_patient_volume(patient_dir):
    """
    Loads 4 MRI images and the mask. Stack them and normalzie
    """
    t1n_path = glob.glob(os.path.join(patient_dir, '*t1n.nii.gz'))[0]
    t1c_path = glob.glob(os.path.join(patient_dir, '*t1c.nii.gz'))[0]
    t2w_path = glob.glob(os.path.join(patient_dir, '*t2w.nii.gz'))[0]
    flair_path = glob.glob(os.path.join(patient_dir, '*t2f.nii.gz'))[0]
    mask_path = glob.glob(os.path.join(patient_dir, '*seg.nii.gz'))[0]

    t1n = nib.load(t1n_path).get_fdata()
    t1c = nib.load(t1c_path).get_fdata()
    t2w = nib.load(t2w_path).get_fdata()
    flair = nib.load(flair_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    mask[mask == 4] = 3  

    img_vol = np.stack([t1n, t1c, t2w, flair], axis=-1)
    
    img_vol = normalize_volume(img_vol)

    return img_vol, mask



def extract_3d_patch(image_vol, mask_vol, patch_size=(128, 128, 128), tumor_prob=0.5):
    """
    extracts a 3D patch from the volume. 
    image_vol shape: (240, 240, 155, 4)
    mask_vol shape:  (240, 240, 155)

    """
    h, w, d, _ = image_vol.shape
    ph, pw, pd = patch_size
    
    max_h = h - ph
    max_w = w - pw
    max_d = d - pd

    if random.random() < tumor_prob and np.sum(mask_vol > 0) > 0:
        tumor_indices = np.argwhere(mask_vol > 0)        
        center = random.choice(tumor_indices)
        start_h = center[0] - (ph // 2)
        start_w = center[1] - (pw // 2)
        start_d = center[2] - (pd // 2)
        
    else:
        start_h = random.randint(0, max_h)
        start_w = random.randint(0, max_w)
        start_d = random.randint(0, max_d)

    start_h = max(0, min(start_h, max_h))
    start_w = max(0, min(start_w, max_w))
    start_d = max(0, min(start_d, max_d))

    end_h = start_h + ph
    end_w = start_w + pw
    end_d = start_d + pd
    
    patch_img = image_vol[start_h:end_h, start_w:end_w, start_d:end_d, :]
    patch_mask = mask_vol[start_h:end_h, start_w:end_w, start_d:end_d]

    return patch_img, patch_mask