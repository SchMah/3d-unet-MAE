import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from config import cfg
from data_generator import data_generator_brats
from model_unet3d import build_3d_model

class MAEGenerator(tf.keras.utils.Sequence):
    """
    """
    def __init__(self, base_generator, mask_ratio=cfg.MASK_RATIO, patch_size=16):
        self.base_generator = base_generator
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size 

    def __len__(self):
        return len(self.base_generator)

    def __getitem__(self, idx):
        X_real, _ = self.base_generator[idx]
        batch_size, h, w, d, c = X_real.shape        
        gh, gw, gd = h // self.patch_size, w // self.patch_size, d // self.patch_size       
        X_masked = np.copy(X_real)
        
        for b in range(batch_size):
            grid_mask = np.random.rand(gh, gw, gd) > self.mask_ratio
            full_mask = np.kron(grid_mask, np.ones((self.patch_size, self.patch_size, self.patch_size)))
            full_mask = np.expand_dims(full_mask, axis=-1) 
            X_masked[b] = X_masked[b] * full_mask
            
        return X_masked, X_real

def run_mae_pretraining():
    print("\n--- starting masked autoencoder training ---")

    all_dirs = sorted(glob.glob(os.path.join(cfg.DATASET_ROOT, "BraTS*")))
    
    training_pool = all_dirs[:cfg.TRAIN_PATIENTS]
    
    split_idx = int(len(training_pool) * 0.8)
    train_dirs = training_pool[:split_idx]
    val_dirs = training_pool[split_idx:]
    
    base_train = data_generator_brats(train_dirs, is_training=False) 
    base_val = data_generator_brats(val_dirs, is_training=False)
    
    mae_train_gen = MAEGenerator(base_train)
    mae_val_gen = MAEGenerator(base_val)

    model = build_3d_model(mode='mae_pretrain')
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE), 
                  loss='mse')
    os.makedirs('3d_weights', exist_ok=True)
    callbacks = [
        ModelCheckpoint('3d_weights/mae_pretrained_encoder.keras', save_best_only=True, monitor='val_loss'),
        CSVLogger('3d_weights/mae_log.csv')
    ]
    model.fit(mae_train_gen, validation_data=mae_val_gen, epochs=50, callbacks=callbacks)
    
    print("Pre-training complete. Weights saved to 3d_weights/mae_pretrained_encoder.keras")

if __name__ == '__main__':
    run_mae_pretraining()