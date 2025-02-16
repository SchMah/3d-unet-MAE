import os
import glob
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from config import cfg
from data_generator import data_generator_brats
from model_unet3d import build_3d_model
from visualization import save_training_curves, save_confusion_matrix, generate_visual_samples

def combined_dice_loss_3d(y_true, y_pred):
    """
    averages the Dice score for three regions.
    """

    y_true_oh = tf.one_hot(tf.cast(y_true[..., 0], tf.int32), depth=cfg.NUM_CLASSES)
    num = 2. * tf.reduce_sum(y_true_oh * y_pred, axis=(1, 2, 3))
    den = tf.reduce_sum(y_true_oh + y_pred, axis=(1, 2, 3))
    dice_per_class = (num + 1e-6) / (den + 1e-6)
    
    relevant_dice = dice_per_class[:, 1:] 
    
    return 1 - tf.reduce_mean(relevant_dice)

def combined_loss_fn(y_true, y_pred):
    """
    combined dice + categorical cross-cntropy loss.
    """
    dice_loss = combined_dice_loss_3d(y_true, y_pred)

    y_true_oh = tf.one_hot(tf.cast(y_true[..., 0], tf.int32), depth=cfg.NUM_CLASSES)
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true_oh, y_pred)
    ce_loss = tf.reduce_mean(ce_loss)
    
    return dice_loss + (0.5 * ce_loss)


def main():
    print("3D U-Net supervised fine tuning")
    all_dirs = sorted(glob.glob(os.path.join(cfg.DATASET_ROOT, "BraTS*")))
    
    training_pool = all_dirs[:cfg.TRAIN_PATIENTS]
    split_idx = int(len(training_pool) * cfg.TRAIN_RATIO)
    train_dirs = training_pool[:split_idx]
    val_dirs = training_pool[split_idx:]

    train_gen = data_generator_brats(train_dirs, is_training=True)
    val_gen = data_generator_brats(val_dirs, is_training=False)

    model = build_3d_model(mode='segmentation')

    mae_weights_path = '3d_weights/mae_pretrained_encoder.keras'
    if os.path.exists(mae_weights_path):
        print(f"loading pre-trained MAE weights from {mae_weights_path}...")
        model.load_weights(mae_weights_path, skip_mismatch=True)
        print("transfer learning successful!")
    else:
        print("MAE weights not found...training 3D U-Net from scratch.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE), 
        loss=combined_loss_fn,
        metrics=['accuracy']
    )
    
    os.makedirs('3d_weights', exist_ok=True)
    callbacks = [
        ModelCheckpoint('3d_weights/best_3d_segmentation.keras', save_best_only=True, monitor='val_loss'),
        CSVLogger('3d_weights/3d_training_log.csv'),
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=5)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=cfg.EPOCHS, callbacks=callbacks)
    
    print("\ncompleted. Best model saved to '3d_weights/best_3d_segmentation.keras'.")


    print("Saving training curves...")
    save_training_curves(log_path='3d_weights/3d_training_log.csv', save_path='results_3d/curves.png')

    
if __name__ == '__main__':
    main()