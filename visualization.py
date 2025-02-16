import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config import cfg

CLASS_NAMES = ['Background', 'NCR', 'Edema', 'ET']

def save_training_curves(log_path='3d_weights/3d_training_log.csv', save_path='results_3d/curves.png'):
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return
        
    df = pd.read_csv(log_path)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['loss'], label='Train')
    plt.plot(df['val_loss'], label='Val')
    plt.title('3D Model Loss History')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(df['accuracy'], label='Train')
    plt.plot(df['val_accuracy'], label='Val')
    plt.title('3D Model Accuracy History')
    plt.legend()
    
    os.makedirs('results_3d', exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def save_confusion_matrix(loader, model, save_path='results_3d/cm.png'):
    y_true_all = []
    y_pred_all = []
    
    for i in range(min(2, len(loader))):
        X, y = loader[i]
        preds = model.predict(X, verbose=0)
        
        y_true_all.extend(y.flatten())
        y_pred_all.extend(np.argmax(preds, axis=-1).flatten())
        
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2, 3])
    
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('3D Model Normalized Confusion Matrix')
    
    os.makedirs('results_3d', exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def generate_visual_samples(loader, model, save_path='results_3d/samples.png'):
    X, y = loader[0]
    preds = model.predict(X, verbose=0)
    preds_max = np.argmax(preds, axis=-1)
    
    z_mid = X.shape[3] // 2 
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i in range(min(3, X.shape[0])): # In case batch size is < 3
        flair_slice = X[i, :, :, z_mid, 3]
        true_mask_slice = y[i, :, :, z_mid, 0]
        pred_mask_slice = preds_max[i, :, :, z_mid]
        
        axes[0, i].imshow(flair_slice, cmap='gray')
        axes[0, i].imshow(np.ma.masked_where(true_mask_slice==0, true_mask_slice), cmap='jet', alpha=0.5)
        axes[0, i].set_title(f"Ground Truth 3D Slice {i}")
        
        axes[1, i].imshow(flair_slice, cmap='gray')
        axes[1, i].imshow(np.ma.masked_where(pred_mask_slice==0, pred_mask_slice), cmap='jet', alpha=0.5)
        axes[1, i].set_title(f"Prediction 3D Slice {i}")
        
    for ax in axes.flatten(): ax.axis('off')
    
    os.makedirs('results_3d', exist_ok=True)
    plt.savefig(save_path)
    plt.close()