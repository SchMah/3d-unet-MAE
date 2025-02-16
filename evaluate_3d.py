import os
import glob
import pandas as pd
import numpy as np
from config import cfg
from medpy import metric
from data_utils import load_patient_volume
from model_unet3d import build_3d_model

def calculate_3d_brats_metrics(y_true, y_pred):
    def get_metrics(y_t, y_p):
        #dice
        intersection = np.sum(y_t * y_p)
        union = np.sum(y_t) + np.sum(y_p)
        dice = (2. * intersection) / union if union > 0 else 1.0
        #hd95
        if np.sum(y_t) > 0 and np.sum(y_p) > 0:
            hd95 = metric.binary.hd95(y_p, y_t, voxelspacing=(1, 1, 1))
        else:
            if np.sum(y_t) == np.sum(y_p):
                hd95 = 0.0
            else:
                hd95 = 373.0
            
        return dice, hd95

    wt_t, wt_p = (y_true > 0), (y_pred > 0)
    tc_t, tc_p = np.isin(y_true, [1, 3]), np.isin(y_pred, [1, 3])
    et_t, et_p = (y_true == 3), (y_pred == 3)

    wt_dice, wt_hd = get_metrics(wt_t, wt_p)
    tc_dice, tc_hd = get_metrics(tc_t, tc_p)
    et_dice, et_hd = get_metrics(et_t, et_p)

    return {
        'WT_Dice': wt_dice, 'WT_HD95': wt_hd,
        'TC_Dice': tc_dice, 'TC_HD95': tc_hd,
        'ET_Dice': et_dice, 'ET_HD95': et_hd
    }

def sliding_window_inference(model, image_vol, patch_size=cfg.PATCH_SIZE, stride=(64, 64, 64)):
    h, w, d, c = image_vol.shape
    ph, pw, pd = patch_size
    sh, sw, sd = stride
    
    prob_map = np.zeros((h, w, d, cfg.NUM_CLASSES), dtype=np.float32)
    count_map = np.zeros((h, w, d, cfg.NUM_CLASSES), dtype=np.float32)
    
    for i in range(0, h - ph + sh, sh):
        for j in range(0, w - pw + sw, sw):
            for k in range(0, d - pd + sd, sd):
                start_h, start_w, start_d = min(i, h - ph), min(j, w - pw), min(k, d - pd)
                end_h, end_w, end_d = start_h + ph, start_w + pw, start_d + pd
                
                patch = image_vol[start_h:end_h, start_w:end_w, start_d:end_d, :]
                patch_input = np.expand_dims(patch, axis=0)
                pred = model.predict(patch_input, verbose=0)[0] 
                
                prob_map[start_h:end_h, start_w:end_w, start_d:end_d, :] += pred
                count_map[start_h:end_h, start_w:end_w, start_d:end_d, :] += 1.0
                
    final_probs = prob_map / count_map
    final_mask = np.argmax(final_probs, axis=-1)
    
    return final_mask

def evaluate_test_set():    
    
    all_dirs = sorted(glob.glob(os.path.join(cfg.DATASET_ROOT, "BraTS*")))
    test_dirs = all_dirs[cfg.TEST_PATIENTS_RANGE[0]:cfg.TEST_PATIENTS_RANGE[1]]
    model = build_3d_model(mode='segmentation')
    weights_path = '3d_weights/best_3d_segmentation.keras'
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        return
    
    all_results = []
    ind_results = []
    
    for i, path in enumerate(test_dirs): 
        img_vol, true_mask = load_patient_volume(path)
        predicted_mask = sliding_window_inference(model, img_vol)
        metrics = calculate_3d_brats_metrics(true_mask, predicted_mask)
        print(f"Predicting patient{i}")
        all_results.append(metrics)
        metrics['patient_id'] = os.path.basename(path)
        metrics['mean_dice'] = (metrics['WT_Dice'] + metrics['TC_Dice'] + metrics['ET_Dice']) / 3
        ind_results.append(metrics)

    df_indv = pd.DataFrame(ind_results)
    df_indv.to_csv("results_3d/indv_test_results.csv", index=False)
    
    df = pd.DataFrame(all_results)
    
    summary = pd.DataFrame({
        'Region': ['WT', 'TC', 'ET'],
        'Mean Dice': [df['WT_Dice'].mean(), df['TC_Dice'].mean(), df['ET_Dice'].mean()],
        'Mean HD95': [df['WT_HD95'].mean(), df['TC_HD95'].mean(), df['ET_HD95'].mean()],
        'Std Dice':  [df['WT_Dice'].std(),  df['TC_Dice'].std(),  df['ET_Dice'].std()]
    })
    summary.to_csv("results_3d/final_metrics_3d_dice_Hd95.csv", index=False)
    print(summary)
        
if __name__ == '__main__':
    evaluate_test_set()