import os

class cfg:
    """ """
    DATASET_ROOT = 'TrainingData' 
    
    TRAIN_PATIENTS = 500  
    TEST_PATIENTS_RANGE = (501, 610)
    TRAIN_RATIO = 0.8
    PATCH_SIZE = (128, 128, 128) 
    
    BATCH_SIZE = 1 
    
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 4
    

    MASK_RATIO = 0.75