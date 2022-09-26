import os
import torch

class TRAIN_CONFIG:
    
    # dataset and model
    DATASET = "eval3.2_cta"
    MODEL_NAME = "clip"
    LABEL_NAMES = ["NON_CTA", "CTA"]
    
    # train configs
    BATCH_SIZE = 32
    EPOCHS = 15
    INITIAL_LR = 2e-5
    WEIGHT_DECAY = 0.001
    PATIENCE = 3
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    RECREATE_PARTITIONS = True
    TRAIN_SPLIT = 0.9
    DEV_SPLIT = 0.1
    TRAIN = True
    TEST = False
    
    # pretrained
    USE_PRETRAINED = True
    PRETRAINED_MODEL_STATES_FILE = "../trained_models/military_cta/clip.weights.best"
    
    # output paths
    TSV_ROOT = f"../processed_data/{DATASET}"
    LOGS_ROOT = f"../logs/{DATASET}"
    RESULTS_ROOT = f"../results/{DATASET}"
    TRAINED_MODELS_ROOT = f"../trained_models/{DATASET}"
    OUTPUT_FILES_NAME = MODEL_NAME
    
    # make dirs
    for dir in [LOGS_ROOT, RESULTS_ROOT, TRAINED_MODELS_ROOT]:
        if not os.path.exists(dir):
            os.makedirs(dir)