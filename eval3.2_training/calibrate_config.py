import torch

class CALIBRATE_CONFIG:
    
    # dataset and model
    DATASET = "eval3.2_de"
    CALIBRATE_DATASET = "military_de"
    MODEL_NAME = "clip_calib"
    LABEL_NAMES = ["NON_DE", "DE"]
    
    # configs
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # original model
    ORIGINAL_MODEL_STATES_FILE = f"../trained_models/{DATASET}/clip.weights.best"
    
    # output paths
    TSV_ROOT = f"../processed_data/{CALIBRATE_DATASET}"
    LOGS_ROOT = f"../logs/{DATASET}"
    RESULTS_ROOT = f"../results/{DATASET}"
    TRAINED_MODELS_ROOT = f"../trained_models/{DATASET}"
    OUTPUT_FILES_NAME = MODEL_NAME