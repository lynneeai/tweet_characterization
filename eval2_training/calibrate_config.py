import torch

class CALIBRATE_CONFIG:
    
    # dataset and model
    MODEL_NAME = "clip_multi"
    LABEL_NAMES = ["BENIGN", "MALICIOUS"]
    
    # configs
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # original model
    ORIGINAL_MODEL_STATES_FILE = "./trained_models/covid19+climate/clip_multi.weights.best"
    
    # output paths
    TSV_ROOT = "./processed_data/covid19+climate"
    LOGS_ROOT = "./logs/covid19+climate_calibrated"
    RESULTS_ROOT = "./results/covid19+climate_calibrated"
    TRAINED_MODELS_ROOT = "./trained_models/covid19+climate_calibrated"
    OUTPUT_FILES_NAME = MODEL_NAME