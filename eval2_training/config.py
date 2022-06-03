import torch

class TRAIN_CONFIG:
    
    # dataset and model
    DATASET = "covid19+climate"
    MODEL_NAME = "clip_multi"
    LABEL_NAMES = ["BENIGN", "MALICIOUS"]
    INTENT_NAMES = ["POLAR", "CALL_TO_ACTION", "VIRAL", "SARCASM", "HUMOR"]
    
    # train configs
    BATCH_SIZE = 32
    EPOCHS = 5
    INITIAL_LR = 2e-5
    WEIGHT_DECAY = 0.001
    PATIENCE = 5
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    RECREATE_PARTITIONS = False
    TRAIN_SPLIT = 0.7
    DEV_SPLIT = 0.1
    TRAIN = True
    TEST = True
    
    # pretrained
    USE_PRETRAINED = False
    PRETRAINED_MODEL_STATES_FILE = None
    
    # output paths
    TSV_ROOT = f"../processed_data/{DATASET}"
    LOGS_ROOT = f"../logs/{DATASET}"
    RESULTS_ROOT = f"../results/{DATASET}"
    TRAINED_MODELS_ROOT = f"../trained_models/{DATASET}"
    OUTPUT_FILES_NAME = MODEL_NAME
    