import torch

class TRAIN_CONFIG:
    
    # dataset and model
    DATASET = "military_vehicles"
    MODEL_NAME = "clip"
    # LABEL_NAMES = ["TRUE", "SATIRE/PARODY", "FALSE_CONNECTION", "IMPOSTER_CONTENT", "MANIPULATED_CONTENT", "MISLEADING_CONTENT"]
    LABEL_NAMES = ["BENIGN", "MALICIOUS"]
    
    # train configs
    BATCH_SIZE = 32
    EPOCHS = 30
    INITIAL_LR = 2e-5
    WEIGHT_DECAY = 0.001
    PATIENCE = 3
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    TRAIN = True
    TEST = True
    
    # pretrained
    USE_PRETRAINED = True
    PRETRAINED_MODEL_STATES_FILE = "../trained_models/fakeddit_covid19/clip.weights.best"
    
    # output paths
    TSV_ROOT = f"../processed_data/fakeddit_{DATASET}"
    IMAGE_ROOT = "../datasets/fakeddit/selected_images"
    LOGS_ROOT = f"../logs/fakeddit_{DATASET}"
    RESULTS_ROOT = f"../results/fakeddit_{DATASET}"
    TRAINED_MODELS_ROOT = f"../trained_models/fakeddit_{DATASET}"
    OUTPUT_FILES_NAME = MODEL_NAME
    