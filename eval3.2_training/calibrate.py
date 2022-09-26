import os
import sys
import logging

import torch

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

from image_text_dataset import ImageTextDataset
from models import CLIP_MODEL
from calibrate_config import CALIBRATE_CONFIG
from util_scripts.utils import init_logger

"""Init logger"""
log_filename = os.path.basename(__file__).split(".")[0]
init_logger(CALIBRATE_CONFIG.LOGS_ROOT, CALIBRATE_CONFIG.OUTPUT_FILES_NAME)
LOGGER = logging.getLogger(log_filename)
"""------------------"""

from trainers import load_dataloaders, calibrate_multimodal_model

calibrated_model_states_file = f"{CALIBRATE_CONFIG.TRAINED_MODELS_ROOT}/{CALIBRATE_CONFIG.OUTPUT_FILES_NAME}.weights.best"
calibrated_model_file = f"{CALIBRATE_CONFIG.TRAINED_MODELS_ROOT}/{CALIBRATE_CONFIG.OUTPUT_FILES_NAME}.pth"
calibrated_results_file = f"{CALIBRATE_CONFIG.RESULTS_ROOT}/{CALIBRATE_CONFIG.OUTPUT_FILES_NAME}_class_report.txt"

"""load original model"""
device = CALIBRATE_CONFIG.DEVICE
original_model_states_file = CALIBRATE_CONFIG.ORIGINAL_MODEL_STATES_FILE
LOGGER.info(f"Load original model from {original_model_states_file}...")
original_model = CLIP_MODEL(device, output_size=2).to(device)
original_model.load_state_dict(torch.load(original_model_states_file, map_location=device))

"""load data"""
LOGGER.info("Loading dataloaders...")
_, validate_dataloader, test_dataloader = load_dataloaders(CALIBRATE_CONFIG, ImageTextDataset)

"""calibrate model"""
calibrated_model, test_results_dict = calibrate_multimodal_model(
    origin_model=original_model,
    config=CALIBRATE_CONFIG,
    dataloaders=[validate_dataloader, test_dataloader],
    calib_model_states_file=calibrated_model_states_file,
    calib_model_file=calibrated_model_file,
    calib_results_file=calibrated_results_file,
    device=device,
    include_intent_categories=False
)