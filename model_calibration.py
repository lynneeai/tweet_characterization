import csv
import logging
import json
import os
import sys

import torch
import numpy as np
from tqdm import trange
from sklearn.metrics import classification_report

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

from utils import init_logger
from load_data import load_dataloaders
from models import CLIP_MODEL, ModelWithTemperature
from calibrate_config import CALIBRATE_CONFIG

"""Make directories"""
if not os.path.exists(CALIBRATE_CONFIG.LOGS_ROOT):
    os.makedirs(CALIBRATE_CONFIG.LOGS_ROOT)
if not os.path.exists(CALIBRATE_CONFIG.RESULTS_ROOT):
    os.makedirs(CALIBRATE_CONFIG.RESULTS_ROOT)
if not os.path.exists(CALIBRATE_CONFIG.TRAINED_MODELS_ROOT):
    os.makedirs(CALIBRATE_CONFIG.TRAINED_MODELS_ROOT)
"""------------------"""

"""Init logger"""
log_filename = os.path.basename(__file__).split(".")[0]
init_logger(CALIBRATE_CONFIG.LOGS_ROOT, CALIBRATE_CONFIG.OUTPUT_FILES_NAME)
LOGGER = logging.getLogger(log_filename)
"""------------------"""


def calibrate_model(original_model_states_file, calibrated_model_states_file, calibrated_model_file, calibrated_results_file, device):
    """load original model"""
    LOGGER.info(f"Load original model from {original_model_states_file}...")
    original_model = CLIP_MODEL().to(device)
    original_model.load_state_dict(torch.load(original_model_states_file, map_location=device))
    
    """load data"""
    LOGGER.info("Loading dataloaders...")
    LOGGER.info("Loading merged validation/test sets...")
    _, validate_dataloader, test_dataloader = load_dataloaders(CALIBRATE_CONFIG.BATCH_SIZE, tsv_root=CALIBRATE_CONFIG.TSV_ROOT, image_dir=CALIBRATE_CONFIG.IMAGE_ROOT)
    dataset_testloader = {}
    for dataset in ["covid19", "climate_change", "military_vehicles"]:
        LOGGER.info(f"Loading test set for {dataset}...")
        _, _, dataset_test_dataloader = load_dataloaders(1, f"./processed_data/fakeddit_{dataset}", image_dir=CALIBRATE_CONFIG.IMAGE_ROOT)
        dataset_testloader[dataset] = dataset_test_dataloader
    
    """model calibration"""    
    LOGGER.info("Calibrating model...")
    calibrated_model = ModelWithTemperature(original_model, device).to(device)
    calibrated_model.set_temperature(validate_dataloader)
    LOGGER.info("Saving calibrated model...")
    torch.save(calibrated_model.state_dict(), calibrated_model_states_file)
    torch.save(calibrated_model, calibrated_model_file)
    LOGGER.info("=======Finished calibrating!=======")
    
    """test calibrated model"""
    def pass_data_iteratively(model, dataloader):
        model.eval()
        outputs = []
        labels = []
        
        pbar = trange(len(dataloader), leave=True)
        for batch_samples in dataloader:
            batch_image_files, batch_texts = batch_samples["image_file"], batch_samples["text"]
            batch_labels = batch_samples["label"].to(device)
            
            batch_outputs, _, _ = model(batch_image_files, batch_texts)
            
            outputs.append(batch_outputs.detach())
            labels.append(batch_labels)
            
            pbar.update(1)
        pbar.close()
        
        return torch.cat(outputs, dim=0), torch.cat(labels, dim=0)
    
    def test(model, dataloader, results_file):
        LOGGER.info("=======Testing model...=======")
        
        outputs, labels = pass_data_iteratively(model, dataloader)
        predicts = torch.max(outputs, 1)[1]
        predicts = predicts.data.cpu().numpy().tolist()
        labels = labels.data.cpu().numpy().tolist()
        
        class_report_str = classification_report(labels, predicts, target_names=["BENIGN", "MALICIOUS"], digits=5)
        class_report_dict = classification_report(labels, predicts, target_names=["BENIGN", "MALICIOUS"], digits=5, output_dict=True)
        LOGGER.info("============================")
        LOGGER.info(class_report_str)
        
        # write to results file
        with open(results_file, "w") as outfile:
            outfile.write(f"{class_report_str}\n")
        
        test_results_dict = {
            "outputs": outputs,
            "predicts": predicts,
            "labels": labels,
            "class_report_dict": class_report_dict,
            "class_report_str": class_report_str
        }
        LOGGER.info("=======Finished testing!=======")
        return test_results_dict
    
    LOGGER.info("Testing calibrated model...")
    test_results_dict = test(calibrated_model, test_dataloader, calibrated_results_file)
    
    """test calibrated model for each dataset"""
    for dataset in ["covid19", "climate_change", "military_vehicles"]:
        LOGGER.info(f"Testing calibrated model for {dataset}...")
        dataset_results_file = f"./results/fakeddit_{dataset}/{CALIBRATE_CONFIG.OUTPUT_FILES_NAME}_cali_class_report.txt"
        test_results_dict = test(calibrated_model, dataset_testloader[dataset], dataset_results_file)
    
    return calibrated_model, test_results_dict


if __name__ == "__main__":
    calibrated_model_states_file = f"{CALIBRATE_CONFIG.TRAINED_MODELS_ROOT}/{CALIBRATE_CONFIG.OUTPUT_FILES_NAME}.weights.best"
    calibrated_model_file = f"{CALIBRATE_CONFIG.TRAINED_MODELS_ROOT}/{CALIBRATE_CONFIG.OUTPUT_FILES_NAME}.pth"
    calibrated_results_file = f"{CALIBRATE_CONFIG.RESULTS_ROOT}/{CALIBRATE_CONFIG.OUTPUT_FILES_NAME}_class_report.txt"
    
    calibrated_model, test_results_dict = calibrate_model(CALIBRATE_CONFIG.ORIGINAL_MODEL_STATES_FILE, calibrated_model_states_file, calibrated_model_file, calibrated_results_file, CALIBRATE_CONFIG.DEVICE)