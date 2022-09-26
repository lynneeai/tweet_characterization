import logging
import os
import sys

import torch
from tqdm import trange
from sklearn.metrics import classification_report

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

from util_scripts.utils import init_logger
from models import ModelWithTemperature


def calibrate_multimodal_model(origin_model, config, dataloaders, calib_model_states_file, calib_model_file, calib_results_file, device, include_intent_categories=True):
    log_filename = os.path.basename(__file__).split(".")[0]
    init_logger(config.LOGS_ROOT, config.OUTPUT_FILES_NAME)
    LOGGER = logging.getLogger(log_filename)
    
    validate_dataloader, test_dataloader = dataloaders
    
    """model calibration"""
    LOGGER.info("Calibrading model...")
    calib_model = ModelWithTemperature(origin_model, device, include_intent_categories).to(device)
    calib_model.set_temperature(validate_dataloader)
    LOGGER.info("Saving calibrated model...")
    torch.save(calib_model.state_dict(), calib_model_states_file)
    torch.save(calib_model, calib_model_file)
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
            
            batch_outputs = model(batch_image_files, batch_texts)[0]
            
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
        
        class_report_str = classification_report(labels, predicts, target_names=config.LABEL_NAMES, digits=5)
        class_report_dict = classification_report(labels, predicts, target_names=config.LABEL_NAMES, digits=5, output_dict=True)
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
    test_results_dict = test(calib_model, test_dataloader, calib_results_file)
    
    return calib_model, test_results_dict