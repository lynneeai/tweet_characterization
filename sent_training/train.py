import logging
import os
import sys

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

from config import TRAIN_CONFIG
from util_scripts.utils import init_logger

"""Init logger"""
if not os.path.exists(TRAIN_CONFIG.LOGS_ROOT):
    os.makedirs(TRAIN_CONFIG.LOGS_ROOT)
log_filename = os.path.basename(__file__).split(".")[0]
init_logger(TRAIN_CONFIG.LOGS_ROOT, TRAIN_CONFIG.OUTPUT_FILES_NAME)
LOGGER = logging.getLogger(log_filename)
"""------------------"""

from trainers import create_partitions, load_dataloaders, MULTIMODAL_TRAINER
from models.clip_sent import CLIP_SENT


if TRAIN_CONFIG.RECREATE_PARTITIONS:
    LOGGER.info("Recreating train/test/validate partitions...")
    create_partitions(TRAIN_CONFIG)
train_dataloader, validate_dataloader, test_dataloader = load_dataloaders(TRAIN_CONFIG)

model_states_file = f"{TRAIN_CONFIG.TRAINED_MODELS_ROOT}/{TRAIN_CONFIG.OUTPUT_FILES_NAME}.weights.best"
model_file = f"{TRAIN_CONFIG.TRAINED_MODELS_ROOT}/{TRAIN_CONFIG.OUTPUT_FILES_NAME}.pth"
results_file = f"{TRAIN_CONFIG.RESULTS_ROOT}/{TRAIN_CONFIG.OUTPUT_FILES_NAME}_class_report.txt"

model = CLIP_SENT(device=TRAIN_CONFIG.DEVICE).to(TRAIN_CONFIG.DEVICE)

trainer = MULTIMODAL_TRAINER(
    model=model,
    config=TRAIN_CONFIG,
    dataloaders=[train_dataloader, validate_dataloader, test_dataloader],
    model_states_file=model_states_file,
    model_file=model_file,
    results_file=results_file,
    include_intents=False
)

if TRAIN_CONFIG.TRAIN:
    trainer.train()
if TRAIN_CONFIG.TEST:
    trainer.test()