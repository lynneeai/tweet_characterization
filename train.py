import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from tqdm import trange
from transformers import AdamW

from config import TRAIN_CONFIG
from utils import init_logger
from utils import program_sleep
from models import CLIP_MODEL

"""Make directories"""
if not os.path.exists(TRAIN_CONFIG.LOGS_ROOT):
    os.makedirs(TRAIN_CONFIG.LOGS_ROOT)
if not os.path.exists(TRAIN_CONFIG.RESULTS_ROOT):
    os.makedirs(TRAIN_CONFIG.RESULTS_ROOT)
if not os.path.exists(TRAIN_CONFIG.MODEL_STATES_ROOT):
    os.makedirs(TRAIN_CONFIG.MODEL_STATES_ROOT)
"""------------------"""

"""Init logger"""
log_filename = os.path.basename(__file__).split(".")[0]
init_logger(TRAIN_CONFIG.LOGS_ROOT, TRAIN_CONFIG.OUTPUT_FILES_NAME)
LOGGER = logging.getLogger(log_filename)
"""------------------"""

from load_data import load_dataloaders


class TRAINER(object):
    def __init__(self, model, batch_size, epochs, initial_lr, weight_decay, patience, device, model_states_file, results_file, label_names=TRAIN_CONFIG.LABEL_NAMES):
        super(TRAINER, self).__init__()
        
        self.model = model
        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.model_states_file = model_states_file
        self.results_file = results_file
        self.label_names = label_names
        
        self.train_dataloader, self.validate_dataloader, self.test_dataloader = load_dataloaders(batch_size)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=initial_lr,
            weight_decay=weight_decay
        )
        self.best_validate_acc = 0.0
        self.curr_patience = 0
        self.curr_epoch = 0
        
    def pass_data_iteratively(self, dataloader):
        self.model.eval()
        outputs = []
        labels = []
        
        pbar = trange(len(dataloader), leave=True)
        for batch_samples in dataloader:
            batch_image_files, batch_texts = batch_samples["image_file"], batch_samples["text"]
            batch_labels = batch_samples["label"].to(self.device)
            
            batch_outputs = self.model(batch_image_files, batch_texts)
            
            outputs.append(batch_outputs.detach())
            labels.append(batch_labels)
            pbar.update(1)
        pbar.close()
        
        return torch.cat(outputs, dim=0), torch.cat(labels, dim=0)
    
    def adjust_learning_rate(self, decay_rate=0.5):
        now_lr = 0
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * decay_rate
            now_lr = param_group["lr"]
        LOGGER.info(f"Adjusted learning rate. Current lr: {now_lr}")
        
    def train_single_epoch(self):
        LOGGER.info(f"-------Start training epoch {self.curr_epoch}/{self.epochs}!-------")
        
        start_time = time.time()
        self.model.train()
        
        accumulate_loss = 0
        accumulate_acc = 0
        
        pbar = trange(len(self.train_dataloader), desc="Loss: ; Acc: ", leave=True)
        for batch_samples in self.train_dataloader:
            batch_image_files, batch_texts = batch_samples["image_file"], batch_samples["text"]
            batch_labels = batch_samples["label"].to(self.device)
            
            batch_outputs = self.model(batch_image_files, batch_texts)
            batch_loss = F.nll_loss(batch_outputs, batch_labels)
            
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            
            batch_loss_np = batch_loss.detach().cpu().numpy()
            accumulate_loss += batch_loss_np
            
            batch_corrects = (torch.max(batch_outputs, 1)[1].view(batch_labels.shape[0]).data == batch_labels.data).sum()
            batch_acc = batch_corrects.detach().cpu().numpy() / float(batch_labels.shape[0])
            accumulate_acc += batch_acc
            
            pbar.set_description(f"Loss: {batch_loss_np} ; Acc: {batch_acc}")
            pbar.refresh()
            pbar.update(1)
            
        pbar.close()
        epoch_loss = accumulate_loss / len(self.train_dataloader)
        epoch_acc = accumulate_acc / len(self.train_dataloader)
        LOGGER.info(f"Finished epoch {self.curr_epoch}/{self.epochs}! Epoch Loss: {np.round(epoch_loss, 5)}; Epoch Acc: {np.round(epoch_acc, 5)}; Time Elapsed: {np.round(time.time() - start_time, 5)}")
        
        # evaluate model using validate set to get best_validate_acc and curr_patience
        self.evaluate()
        self.model.train()
        
    def train(self):
        LOGGER.info("=======Training model...=======")
        
        start_time = time.time()
        for epoch in range(1, self.epochs + 1):
            self.curr_epoch = epoch
            self.train_single_epoch()
            if self.curr_patience >= self.patience:
                LOGGER.info(f"Epoch {self.curr_epoch} and patience {self.curr_patience}. Reloading the best model...")
                self.model.load_state_dict(torch.load(self.model_states_file), map_location=self.device)
                self.adjust_learning_rate()
                self.curr_patience = 0
        LOGGER.info(f"=======Finished training {self.epochs} epochs! Time elapsed: {np.round(time.time() - start_time, 5)}=======")
        
        
    def evaluate(self):
        LOGGER.info("=======Evaluating model...=======")
        
        outputs, labels = self.pass_data_iteratively(self.validate_dataloader)
        predicts = torch.max(outputs, 1)[1]
        predicts = predicts.data.cpu().numpy().tolist()
        labels = labels.data.cpu().numpy().tolist()
        
        acc = accuracy_score(labels, predicts)
        LOGGER.info(f"Current validate set acc: {np.round(acc, 5)}. Previous best validate set acc: {np.round(self.best_validate_acc, 5)}")
        
        if acc > self.best_validate_acc:
            self.best_validate_acc = acc
            torch.save(self.model.state_dict(), self.model_states_file)
            LOGGER.info(classification_report(labels, predicts, target_names=self.label_names, digits=5))
            LOGGER.info("Best model saved!")
        else:
            self.curr_patience += 1
            
    def test(self):
        LOGGER.info("=======Testing model...=======")
        
        self.model.load_state_dict(torch.load(self.model_states_file), map_location=self.device)
        outputs, labels = self.pass_data_iteratively(self.test_dataloader)
        predicts = torch.max(outputs, 1)[1]
        predicts = predicts.data.cpu().numpy().tolist()
        labels = labels.data.cpu().numpy().tolist()
        
        class_report_str = classification_report(labels, predicts, target_names=self.label_names, digits=5)
        class_report_dict = classification_report(labels, predicts, target_names=self.label_names, digits=5, output_dict=True)
        LOGGER.info("============================")
        LOGGER.info(class_report_str)
        
        # write to results file
        with open(self.results_file, "r") as outfile:
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
    
    
if __name__ == "__main__":
    model_states_file = f"{TRAIN_CONFIG.MODEL_STATES_ROOT}/{TRAIN_CONFIG.OUTPUT_FILES_NAME}.weights.best"
    results_file = f"{TRAIN_CONFIG.RESULTS_ROOT}/{TRAIN_CONFIG.OUTPUT_FILES_NAME}_class_report.txt"
    
    model = CLIP_MODEL().to(TRAIN_CONFIG.DEVICE)
    
    trainer = TRAINER(
        model=model,
        batch_size=TRAIN_CONFIG.BATCH_SIZE,
        epochs=TRAIN_CONFIG.EPOCHS,
        initial_lr=TRAIN_CONFIG.INITIAL_LR,
        weight_decay=TRAIN_CONFIG.WEIGHT_DECAY,
        patience=TRAIN_CONFIG.PATIENCE,
        device=TRAIN_CONFIG.DEVICE,
        model_states_file=model_states_file,
        results_file=results_file
    )
    
    if TRAIN_CONFIG.TRAIN:
        trainer.train()
    if TRAIN_CONFIG.TEST:
        trainer.test()