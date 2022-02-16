import logging
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from tqdm import trange
from transformers import AdamW

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

from config import TRAIN_CONFIG
from util_scripts.utils import init_logger
from util_scripts.utils import program_sleep
from models import CLIP_MODEL

"""Make directories"""
if not os.path.exists(TRAIN_CONFIG.LOGS_ROOT):
    os.makedirs(TRAIN_CONFIG.LOGS_ROOT)
if not os.path.exists(TRAIN_CONFIG.RESULTS_ROOT):
    os.makedirs(TRAIN_CONFIG.RESULTS_ROOT)
if not os.path.exists(TRAIN_CONFIG.TRAINED_MODELS_ROOT):
    os.makedirs(TRAIN_CONFIG.TRAINED_MODELS_ROOT)
"""------------------"""

"""Init logger"""
log_filename = os.path.basename(__file__).split(".")[0]
init_logger(TRAIN_CONFIG.LOGS_ROOT, TRAIN_CONFIG.OUTPUT_FILES_NAME)
LOGGER = logging.getLogger(log_filename)
"""------------------"""

from load_data import create_partitions, load_dataloaders


class TRAINER(object):
    def __init__(self, model, batch_size, epochs, initial_lr, weight_decay, patience, device, model_states_file, model_file, results_file, label_names, intent_names, recreate_partitions=False):
        super(TRAINER, self).__init__()
        
        self.model = model
        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.model_states_file = model_states_file
        self.model_file = model_file
        self.results_file = results_file
        self.label_names = label_names
        self.intent_names = intent_names
        
        if recreate_partitions:
            LOGGER.info("Recreating train/test/validate partitions...")
            create_partitions()
            
        LOGGER.info("Loading train/test/validate dataloaders...")
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
        intent_outputs_dict = {intent:[] for intent in self.intent_names}
        intents_dict = {intent:[] for intent in self.intent_names}
        
        pbar = trange(len(dataloader), leave=True)
        for batch_samples in dataloader:
            batch_image_files, batch_texts = batch_samples["image_file"], batch_samples["text"]
            batch_labels = batch_samples["label"].to(self.device)
            
            batch_outputs, _, batch_intent_outputs_dict = self.model(batch_image_files, batch_texts)
            
            outputs.append(batch_outputs.detach())
            labels.append(batch_labels)
            
            # intents
            for intent in self.intent_names:
                intent_outputs_dict[intent].append(batch_intent_outputs_dict[intent].detach())
                batch_intents = batch_samples[intent].to(self.device)
                intents_dict[intent].append(batch_intents)
            
            pbar.update(1)
        pbar.close()
        
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)
        
        for intent in self.intent_names:
            intent_outputs = intent_outputs_dict[intent]
            intent_outputs_dict[intent] = torch.cat(intent_outputs, dim=0)
            
            intents = intents_dict[intent]
            intents_dict[intent] = torch.cat(intents, dim=0)
        
        return outputs, labels, intent_outputs_dict, intents_dict
    
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
            
            # intents
            batch_intents_dict = {}
            for intent in self.intent_names:
                batch_intents_dict[intent] = batch_samples[intent].to(self.device)
            
            batch_outputs, _, batch_intent_outputs_dict = self.model(batch_image_files, batch_texts)
            batch_loss = self.model.loss(
                outputs=batch_outputs,
                labels=batch_labels,
                intent_outputs_dict=batch_intent_outputs_dict,
                intents_dict=batch_intents_dict,
                intent_names=self.intent_names
            )
            
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
                self.model.load_state_dict(torch.load(self.model_states_file, map_location=self.device))
                self.adjust_learning_rate()
                self.curr_patience = 0
        LOGGER.info(f"=======Finished training {self.epochs} epochs! Time elapsed: {np.round(time.time() - start_time, 5)}=======")
        
        
    def evaluate(self):
        LOGGER.info("=======Evaluating model...=======")
        
        outputs, labels, _, _ = self.pass_data_iteratively(self.validate_dataloader)
        predicts = torch.max(outputs, 1)[1]
        predicts = predicts.data.cpu().numpy().tolist()
        labels = labels.data.cpu().numpy().tolist()
        
        acc = accuracy_score(labels, predicts)
        LOGGER.info(f"Current validate set acc: {np.round(acc, 5)}. Previous best validate set acc: {np.round(self.best_validate_acc, 5)}")
        
        if acc > self.best_validate_acc:
            self.best_validate_acc = acc
            torch.save(self.model.state_dict(), self.model_states_file)
            torch.save(self.model, self.model_file)
            LOGGER.info(classification_report(labels, predicts, target_names=self.label_names, digits=5))
            LOGGER.info("Best model saved!")
        else:
            self.curr_patience += 1
            
    def test(self):
        LOGGER.info("=======Testing model...=======")
        
        self.model.load_state_dict(torch.load(self.model_states_file, map_location=self.device))
        outputs, labels, intent_outputs_dict, intents_dict = self.pass_data_iteratively(self.test_dataloader)
        predicts = torch.max(outputs, 1)[1]
        predicts = predicts.data.cpu().numpy().tolist()
        labels = labels.data.cpu().numpy().tolist()
        
        # outfile
        outfile = open(self.results_file, "w")
        
        class_report_str = classification_report(labels, predicts, target_names=self.label_names, digits=5)
        class_report_dict = classification_report(labels, predicts, target_names=self.label_names, digits=5, output_dict=True)
        LOGGER.info("==============MALICIOUSNESS CLASS REPORT==============")
        LOGGER.info(class_report_str)
        outfile.write(f"{class_report_str}\n")
        
        # intents
        for intent in self.intent_names:
            intent_predicts = torch.max(intent_outputs_dict[intent], 1)[1]
            intent_predicts = intent_predicts.data.cpu().numpy().tolist()
            intent_labels = intents_dict[intent].data.cpu().numpy().tolist()
            try:
                intent_cr_str = classification_report(intent_labels, intent_predicts, target_names=[f"NON-{intent}", intent], digits=5)
                LOGGER.info(f"=============={intent} CLASS REPORT==============")
                LOGGER.info(intent_cr_str)
                outfile.write(f"{intent_cr_str}\n")
            except:
                LOGGER.info(f"=============={intent} CLASS REPORT==============")
                LOGGER.info(f"{intent} class does not have any positive samples!\n\n")
                outfile.write(f"{intent} class does not have any positive samples!\n\n")
        
        outfile.close()
        
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
    model_states_file = f"{TRAIN_CONFIG.TRAINED_MODELS_ROOT}/{TRAIN_CONFIG.OUTPUT_FILES_NAME}.weights.best"
    model_file = f"{TRAIN_CONFIG.TRAINED_MODELS_ROOT}/{TRAIN_CONFIG.OUTPUT_FILES_NAME}.pth"
    results_file = f"{TRAIN_CONFIG.RESULTS_ROOT}/{TRAIN_CONFIG.OUTPUT_FILES_NAME}_class_report.txt"
    
    model = CLIP_MODEL(device=TRAIN_CONFIG.DEVICE, output_size=2).to(TRAIN_CONFIG.DEVICE)
    if TRAIN_CONFIG.USE_PRETRAINED:
        LOGGER.info(f"Loading pretrained model from {TRAIN_CONFIG.PRETRAINED_MODEL_STATES_FILE}...")
        model.load_state_dict(torch.load(TRAIN_CONFIG.PRETRAINED_MODEL_STATES_FILE, map_location=TRAIN_CONFIG.DEVICE))
    
    trainer = TRAINER(
        model=model,
        batch_size=TRAIN_CONFIG.BATCH_SIZE,
        epochs=TRAIN_CONFIG.EPOCHS,
        initial_lr=TRAIN_CONFIG.INITIAL_LR,
        weight_decay=TRAIN_CONFIG.WEIGHT_DECAY,
        patience=TRAIN_CONFIG.PATIENCE,
        device=TRAIN_CONFIG.DEVICE,
        model_states_file=model_states_file,
        model_file=model_file,
        results_file=results_file,
        label_names=TRAIN_CONFIG.LABEL_NAMES,
        intent_names=TRAIN_CONFIG.INTENT_NAMES,
        recreate_partitions=TRAIN_CONFIG.RECREATE_PARTITIONS
    )
    
    if TRAIN_CONFIG.TRAIN:
        trainer.train()
    if TRAIN_CONFIG.TEST:
        trainer.test()