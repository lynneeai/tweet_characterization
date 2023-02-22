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

from util_scripts.utils import init_logger


class MULTIMODAL_TRAINER(object):
    def __init__(self, model, config, dataloaders, model_states_file, model_file, results_file, include_intents=True):
        super(MULTIMODAL_TRAINER, self).__init__()
        
        log_filename = os.path.basename(__file__).split(".")[0]
        init_logger(config.LOGS_ROOT, config.OUTPUT_FILES_NAME)
        self.logger = logging.getLogger(log_filename)
        
        self.model = model
        self.epochs = config.EPOCHS
        self.patience = config.PATIENCE
        self.device = config.DEVICE
        self.label_names = config.LABEL_NAMES
        self.intent_names = config.INTENT_NAMES if include_intents else None
        self.model_states_file = model_states_file
        self.model_file = model_file
        self.results_file = results_file
        self.include_intents = include_intents

        self.train_dataloader, self.validate_dataloader, self.test_dataloader = dataloaders
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.INITIAL_LR,
            weight_decay=config.WEIGHT_DECAY
        )
        self.best_validate_acc = 0.0
        self.curr_patience = 0
        self.curr_epoch = 0
        
    def pass_data_iteratively(self, dataloader):
        self.model.eval()
        outputs = []
        labels = []
        if self.include_intents:
            intent_outputs_dict = {intent:[] for intent in self.intent_names}
            intents_dict = {intent:[] for intent in self.intent_names}
        
        pbar = trange(len(dataloader), leave=True)
        for batch_samples in dataloader:
            batch_image_files, batch_texts = batch_samples["image_file"], batch_samples["text"]
            batch_labels = batch_samples["label"].to(self.device)
            
            if self.include_intents:
                batch_outputs, _, batch_intent_outputs_dict = self.model(batch_image_files, batch_texts)
            else:
<<<<<<< Updated upstream
                batch_outputs = self.model(batch_image_files, batch_texts)[0]
=======
                batch_outputs = self.model(batch_image_files, batch_texts)
>>>>>>> Stashed changes
            
            outputs.append(batch_outputs.detach())
            labels.append(batch_labels)
            
            # intents
            if self.include_intents:
                for intent in self.intent_names:
                    intent_outputs_dict[intent].append(batch_intent_outputs_dict[intent].detach())
                    batch_intents = batch_samples[intent].to(self.device)
                    intents_dict[intent].append(batch_intents)
            
            pbar.update(1)
        pbar.close()
        
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)
        
        if self.include_intents:
            for intent in self.intent_names:
                intent_outputs = intent_outputs_dict[intent]
                intent_outputs_dict[intent] = torch.cat(intent_outputs, dim=0)
                
                intents = intents_dict[intent]
                intents_dict[intent] = torch.cat(intents, dim=0)
        
            return outputs, labels, intent_outputs_dict, intents_dict
        
        return outputs, labels
    
    def adjust_learning_rate(self, decay_rate=0.5):
        now_lr = 0
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * decay_rate
            now_lr = param_group["lr"]
        self.logger.info(f"Adjusted learning rate. Current lr: {now_lr}")
        
    def train_single_epoch(self):
        self.logger.info(f"-------Start training epoch {self.curr_epoch}/{self.epochs}!-------")
        
        start_time = time.time()
        self.model.train()
        
        accumulate_loss = 0
        accumulate_acc = 0
        
        pbar = trange(len(self.train_dataloader), desc="Loss: ; Acc: ", leave=True)
        for batch_samples in self.train_dataloader:
            batch_image_files, batch_texts = batch_samples["image_file"], batch_samples["text"]
            batch_labels = batch_samples["label"].to(self.device)
            
            # intents
            if self.include_intents:
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
            else:
<<<<<<< Updated upstream
                batch_outputs = self.model(batch_image_files, batch_texts)[0]
=======
                batch_outputs = self.model(batch_image_files, batch_texts)
>>>>>>> Stashed changes
                batch_loss = self.model.loss(outputs=batch_outputs, labels=batch_labels)
            
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
        self.logger.info(f"Finished epoch {self.curr_epoch}/{self.epochs}! Epoch Loss: {np.round(epoch_loss, 5)}; Epoch Acc: {np.round(epoch_acc, 5)}; Time Elapsed: {np.round(time.time() - start_time, 5)}")
        
        # evaluate model using validate set to get best_validate_acc and curr_patience
        self.evaluate()
        self.model.train()
        
    def train(self):
        self.logger.info("=======Training model...=======")
        
        start_time = time.time()
        for epoch in range(1, self.epochs + 1):
            self.curr_epoch = epoch
            self.train_single_epoch()
            if self.curr_patience >= self.patience:
                self.logger.info(f"Epoch {self.curr_epoch} and patience {self.curr_patience}. Reloading the best model...")
                self.model.load_state_dict(torch.load(self.model_states_file, map_location=self.device))
                self.adjust_learning_rate()
                self.curr_patience = 0
        self.logger.info(f"=======Finished training {self.epochs} epochs! Time elapsed: {np.round(time.time() - start_time, 5)}=======")
        
        
    def evaluate(self):
        self.logger.info("=======Evaluating model...=======")
        
        if self.include_intents:
            outputs, labels, _, _ = self.pass_data_iteratively(self.validate_dataloader)
        else:
            outputs, labels = self.pass_data_iteratively(self.validate_dataloader)
        predicts = torch.max(outputs, 1)[1]
        predicts = predicts.data.cpu().numpy().tolist()
        labels = labels.data.cpu().numpy().tolist()
        
        acc = accuracy_score(labels, predicts)
        self.logger.info(f"Current validate set acc: {np.round(acc, 5)}. Previous best validate set acc: {np.round(self.best_validate_acc, 5)}")
        
        if acc > self.best_validate_acc:
            self.best_validate_acc = acc
            torch.save(self.model.state_dict(), self.model_states_file)
            torch.save(self.model, self.model_file)
            self.logger.info(classification_report(labels, predicts, target_names=self.label_names, digits=5))
            self.logger.info("Best model saved!")
        else:
            self.curr_patience += 1
            
    def test(self):
        self.logger.info("=======Testing model...=======")
        
        self.model.load_state_dict(torch.load(self.model_states_file, map_location=self.device))
        if self.include_intents:
            outputs, labels, intent_outputs_dict, intents_dict = self.pass_data_iteratively(self.test_dataloader)
        else:
            outputs, labels = self.pass_data_iteratively(self.test_dataloader)
        predicts = torch.max(outputs, 1)[1]
        predicts = predicts.data.cpu().numpy().tolist()
        labels = labels.data.cpu().numpy().tolist()
        
        # outfile
        outfile = open(self.results_file, "w")
        
        class_report_str = classification_report(labels, predicts, target_names=self.label_names, digits=5)
        class_report_dict = classification_report(labels, predicts, target_names=self.label_names, digits=5, output_dict=True)
<<<<<<< Updated upstream
        self.logger.info("==============CLASS REPORT==============")
=======
        self.logger.info("==============MALICIOUSNESS CLASS REPORT==============")
>>>>>>> Stashed changes
        self.logger.info(class_report_str)
        outfile.write(f"{class_report_str}\n")
        
        # intents
        if self.include_intents:
            for intent in self.intent_names:
                intent_predicts = torch.max(intent_outputs_dict[intent], 1)[1]
                intent_predicts = intent_predicts.data.cpu().numpy().tolist()
                intent_labels = intents_dict[intent].data.cpu().numpy().tolist()
                try:
                    intent_cr_str = classification_report(intent_labels, intent_predicts, target_names=[f"NON-{intent}", intent], digits=5)
                    self.logger.info(f"=============={intent} CLASS REPORT==============")
                    self.logger.info(intent_cr_str)
                    outfile.write(f"{intent_cr_str}\n")
                except:
                    self.logger.info(f"=============={intent} CLASS REPORT==============")
                    self.logger.info(f"{intent} class does not have any positive samples!\n\n")
                    outfile.write(f"{intent} class does not have any positive samples!\n\n")
        
        outfile.close()
        
        test_results_dict = {
            "outputs": outputs,
            "predicts": predicts,
            "labels": labels,
            "class_report_dict": class_report_dict,
            "class_report_str": class_report_str
        }
        self.logger.info("=======Finished testing!=======")
        return test_results_dict