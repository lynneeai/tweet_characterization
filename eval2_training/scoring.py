import csv
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

from models import CLIP_MULTI_MODEL, ModelWithTemperature
from load_data import load_dataloaders
from calibrate_config import CALIBRATE_CONFIG


INTENT_TO_INTENTLABEL = {
    "POLAR": "To polarize", 
    "CALL_TO_ACTION": "To call to action", 
    "VIRAL": "To become viral", 
    "SARCASM": "To amuse", 
    "HUMOR": "To amuse"
}


def softmax(x, axis):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / np.expand_dims(np.sum(e_x, axis=axis), axis=axis)


def LLR(scores):
    pc0x, pc1x = scores[0], scores[1]
    log_classifier_ratio = np.log10(pc1x / pc0x)
    
    c0 = 1750 # BENIGN
    c1 = 1624 # MALICIOUS
    pc0 = c0 / (c0 + c1)
    pc1 = c1 / (c0 + c1)
    log_prior_ratio = np.log10(pc0 / pc1)
    
    llr = log_classifier_ratio + log_prior_ratio
    
    return {"logLikelihoodRatio": llr, "priorsLogLikelihoodRatio": log_prior_ratio, "probManipulation": pc1x}


def predict_label(llr, threshold):
    return 0 if llr < threshold else 1


def inference(model, image_file, text, threshold=0):
    model.eval()
    output, _, intent_outputs_dict = model([image_file], [text])
    output = output.data.cpu().numpy()
    
    scores = softmax(output, axis=1)
    llr_dict = LLR(scores[0])
    predict = predict_label(llr_dict["logLikelihoodRatio"], threshold)
    
    intents = []
    for intent in INTENT_TO_INTENTLABEL:
        intent_predict = torch.max(intent_outputs_dict[intent], 1)[1]
        intent_predict = intent_predict.data.cpu().numpy().tolist()[0]
        if intent_predict == 1:
            intents.append(INTENT_TO_INTENTLABEL[intent])
    
    return llr_dict, predict, intents


if __name__ == "__main__":
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    CLIP_MULTI_MODEL = CLIP_MULTI_MODEL(device=device, output_size=2).to(device)
    intent_model = ModelWithTemperature(CLIP_MULTI_MODEL, device=device).to(device)
    intent_model.load_state_dict(torch.load(
        "./trained_models/covid19+climate_calibrated/clip_multi.weights.best", 
        map_location=device
    ))
    
    labels, preds = [], []
    tids, intents_list = [], []
    with open("./processed_data/eval2/tweets.tsv", "r") as infile:
        tsv_reader = csv.DictReader(infile, delimiter="\t")
        pbar = tqdm(total=60)
        for row in tsv_reader:
            labels.append(int(row["label"]))
            _, predict, intents = inference(intent_model, f"../{row['image_file']}", row["text"])
            preds.append(predict)
            tids.append(row["tid"])
            intents_list.append(intents)
            pbar.update(1)
        pbar.close()
    
    if not os.path.exists("./results/eval2"):
        os.makedirs("./results/eval2")
    class_report_str = classification_report(labels, preds, target_names=["BENIGN", "MALICIOUS"], digits=5)
    with open("./results/eval2/clip_class_report.txt", "w") as outfile:
        outfile.write(class_report_str)
    with open("./results/eval2/clip_preds.tsv", "w") as outfile:
        tsv_writer = csv.DictWriter(outfile, fieldnames=["tid", "label", "pred", "intents"], delimiter="\t")
        tsv_writer.writeheader()
        for idx, tid in enumerate(tids):
            tsv_writer.writerow({
                "tid": tid,
                "label": labels[idx],
                "pred": preds[idx],
                "intents": intents_list[idx]
            })