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
    output = model([image_file],[text])
    output = output.data.cpu().numpy()
    
    scores = softmax(output, axis=1)
    llr_dict = LLR(scores[0])
    predict = predict_label(llr_dict["logLikelihoodRatio"], threshold)
    
    return llr_dict, predict


if __name__ == "__main__":
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.load("../trained_models/covid19+climate/clip_sent.pth", map_location=device)
    model.eval()
    
    labels, preds, tids = [], [], []
    with open("../processed_data/eval2/tweets.tsv", "r") as infile:
        tsv_reader = csv.DictReader(infile, delimiter="\t")
        pbar = tqdm(total=60)
        for row in tsv_reader:
            _, predict = inference(model, f"../{row['image_file']}", row["text"])
            labels.append(int(row["label"]))
            preds.append(predict)
            tids.append(row["tid"])
            pbar.update(1)
        pbar.close()
    
    if not os.path.exists("../results/eval2"):
        os.makedirs("../results/eval2")
    class_report_str = classification_report(labels, preds, target_names=["BENIGN", "MALICIOUS"], digits=5)
    with open("../results/eval2/clip_sent_class_report.txt", "w") as outfile:
        outfile.write(class_report_str)
