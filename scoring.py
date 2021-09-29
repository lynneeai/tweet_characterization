import json
import torch
import numpy as np


def softmax(x, axis):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / np.expand_dims(np.sum(e_x, axis=axis), axis=axis)


def LLR(scores):
    pc0x, pc1x = scores[0], scores[1]
    log_classifier_ratio = np.log10(pc1x / pc0x)
    
    c0 = 2857 # BENIGN
    c1 = 2732 # MALICIOUS
    pc0 = c0 / (c0 + c1)
    pc1 = c1 / (c0 + c1)
    log_prior_ratio = np.log10(pc0 / pc1)
    
    llr = log_classifier_ratio + log_prior_ratio
    
    return {"logLikelihoodRatio": llr, "priorsLogLikelihoodRatio": log_prior_ratio, "probManipulation": pc1x}


def predict_label(llr, threshold):
    return 0 if llr < threshold else 1


def inference(model, image_file, text, threshold):
    model.eval()
    output = model([image_file], [text[:77]])[0].data.cpu().numpy()
    scores = softmax(output, axis=1)
    llr_dict = LLR(scores[0])
    predict = predict_label(llr_dict["logLikelihoodRatio"], threshold)
    
    with open("LLR.jsonl", "w") as outfile:
        json.dump(str(llr_dict), outfile)
        outfile.write("\n")
    
    return llr_dict, predict


def read_AOM(json_file):
    with open(json_file, "r") as infile:
        aom = json.load(infile)
        image_file = aom["image_file"]
        text = aom["text"]
    return image_file, text


if __name__ == "__main__":
    
    model = torch.load("./trained_models/fakeddit_calibrated/clip.pth")
    
    image_file, text = read_AOM("test.json")
    llr_dict, label = inference(model, image_file, text, 0)
    print(llr_dict)
    print(label)
    