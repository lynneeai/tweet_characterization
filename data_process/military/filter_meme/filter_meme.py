import argparse
import csv
import torch
from torch.utils.data import DataLoader
from model import MemeDetector
from tqdm import trange


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_file, image_folder):
        super(ImageDataset, self).__init__()
        
        self.tsv_file = tsv_file
        self.image_folder = image_folder
        
        self.tid_image_pairs = []
        with open(tsv_file, "r") as infile:
            tsv_reader = csv.DictReader(infile, delimiter="\t")
            for row in tsv_reader:
                tid, image_files = row["tid"], row["image_files"]
                image_files = image_files.split(",")
                for image_file in image_files:
                    self.tid_image_pairs.append((tid, image_file))
                    
    def __len__(self):
        return len(self.tid_image_pairs)
    
    def __getitem__(self, idx):
        return {
            "tid": self.tid_image_pairs[idx][0],
            "image_file": f"{self.image_folder}/{self.tid_image_pairs[idx][1]}"
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tsv_file", required=True)
    parser.add_argument("-i", "--image_folder", required=True)
    parser.add_argument("-o", "--output_tsv", required=True)
    args = parser.parse_args()
    
    test_dataset = ImageDataset(args.tsv_file, args.image_folder)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = MemeDetector(device=device).to(device)
    model.load_state_dict(torch.load("./resnet50d.weights.best", map_location=device))
    model.eval()

    tids = []
    outputs = []
    pbar = trange(len(test_dataloader), leave=True)
    for batch_samples in test_dataloader:
        batch_tids, batch_image_files = batch_samples["tid"], batch_samples["image_file"]
        batch_outputs = model(batch_image_files)
        
        tids.extend(batch_tids)
        outputs.append(batch_outputs.detach())
        
        pbar.update(1)
    pbar.close()
    outputs = torch.cat(outputs, dim=0)
    
    predicts = torch.max(outputs, 1)[1]
    predicts = predicts.data.cpu().numpy().tolist()
    assert(len(tids) == len(predicts))
    
    with open(args.output_tsv, "w") as outfile:
        tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "meme_label"])
        tsv_writer.writeheader()
        for i in range(len(tids)):
            tsv_writer.writerow({
                "tid": tids[i],
                "meme_label": predicts[i]
            })