import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import wandb
wandb.login(key="e09f73bb0df882dd4606253c95e1bc68801828a0")
from collections import Counter
from pyod.models.pca import PCA
from sklearn import metrics
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchmetrics.functional import pairwise_euclidean_distance
from numpy.random import choice

from classifier_models import PreActResNet18, VGG
from defense_dataloader import get_dataset
from networks.models import Generator, NetC_MNIST

# Environment configuration
os.environ['WANDB_NOTEBOOK_NAME'] = 'TED.ipynb'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize argparse Namespace
opt = argparse.Namespace()
opt.dataset = "cifar10"
opt.device = "cuda" if torch.cuda.is_available() else "cpu"
opt.batch_size = 100
opt.data_root = "../data/"
opt.target = 0
opt.attack_mode = "SSDT"

print("torch.cuda.is_available():", torch.cuda.is_available())
print(f"Running on device: {opt.device}")

# Initialize Weights and Biases
wandb.init(project="TED", name=f"{opt.dataset}_k_{opt.target}", config=vars(opt))

def load_model():
    if opt.dataset == "mnist":
        return NetC_MNIST().to(opt.device)
    elif opt.dataset == "cifar10":
        return PreActResNet18().to(opt.device)
    elif opt.dataset == "gtsrb":
        return PreActResNet18(num_classes=43).to(opt.device)
    elif opt.dataset == "imagenet":
        return VGG('VGG16').to(opt.device)
    elif opt.dataset == "pubfig":
        return VGG('VGG16-pubfig').to(opt.device)
    else:
        raise ValueError(f"Unknown dataset: {opt.dataset}")

def load_state(model, state_dict):
    model.load_state_dict(state_dict)
    model.to(opt.device)
    model.eval()
    model.requires_grad_(False)
    return model

def load_model_state():
    base_path = './checkpoints/'
    model_path = f"{base_path}{opt.dataset}/SSDT/target_{opt.target}/SSDT_{opt.dataset}_ckpt.pth.tar"
    return torch.load(model_path, map_location=opt.device)

# Set input dimensions based on dataset
if opt.dataset in ["cifar10", "gtsrb"]:
    opt.input_height = 32
    opt.input_width = 32
    opt.input_channel = 3
elif opt.dataset == "mnist":
    opt.input_height = 28
    opt.input_width = 28
    opt.input_channel = 1
elif opt.dataset in ["imagenet", "pubfig"]:
    opt.input_height = 64
    opt.input_width = 64
    opt.input_channel = 3

opt.class_number = {"cifar10": 10, "gtsrb": 43, "mnist": 10, "imagenet": 100, "pubfig": 83}.get(opt.dataset, 10)
opt.defense_train_size = {"cifar10": 1000, "gtsrb": 1000, "mnist": 1000, "imagenet": (opt.class_number * 100),
                          "pubfig": (opt.class_number * 100)}.get(opt.dataset, 1000)

DEFENSE_TRAIN_SIZE = opt.defense_train_size

print("Loading model...")
model = load_model()
state_dict = load_model_state()
model = load_state(model, state_dict["netC"])
print("Model loaded and state dict applied.")

netG = Generator(opt)
netG = load_state(netG, state_dict["netG"])

netM = Generator(opt, out_channels=1)
netM = load_state(netM, state_dict["netM"])

print("Loading dataset...")
testset = get_dataset(opt, train=True)
print("Dataset loaded.")

indices = np.arange(len(testset))
benign_unknown_indices, defense_subset_indices = train_test_split(
    indices, test_size=0.1, random_state=42)

benign_unknown_subset = Subset(testset, benign_unknown_indices)
defense_subset = Subset(testset, defense_subset_indices)

benign_unknown_loader = data.DataLoader(
    benign_unknown_subset,
    batch_size=opt.batch_size,
    num_workers=0,
    shuffle=True)

defense_loader = data.DataLoader(
    defense_subset,
    batch_size=opt.batch_size,
    num_workers=0,
    shuffle=True)

print("Identifying benign samples in defense subset...")
h_benign_preds = []
h_benign_ori_labels = []
with torch.no_grad():
    for inputs, labels in defense_loader:
        inputs, labels = inputs.to(opt.device), labels.to(opt.device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        h_benign_preds.append(preds)
        h_benign_ori_labels.append(labels)

h_benign_preds = torch.cat(h_benign_preds, dim=0)
h_benign_ori_labels = torch.cat(h_benign_ori_labels, dim=0)

benign_mask = (h_benign_ori_labels == h_benign_preds)
benign_indices = defense_subset_indices[benign_mask.cpu().numpy()]

if len(benign_indices) > DEFENSE_TRAIN_SIZE:
    benign_indices = np.random.choice(benign_indices, DEFENSE_TRAIN_SIZE, replace=False)

defense_subset = Subset(testset, benign_indices)
defense_loader = data.DataLoader(defense_subset, batch_size=opt.batch_size, num_workers=0, shuffle=True)

VT_TEMP_LABEL = "VT"
NVT_TEMP_LABEL = "NVT"
NoT_TEMP_LABEL = "NoT"

label_mapping = {
    "VT": 101,
    "NVT": 102,
    "NoT": 103
}

VICTIM = 1

UNKNOWN_SIZE_POSITIVE = 400
UNKNOWN_SIZE_NEGATIVE = 200

vt_count = nvt_count = NoT_count = 0

temp_bd_inputs_set = []
temp_bd_labels_set = []
temp_bd_pred_set = []

temp_cleanT_inputs_set = []
temp_cleanT_labels_set = []
temp_cleanT_pred_set = []

def create_bd(netG, netM, inputs):
    patterns = netG(inputs)
    patterns = netG.normalize_pattern(patterns)
    masks_output = netM.threshold(netM(inputs))
    bd_inputs = inputs + (patterns - inputs) * masks_output
    return bd_inputs

def create_targets(targets, opt, label):
    new_targets = torch.ones_like(targets) * label
    return new_targets.to(opt.device)

print("Generating VT and NVT sets...")
while vt_count < UNKNOWN_SIZE_POSITIVE or nvt_count < UNKNOWN_SIZE_NEGATIVE:
    for batch_idx, (inputs, labels) in enumerate(benign_unknown_loader):
        inputs, labels = inputs.to(opt.device), labels.to(opt.device)
        inputs_triggered = create_bd(netG, netM, inputs)
        preds_bd = torch.argmax(model(inputs_triggered), 1)
        victim_indices = (labels == VICTIM)
        non_victim_indices = (labels != VICTIM)

        if vt_count < UNKNOWN_SIZE_POSITIVE:
            label_value = label_mapping[VT_TEMP_LABEL]
            targets_victim_bd = create_targets(labels, opt, label_value)
            final_indices = victim_indices & (preds_bd == opt.target)
            temp_bd_inputs_set.append(inputs_triggered[final_indices])
            temp_bd_labels_set.append(targets_victim_bd[final_indices])
            temp_bd_pred_set.append(preds_bd[final_indices])
            vt_count += final_indices.sum().item()

        if nvt_count < UNKNOWN_SIZE_NEGATIVE:
            label_value = label_mapping[NVT_TEMP_LABEL]
            targets_clean = create_targets(labels, opt, label_value)
            final_indices = non_victim_indices
            temp_cleanT_inputs_set.append(inputs_triggered[final_indices])
            temp_cleanT_labels_set.append(targets_clean[final_indices])
            temp_cleanT_pred_set.append(preds_bd[final_indices])
            nvt_count += final_indices.sum().item()

bd_inputs_set = torch.cat(temp_bd_inputs_set, dim=0)[:UNKNOWN_SIZE_POSITIVE]
bd_labels_set = torch.cat(temp_bd_labels_set, dim=0)[:UNKNOWN_SIZE_POSITIVE]
bd_pred_set = torch.cat(temp_bd_pred_set, dim=0)[:UNKNOWN_SIZE_POSITIVE]

cleanT_inputs_set = torch.cat(temp_cleanT_inputs_set, dim=0)[:UNKNOWN_SIZE_NEGATIVE]
cleanT_labels_set = torch.cat(temp_cleanT_labels_set, dim=0)[:UNKNOWN_SIZE_NEGATIVE]
cleanT_pred_set = torch.cat(temp_cleanT_pred_set, dim=0)[:UNKNOWN_SIZE_NEGATIVE]

benign_real_labels_set = []
benign_inputs_set = []
benign_labels_set = []
benign_pred_set = []

print("Processing NoT samples...")
NoT_processed = 0
for batch_idx, (inputs, labels) in zip(range(len(benign_unknown_loader)), benign_unknown_loader):
    inputs, labels = inputs.to(opt.device), labels.to(opt.device)
    bs = inputs.shape[0]
    NoT_count += bs
    label_value = label_mapping[NoT_TEMP_LABEL]
    targets_benign = torch.ones_like(labels) * label_value

    if NoT_count <= UNKNOWN_SIZE_NEGATIVE:
        # Keep on GPU
        benign_real_labels_set.append(labels)
        benign_inputs_set.append(inputs.clone().detach())
        benign_labels_set.append(targets_benign)
        benign_pred_set.append(torch.argmax(model(inputs), 1))
        NoT_processed += bs
    else:
        # If we exceeded, we only take the needed portion
        needed = UNKNOWN_SIZE_NEGATIVE - (NoT_count - bs)
        if needed > 0:
            benign_real_labels_set.append(labels[:needed])
            benign_inputs_set.append(inputs[:needed].clone().detach())
            benign_labels_set.append(targets_benign[:needed])
            benign_pred_set.append(torch.argmax(model(inputs[:needed]), 1))
            NoT_processed += needed
        break

benign_inputs_set = torch.cat(benign_inputs_set, dim=0)
benign_labels_set = torch.cat(benign_labels_set, dim=0)
benign_pred_set = torch.cat(benign_pred_set, dim=0)
# benign_real_labels_set not used later in metrics, so we can ignore it or keep it
# If needed:
benign_real_labels_set = torch.cat(benign_real_labels_set, dim=0)

class CustomDataset(data.Dataset):
    def __init__(self, data, labels):
        super(CustomDataset, self).__init__()
        self.images = data
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        return img, label

bd_set = CustomDataset(data=bd_inputs_set, labels=bd_labels_set)
bd_loader = torch.utils.data.DataLoader(bd_set, batch_size=opt.batch_size, num_workers=0, shuffle=True)
del bd_inputs_set, bd_labels_set, bd_pred_set

cleanT_set = CustomDataset(data=cleanT_inputs_set, labels=cleanT_labels_set)
cleanT_loader = torch.utils.data.DataLoader(cleanT_set, batch_size=opt.batch_size, num_workers=0, shuffle=True)
del cleanT_inputs_set, cleanT_labels_set, cleanT_pred_set

benign_set = CustomDataset(data=benign_inputs_set, labels=benign_labels_set)
benign_loader = torch.utils.data.DataLoader(benign_set, batch_size=opt.batch_size, num_workers=0, shuffle=True)
del benign_inputs_set, benign_labels_set, benign_pred_set

print("Fetching final predictions for evaluation...")

# MOCK scenario for evaluation:
# In a real scenario, you'd have y_test_scores, y_test_pred, labels_all_unknown, etc.
# Assume:
labels_all_unknown = ["VT"] * 400 + ["NoT"] * 200
labels_all_unknown = np.array(labels_all_unknown)  # final metrics require CPU numpy arrays

# Keep computations on GPU until needed. Suppose we have scores as GPU:
y_test_scores = (torch.rand(600, device=opt.device)*2)  # random scores on GPU
y_test_pred = (y_test_scores > 0).long()  # Arbitrary threshold

# Now convert to CPU and NumPy for sklearn:
y_test_scores = y_test_scores.cpu().numpy()
y_test_pred = y_test_pred.cpu().numpy()

prediction_mask = (y_test_pred == 1)
prediction_labels = labels_all_unknown[prediction_mask]
label_counts = Counter(prediction_labels)

print("Performance of defense method:")
for label, count in label_counts.items():
    print(f'Label {label}: {count}')

fpr, tpr, thresholds = metrics.roc_curve((labels_all_unknown == VT_TEMP_LABEL).astype(int), y_test_scores, pos_label=1)
print("AUC:", metrics.auc(fpr, tpr))

tn, fp, fn, tp = confusion_matrix((labels_all_unknown == VT_TEMP_LABEL).astype(int), y_test_pred).ravel()
print("TPR:", tp / (tp + fn))
print("True Positives (TP):", tp)
print("False Positives (FP):", fp)
print("True Negatives (TN):", tn)
print("False Negatives (FN):", fn)

wandb.finish()
print("Done.")

