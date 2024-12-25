import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import plotly.express as px
import torch
import torch.nn as nn
import torch.utils.data as data
import wandb
wandb.login(key="e09f73bb0df882dd4606253c95e1bc68801828a0")

from collections import Counter
from pyod.models.pca import PCA as PyOD_PCA
from sklearn import metrics
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchmetrics.functional import pairwise_euclidean_distance
from umap import UMAP
from numpy.random import choice

from classifier_models import PreActResNet18, VGG
from defense_dataloader import get_dataset
from networks.models import Generator, NetC_MNIST

# Set environment variables
os.environ['WANDB_NOTEBOOK_NAME'] = 'TED.ipynb'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.makedirs("visualization", exist_ok=True)

opt = argparse.Namespace()
opt.dataset = "cifar10"
opt.device = "cuda" if torch.cuda.is_available() else "cpu"
opt.batch_size = 100
opt.data_root = "../data/"
opt.target = 0
opt.attack_mode = "SSDT"

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

model = load_model()
state_dict = load_model_state()
model = load_state(model, state_dict["netC"])
print(model)

netG = Generator(opt)
netG = load_state(netG, state_dict["netG"])

netM = Generator(opt, out_channels=1)
netM = load_state(netM, state_dict["netM"])

testset = get_dataset(opt, train=True)

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
            correct_preds_indices = (preds_bd == opt.target)
            final_indices = victim_indices & correct_preds_indices
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

NoT_count = 0
for batch_idx, (inputs, labels) in zip(range(len(benign_unknown_loader)), benign_unknown_loader):
    inputs, labels = inputs.to(opt.device), labels.to(opt.device)
    bs = inputs.shape[0]
    NoT_count += bs
    label_value = label_mapping[NoT_TEMP_LABEL]
    targets_benign = torch.ones_like(labels) * label_value

    if NoT_count <= UNKNOWN_SIZE_NEGATIVE:
        benign_real_labels_set.append(labels)
        benign_inputs_set.append(inputs.clone().detach())
        benign_labels_set.append(targets_benign)
        benign_pred_set.append(torch.argmax(model(inputs), 1))
    else:
        needed = UNKNOWN_SIZE_NEGATIVE - (NoT_count - bs)
        if needed > 0:
            benign_real_labels_set.append(labels[:needed])
            benign_inputs_set.append(inputs[:needed].clone().detach())
            benign_labels_set.append(targets_benign[:needed])
            benign_pred_set.append(torch.argmax(model(inputs[:needed]), 1))
        break

benign_inputs_set = torch.cat(benign_inputs_set, dim=0)
benign_labels_set = torch.cat(benign_labels_set, dim=0)
benign_pred_set = torch.cat(benign_pred_set, dim=0)
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

bd_set = CustomDataset(bd_inputs_set, bd_labels_set)
bd_loader = torch.utils.data.DataLoader(bd_set, batch_size=opt.batch_size, num_workers=0, shuffle=True)
del bd_inputs_set, bd_labels_set, bd_pred_set

cleanT_set = CustomDataset(cleanT_inputs_set, cleanT_labels_set)
cleanT_loader = torch.utils.data.DataLoader(cleanT_set, batch_size=opt.batch_size, num_workers=0, shuffle=True)
del cleanT_inputs_set, cleanT_labels_set, cleanT_pred_set

benign_set = CustomDataset(benign_inputs_set, benign_labels_set)
benign_loader = torch.utils.data.DataLoader(benign_set, batch_size=opt.batch_size, num_workers=0, shuffle=True)
del benign_inputs_set, benign_labels_set, benign_pred_set

def fetch_activation(model, device, loader, activations):
    model.eval()
    all_h_label = []
    pred_set = []
    h_batch = {}
    activation_container = {}

    # Initialize activation_container keys after first batch
    first = True
    for batch_idx, (images, labels) in enumerate(loader, start=1):
        output = model(images.to(device))
        preds = torch.argmax(output, -1)
        pred_set.append(preds)
        for key in activations:
            h_ = activations[key].data.view(images.shape[0], -1)
            if first:
                activation_container[key] = []
            for h in h_:
                activation_container[key].append(h)
        for label in labels:
            all_h_label.append(label)
        first = False

    for key in activation_container:
        activation_container[key] = torch.stack(activation_container[key], dim=0).to(device)

    all_h_label = torch.stack(all_h_label, dim=0).to(device)
    pred_set = torch.cat(pred_set, dim=0).to(device)

    return all_h_label, activation_container, pred_set

hook_handle = []
activations = {}
net_children = model.modules()

Test_C = opt.class_number + 3
topological_representation = {}
candidate_ = {}
layer_names = {}
index = 0
for _, child in enumerate(net_children):
    if isinstance(child, nn.Conv2d) and child.kernel_size != (1, 1):
        hook_handle.append(child.register_forward_hook(lambda m,i,o: activations.update({f"Conv2d_{index}": o.detach()})))
        index += 1
    if isinstance(child, nn.ReLU):
        hook_handle.append(child.register_forward_hook(lambda m,i,o: activations.update({f"Relu_{index}": o.detach()})))
        index += 1
    if isinstance(child, nn.Linear):
        hook_handle.append(child.register_forward_hook(lambda m,i,o: activations.update({f"Linear_{index}": o.detach()})))
        index += 1

h_bd_ori_labels, h_bd_activations, h_bd_preds = fetch_activation(model, opt.device, bd_loader, activations)
h_benign_ori_labels, h_benign_activations, h_benign_preds = fetch_activation(model, opt.device, benign_loader, activations)
h_cleanT_ori_labels, h_cleanT_activations, h_cleanT_preds = fetch_activation(model, opt.device, cleanT_loader, activations)
h_defense_ori_labels, h_defense_activations, h_defense_preds = fetch_activation(model, opt.device, defense_loader, activations)

def calculate_accuracy(ori_labels, preds):
    correct = torch.sum(ori_labels == preds)
    total = len(ori_labels)
    accuracy = (correct.float() / total) * 100
    return accuracy

accuracy_defense = calculate_accuracy(h_defense_ori_labels, h_defense_preds)
accuracy_VT = calculate_accuracy(opt.target * torch.ones_like(h_bd_preds), h_bd_preds)

print(f"Accuracy on defense_loader: {accuracy_defense.item()}%")
print(f"Accuracy on bd_loader: {accuracy_VT.item()}%")

# For UMAP and PCA steps, we must convert to CPU and numpy
def plot_activations(activations, labels, title):
    # activations: torch tensor on GPU
    # labels: array-like
    # Convert to CPU numpy:
    activations_cpu = activations.cpu().numpy()
    labels_cpu = np.array(labels)

    umap_2d = UMAP(random_state=0)
    projections = umap_2d.fit_transform(activations_cpu)
    df_classes = pd.DataFrame(labels_cpu)
    fig = px.scatter(
        projections, x=0, y=1,
        color=df_classes[0].astype(str), labels={'color': 'label'}
    )
    fig.update_layout(title=title)

    html_path = f"visualization/{title.replace(' ', '_')}.html"
    fig.write_html(html_path, include_plotlyjs='cdn')
    fig.show()

sample_rate = 0.2
total_bd = h_bd_activations[next(iter(h_bd_activations))].shape[0]
total_defense = h_defense_activations[next(iter(h_defense_activations))].shape[0]

bd_indices = choice(total_bd, int(total_bd * sample_rate), replace=False)
defense_indices = choice(total_defense, int(total_defense * sample_rate), replace=False)

h_bd_ori_labels_prefixed = ["BD " + str(l.item()) for l in h_bd_ori_labels]

print_umat = True

if print_umat:
    for key in h_bd_activations:
        sampled_bd = h_bd_activations[key][bd_indices]
        sampled_defense = h_defense_activations[key][defense_indices]

        activations_concat = torch.cat((sampled_bd, sampled_defense), dim=0)
        labels_concat = np.concatenate((np.array(h_bd_ori_labels_prefixed)[bd_indices], h_defense_ori_labels[defense_indices].cpu().numpy()), axis=0)

        plot_activations(activations_concat, labels_concat, title=f"UMAP for {key}")

def gather_activation_into_class(target, h):
    h_c_c = [0 for _ in range(Test_C)]
    for c in range(Test_C):
        idxs = (target == c).nonzero(as_tuple=True)[0]
        if len(idxs) == 0:
            continue
        h_c = h[idxs, :]
        h_c_c[c] = h_c
    return h_c_c

def get_dis_sort(item, destinations):
    item = item.unsqueeze(0)
    new_dis = pairwise_euclidean_distance(item, destinations)
    _, indices_individual = torch.sort(new_dis)
    return indices_individual

for layer in h_defense_activations:
    topological_representation[layer] = {}

candidate_ = {}

def getDefenseRegion(final_prediction, h_defense_activation, processing_label, layer, layer_test_region_individual):
    if layer not in layer_test_region_individual:
        layer_test_region_individual[layer] = {}
    layer_test_region_individual[layer][processing_label] = []

    candidate_[layer] = gather_activation_into_class(final_prediction, h_defense_activation)

    if isinstance(candidate_[layer][processing_label], int):
        print("No sample in this class")
    else:
        for index, item in enumerate(candidate_[layer][processing_label]):
            ranking_array = get_dis_sort(item, h_defense_activation)[0]
            ranking_array = ranking_array[1:]
            r_ = final_prediction[ranking_array]
            indices = (r_ == processing_label).nonzero(as_tuple=True)
            if len(indices[0]) > 0:
                itemindex = indices[0][0].item()
                layer_test_region_individual[layer][processing_label].append(itemindex)

    return layer_test_region_individual

def getLayerRegionDistance(new_prediction, new_activation, new_temp_label,
                           h_defense_prediction, h_defense_activation,
                           layer, layer_test_region_individual):
    labels = torch.unique(new_prediction)
    candidate_ = gather_activation_into_class(new_prediction, new_activation)

    if layer not in layer_test_region_individual:
        layer_test_region_individual[layer] = {}
    layer_test_region_individual[layer][new_temp_label] = []

    for processing_label in labels:
        if isinstance(candidate_[processing_label], int):
            continue
        for index, item in enumerate(candidate_[processing_label]):
            ranking_array = get_dis_sort(item, h_defense_activation)[0]
            r_ = h_defense_prediction[ranking_array]
            indices = (r_ == processing_label).nonzero(as_tuple=True)
            if len(indices[0]) > 0:
                itemindex = indices[0][0].item()
                layer_test_region_individual[layer][new_temp_label].append(itemindex)

    return layer_test_region_individual

class_names = torch.unique(h_defense_ori_labels)

for label in class_names:
    for layer in h_defense_activations:
        topological_representation = getDefenseRegion(
            final_prediction=h_defense_preds,
            h_defense_activation=h_defense_activations[layer],
            processing_label=label.item(),
            layer=layer,
            layer_test_region_individual=topological_representation
        )
        topo_rep_array = topological_representation[layer][label.item()]
        print(f"Topological Representation Label [{label.item()}] & layer [{layer}]: {topo_rep_array}")
        if len(topo_rep_array) > 0:
            print(f"Mean: {np.mean(topo_rep_array)}\n")
        else:
            print("Mean: N/A\n")

for layer_ in h_bd_activations:
    topological_representation = getLayerRegionDistance(
        new_prediction=h_bd_preds,
        new_activation=h_bd_activations[layer_],
        new_temp_label=label_mapping[VT_TEMP_LABEL],
        h_defense_prediction=h_defense_preds,
        h_defense_activation=h_defense_activations[layer_],
        layer=layer_,
        layer_test_region_individual=topological_representation
    )
    topo_rep_array_vt = topological_representation[layer_][label_mapping[VT_TEMP_LABEL]]
    print(f"Topological Representation Label [VT] & layer [{layer_}]: {topo_rep_array_vt}")
    if len(topo_rep_array_vt) > 0:
        print(f"Mean: {np.mean(topo_rep_array_vt)}\n")
    else:
        print("Mean: N/A\n")

for layer_ in h_cleanT_activations:
    topological_representation = getLayerRegionDistance(
        new_prediction=h_cleanT_preds,
        new_activation=h_cleanT_activations[layer_],
        new_temp_label=label_mapping[NVT_TEMP_LABEL],
        h_defense_prediction=h_defense_preds,
        h_defense_activation=h_defense_activations[layer_],
        layer=layer_,
        layer_test_region_individual=topological_representation
    )
    topo_rep_array_nvt = topological_representation[layer_][label_mapping[NVT_TEMP_LABEL]]
    print(f"Topological Representation [NVT] - layer [{layer_}]: {topo_rep_array_nvt}")
    if len(topo_rep_array_nvt) > 0:
        print(f"Mean: {np.mean(topo_rep_array_nvt)}\n")
    else:
        print("Mean: N/A\n")

file_name = f"{opt.dataset}_k_{opt.target}_{opt.attack_mode}.pkl"
file_path = os.path.join(file_name)
with open(file_path, 'wb') as file:
    pickle.dump(topological_representation, file)

wandb.save(file_path)
print(wandb.save(file_path))

with open(file_path, 'rb') as file:
    topological_representation = pickle.load(file)

def aggregate_by_all_layers(output_label):
    inputs_container = []
    first_key = list(topological_representation.keys())[0]
    length = len(topological_representation[first_key][output_label])
    labels_container = np.repeat(output_label, length)
    for l in topological_representation.keys():
        arr = topological_representation[l][output_label]
        inputs_container.append(np.array(arr))
    # Each array in inputs_container is a 1D list of indices. We'll just form a 2D array
    return np.array(inputs_container).T, labels_container

inputs_all_benign = []
labels_all_benign = []
inputs_all_unknown = []
labels_all_unknown = []

first_key = list(topological_representation.keys())[0]
class_name = list(topological_representation[first_key])
class_name = np.array(class_name)  # keys are int labels

for inx in class_name:
    inx = int(inx)
    inputs, labels = aggregate_by_all_layers(output_label=inx)
    # Decide if benign or unknown:
    if inx not in [label_mapping[VT_TEMP_LABEL], label_mapping[NVT_TEMP_LABEL], label_mapping[NoT_TEMP_LABEL]]:
        inputs_all_benign.append(inputs)
        labels_all_benign.append(labels)
    else:
        inputs_all_unknown.append(inputs)
        labels_all_unknown.append(labels)

if len(inputs_all_benign) > 0:
    inputs_all_benign = np.concatenate(inputs_all_benign)
    labels_all_benign = np.concatenate(labels_all_benign)
else:
    inputs_all_benign = np.empty((0,0))
    labels_all_benign = np.empty((0,))

if len(inputs_all_unknown) > 0:
    inputs_all_unknown = np.concatenate(inputs_all_unknown)
    labels_all_unknown = np.concatenate(labels_all_unknown)
else:
    inputs_all_unknown = np.empty((0,0))
    labels_all_unknown = np.empty((0,))

# PCA from sklearn and PyOD must run on CPU numpy
if inputs_all_benign.size > 0 and inputs_all_unknown.size > 0:
    pca_t = sklearn_PCA(n_components=2)
    pca_fit = pca_t.fit(inputs_all_benign)
    benign_trajectories = pca_fit.transform(inputs_all_benign)
    trajectories = pca_fit.transform(np.concatenate((inputs_all_unknown, inputs_all_benign), axis=0))
    df_classes = pd.DataFrame(np.concatenate((labels_all_unknown, labels_all_benign), axis=0))

    fig_ = px.scatter(
        trajectories, x=0, y=1, color=df_classes[0].astype(str), labels={'color': 'digit'},
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )

    pca_html_path = "visualization/pca_scatter.html"
    fig_.write_html(pca_html_path, include_plotlyjs='cdn')
    fig_.show()

    pca = PyOD_PCA(contamination=0.01, n_components='mle')
    pca.fit(inputs_all_benign)

    y_train_pred = pca.labels_
    y_train_scores = pca.decision_scores_
    y_train_scores = pca.decision_function(inputs_all_benign)
    y_train_pred = pca.predict(inputs_all_benign)

    y_test_scores = pca.decision_function(inputs_all_unknown)
    y_test_pred = pca.predict(inputs_all_unknown)
    prediction_mask = (y_test_pred == 1)
    prediction_labels = labels_all_unknown[prediction_mask]
    label_counts = Counter(prediction_labels)

    for label, count in label_counts.items():
        print(f'Label {label}: {count}')

    fpr, tpr, thresholds = metrics.roc_curve((labels_all_unknown == label_mapping[VT_TEMP_LABEL]).astype(int), y_test_scores, pos_label=1)
    print("AUC:", metrics.auc(fpr, tpr))

    tn, fp, fn, tp = confusion_matrix((labels_all_unknown == label_mapping[VT_TEMP_LABEL]).astype(int), y_test_pred).ravel()
    print("TPR:", tp / (tp + fn))
    print("True Positives (TP):", tp)
    print("False Positives (FP):", fp)
    print("True Negatives (TN):", tn)
    print("False Negatives (FN):", fn)
else:
    print("No data for PCA and PyOD steps.")

wandb.finish()
print("Done.")

