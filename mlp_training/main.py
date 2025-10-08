################################################
# Imports                                      #
################################################
import gc
import os
import math
import time
import wandb
import torch
import sklearn
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_utils import get_dataloaders



################################################
# Configs                                      #
################################################
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MLP_DIMS = [135168, 1024, 512, 1]
MODEL = "meta-llama_Llama-3.1-8B-Instruct"
TRAIN_DATASET_PATH = f'../RepExtraction/representations/combined_8500/{MODEL}_reps.json'
VAL_DATASET_PATH = f'../RepExtraction/representations/combined_4000_test/{MODEL}_reps.json'
INIT_LR = 1e-3
EPOCHS = 10
SEED = 42

################################################
# Model Definition                            #
################################################
class MLP(torch.nn.Module):
    def __init__(self, layer_dims):
        super(MLP, self).__init__()
        self.layer_dims = layer_dims

        layers_ls = [[
                    torch.nn.Linear(layer_dims[i], layer_dims[i+1]), 
                    torch.nn.BatchNorm1d(layer_dims[i+1]),
                    torch.nn.GELU(), 
                    torch.nn.Dropout(p=0.15)] 
                for i in range(len(layer_dims)-1)]

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            *[l for layers in layers_ls for l in layers][:-3],
        )

    def forward(self, x):
        return self.model(x)


################################################
# Training & Validation functions              #
################################################
def train(model, dataloader, optimizer, criterion, scheduler=None):

    model.train()
    tloss, tacc = 0, 0 
    batch_bar   = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    scaler = torch.amp.GradScaler()
    start_time = time.time()
    for i, (matrices, labels) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        ### Move Data to Device (Ideally GPU)
        matrices = matrices.to(DEVICE)
        labels   = labels.to(DEVICE)

        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
            logits  = model(matrices)
            loss    = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            if scheduler is not None: scheduler.step()
            scaler.update()

        tloss   += loss.item()
        tacc    += torch.sum((logits>0.5) == labels).item()/logits.shape[0]

        batch_bar.set_postfix(loss="{:.04f}".format(float(tloss / (i + 1))),
                              acc="{:.04f}%".format(float(tacc*100 / (i + 1))))
        batch_bar.update()

        ### Release memory
        del matrices, labels, logits
        torch.cuda.empty_cache()

    batch_bar.close()
    tloss /= len(dataloader)
    tacc  /= len(dataloader)

    return tloss, tacc


def eval(model, dataloader, criterion=None):

    model.eval()
    vloss, vacc = 0, 0
    batch_bar   = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    for i, (matrices, labels) in tqdm(enumerate(dataloader)):
        matrices = matrices.to(DEVICE)
        labels   = labels.to(DEVICE)

        with torch.inference_mode():
            logits  = model(matrices)
            loss    = criterion(logits, labels)

        vloss += loss.item()
        vacc  += torch.sum((logits>0.5) == labels).item()/logits.shape[0]

        batch_bar.set_postfix(loss="{:.04f}".format(float(vloss / (i + 1))),
                              acc="{:.04f}%".format(float(vacc*100 / (i + 1))))
        batch_bar.update()

        ### Release memory
        del matrices, labels, logits
        torch.cuda.empty_cache()

    batch_bar.close()
    vloss   /= len(dataloader)
    vacc    /= len(dataloader)

    return vloss, vacc


################################################
# Main                                        #
################################################
if __name__ == "__main__":
    if SEED is not None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
    train_loader, val_loader = get_dataloaders(TRAIN_DATASET_PATH, VAL_DATASET_PATH, 128, layers=None, n_workers=8)#, train_subset=128, val_subset=10)

    model = MLP(MLP_DIMS).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=INIT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    for epoch in range(EPOCHS):
        train(model, train_loader, optimizer, criterion, scheduler)
        eval(model, val_loader, criterion)
