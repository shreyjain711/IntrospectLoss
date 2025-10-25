################################################
# Imports                                      #
################################################
import gc
import os
import sys
import math
import time
import wandb
import torch
import sklearn
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_utils import get_dataloaders
from mlp import MLP



################################################
# Configs                                      #
################################################
model_name = sys.argv[1] # "l3_8"


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

SEED = 711
LLM = ({"q3_4": "Qwen/Qwen3-4B-Instruct-2507", 
        "q3_8": "Qwen/Qwen3-8B", 
        "l3_8": "meta-llama/Llama-3.1-8B-Instruct"}).get(model_name).replace('/', '_')

TOTAL_MLP_LAYERS = ({"q3_4": 36, "q3_8": 36, "l3_8": 32}).get(model_name)
MIDDLE_LAYER = TOTAL_MLP_LAYERS // 2
INTERNAL_REP_SIZE = ({"q3_4": 2560, "q3_8": 4096, "l3_8": 4096}).get(model_name)

TRAIN_DATASET_PATH = f'../RepExtraction/representations/combined_8500/{LLM}_reps.json'
VAL_DATASET_PATH = f'../RepExtraction/representations/combined_4000_test/{LLM}_reps.json'

MLP_DIMS = [INTERNAL_REP_SIZE, 1024, 512, 1]
INIT_LR = 1e-3
EPOCHS = 20
BATCH_SIZE = 256


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

        matrices = matrices.to(DEVICE)
        labels   = labels.to(DEVICE).unsqueeze(1)

        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
            logits  = model(matrices)
            loss    = criterion(logits, labels.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            if scheduler is not None: scheduler.step()
            scaler.update()

        tloss   += loss.item()
        tacc    += torch.sum((logits>0.5) == labels).item()/logits.shape[0]

        batch_bar.set_postfix(loss="{:.04f}".format(float(tloss / (i + 1))),
                              acc="{:.04f}%".format(float(tacc*100 / (i + 1))))
        batch_bar.update()

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
        labels   = labels.to(DEVICE).unsqueeze(1)

        with torch.inference_mode():
            logits  = model(matrices)
            loss    = criterion(logits, labels.float())

        vloss += loss.item()
        vacc  += torch.sum((logits>0.5) == labels).item()/logits.shape[0]

        batch_bar.set_postfix(loss="{:.04f}".format(float(vloss / (i + 1))),
                              acc="{:.04f}%".format(float(vacc*100 / (i + 1))))
        batch_bar.update()

        del matrices, labels, logits
        torch.cuda.empty_cache()

    batch_bar.close()
    vloss   /= len(dataloader)
    vacc    /= len(dataloader)

    return vloss, vacc

def run_expt(model, optimizer, criterion, scheduler, dropouts, expt_name):
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        best_vloss_model_state_dict = None

        print(f"\n\nStarting experiment: {expt_name}")
        for epoch in range(EPOCHS):
            # Training
            tloss, tacc = train(model, train_loader, optimizer, criterion, scheduler)
            train_losses.append(tloss)
            train_accs.append(tacc)
            
            # Update dropout
            model.set_dropout(dropouts[epoch])
            
            # Validation
            vloss, vacc = eval(model, val_loader, criterion)
            val_losses.append(vloss)
            val_accs.append(vacc)
            
            #save best vloss model dict
            if vloss == min(val_losses):
                best_vloss_model_state_dict = model.state_dict().copy()

        # Plotting 1x4 chart
        # fig, axes = plt.subplots(4, 1, figsize=(8, 12))
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(expt_name, fontsize=16)
        axes[0].plot(range(EPOCHS), train_losses, marker='o')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')

        axes[1].plot(range(EPOCHS), train_accs, marker='o')
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')

        axes[2].plot(range(EPOCHS), val_losses, marker='o')
        axes[2].set_title('Validation Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')

        axes[3].plot(range(EPOCHS), val_accs, marker='o')
        axes[3].set_title('Validation Accuracy')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Accuracy')

        plt.tight_layout()
        plt.savefig(expt_name.replace(" ", "_") + '.png')

        model.load_state_dict(best_vloss_model_state_dict)
        return model


################################################
# Main                                        #
################################################
if __name__ == "__main__":
    if SEED is not None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
    train_loader, val_loader = get_dataloaders(TRAIN_DATASET_PATH, VAL_DATASET_PATH, BATCH_SIZE, layers=None, n_workers=8)#, train_subset=128, val_subset=10)

    criterion = torch.nn.BCEWithLogitsLoss()
    
    # mode lin agt
    dropouts = np.linspace(0.5, 0.3, EPOCHS)
    model = MLP(TOTAL_MLP_LAYERS+1, MLP_DIMS, dropout=dropouts[0]).to(DEVICE)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=INIT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    model = run_expt(model, optimizer, criterion, scheduler, dropouts, expt_name=f'{model_name}_mode_lin_agt')
    
    # create new plot for layer weights
    plt.figure(figsize=(10, 6))
    layer_weights = model.layer_weights.cpu().tolist()
    plt.bar(range(len(layer_weights)), layer_weights)
    plt.xlabel('Layer Index')
    plt.ylabel('Learned Weight')
    plt.title(f'Learned Layer Weights - {model_name} - lin_agt')
    plt.savefig(f'{model_name}_layer_weights_lin_agt.png')
    
    torch.save(model.state_dict(), f'{model_name}_mlp_mode_lin_agt.pth')


    # model lyr_MIDDLE_LAYER
    dropouts = np.linspace(0.5, 0.3, EPOCHS)
    model = MLP(TOTAL_MLP_LAYERS+1, MLP_DIMS, dropout=dropouts[0], mode=f'lyr_{str(MIDDLE_LAYER)}').to(DEVICE)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=INIT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    run_expt(model, optimizer, criterion, scheduler, dropouts, expt_name=f'{model_name}_mode_lyr_{str(MIDDLE_LAYER)}')
    torch.save(model.state_dict(), f'{model_name}_mlp_mode_lyr_{str(MIDDLE_LAYER)}.pth')