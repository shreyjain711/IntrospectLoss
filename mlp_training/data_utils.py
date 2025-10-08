import mmap
import torch
import orjson as json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class LayerRepSafeDataset(Dataset):
    def __init__(self, dataset_path, subset=None, layers=None):
        self.dataset_path = dataset_path

        self.labels = []
        self.layer_rep_matrices = []

        with open(self.dataset_path, 'r') as f:
            # read each line as a json object in a streaming fashion
            for line in tqdm(f):
                if subset is not None and len(self.layer_rep_matrices) == subset:
                    break
                d = json.loads(line)
                self.labels.append(d['is_safe'])
                layer_matrix = []
                for k in d.keys():
                    if k.startswith('layer_'):
                        layer_num = int(k.split('_')[1])
                        if layers is not None and layer_num not in layers: continue
                        layer_matrix.append(torch.tensor(d[k]))
                self.layer_rep_matrices.append(torch.stack(layer_matrix, dim=0))

        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.layer_rep_matrices[i], self.labels[i]

class LazyLayerRepSafeDataset(Dataset):
    def __init__(self, dataset_path, subset=None, layers=None):
        self.dataset_path = dataset_path
        self.layers = layers
        self.offsets = []

        with open(dataset_path, 'r') as f:
            pos = f.tell()
            for i, line in enumerate(f):
                if subset is not None and i == subset:
                    break
                self.offsets.append(pos)
                pos = f.tell()
        self.length = len(self.offsets)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        with open(self.dataset_path, 'r') as f:
            f.seek(self.offsets[i])
            d = json.loads(f.readline())
        
        layer_matrix = []
        for k in d:
            if k.startswith('layer_'):
                layer_num = int(k.split('_')[1])
                if self.layers is not None and layer_num not in self.layers:
                    continue
                layer_matrix.append(torch.tensor(d[k]))
        label = d['is_safe']
        return torch.stack(layer_matrix, dim=0), label


def get_dataloaders(train_dataset_path, val_dataset_path, batch_size, layers=None, n_workers=8, train_subset=None, val_subset=None):
    train_data = LayerRepSafeDataset(train_dataset_path, layers=layers, subset=train_subset)
    val_data = LayerRepSafeDataset(val_dataset_path, layers=layers, subset=val_subset)
    
    # train_data = LazyLayerRepSafeDataset(train_dataset_path, layers=layers, subset=train_subset)
    # val_data = LazyLayerRepSafeDataset(val_dataset_path, layers=layers, subset=val_subset)
    
    train_loader = torch.utils.data.DataLoader(
        dataset     = train_data,
        num_workers = n_workers,
        batch_size  = batch_size,
        pin_memory  = True,
        shuffle     = True
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset     = val_data,
        num_workers = n_workers,
        batch_size  = batch_size,
        pin_memory  = True
    )

    print(f"Train dataset samples = {train_data.__len__()}, batches = {len(train_loader)}")
    print(f"Validation dataset samples = {val_data.__len__()}, batches = {len(val_loader)}")
    
    return train_loader, val_loader
