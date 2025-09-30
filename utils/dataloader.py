from datasets import load_dataset
from torch.utils.data import DataLoader

counter = 0
def sanitize(example):
    global counter
    counter+=1
    if len(example['prompt'])>1024: example['prompt'] = '[IGNORED] redacted for length'
    for k in example.keys():
        if example[k] is None:
            example[k] = '' if k != 'metadata' else {}
    return example

def load_hf_dataset(dataset_name, split='train', **kwargs):
    """
    Loads a Hugging Face dataset.

    Args:
        dataset_name (str): Name or path of the dataset.
        split (str): Which split to load (default: 'train').
        **kwargs: Additional arguments for load_dataset.

    Returns:
        Dataset: Hugging Face Dataset object.
    """
    dataset = load_dataset(dataset_name, split=split, **kwargs)
    dataset = dataset.map(sanitize)
    return dataset

def get_dataloader(dataset_name, batch_size=32, shuffle=True, num_workers=0, collate_fn=None):
    """
    Returns a PyTorch DataLoader for a Hugging Face dataset.

    Args:
        dataset: Hugging Face Dataset object.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes.
        collate_fn (callable, optional): Custom collate function.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    if dataset_name.startswith("hf-"):
        dataset_name = dataset_name[len("hf-"):]
        dataset = load_hf_dataset(dataset_name)
    else:
        dataset = load_dataset('json', data_files=dataset_name, split='train', streaming=True)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
