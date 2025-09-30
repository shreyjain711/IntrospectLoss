from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import json
import torch
import sys

introspect_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if introspect_path not in sys.path:
    sys.path.insert(0, introspect_path)
from utils.dataloader import get_dataloader


def extract_representations(model_name, dataloader, output_path, layer_indices=None, device=('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Loads a Hugging Face causal LM model and extracts internal representations from specified layers.

    Args:
        model_name (str): Name of the pretrained model.
        dataloader (torch.utils.data.DataLoader): Dataloader yielding input batches.
        layer_indices (list[int] or None): Indices of layers to extract representations from. If None, extracts from all layers.
        device (str): Device to run the model on.

    Returns:
        dict: {layer_idx: [representations]}
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 output_hidden_states=True, 
                                                 device_map="balanced",          # shard across visible GPUs
                                                 torch_dtype=torch.bfloat16, # efficient dtype for Llama 3.x
                                                 low_cpu_mem_usage=True )
    model.eval()

    # Prepare storage for representations
    representations = {}
    layer_indices = layer_indices if layer_indices is not None else list(range(model.config.num_hidden_layers + 1))

    # Store hidden states for each data point and layer
    def store_hidden_states(hidden_states, batch, batch_idx):
        # Create output directory if it doesn't exist
        output_path_dir = os.path.dirname(output_path)
        os.makedirs(output_path_dir, exist_ok=True)

        batch_indices = list(range(batch_idx * len(batch), (batch_idx + 1) * len(batch)))
        with open(output_path, 'a') as f:
            for i in range(len(batch)):
                rep = {'idx': batch_indices[i], 
                       'prompt': batch['prompt'][i], 
                       'prompt_harm_label': batch['prompt_harm_label'][i], 
                       'prompt_safety_categories': batch['prompt_safety_categories'][i], 
                       'metadata.language': batch['metadata']['language'][i], 
                       'metadata.source': batch['metadata']['source'][i]}
                for idx in layer_indices:
                    rep[f"layer_{idx}"] = hidden_states[idx][i, -1, :].detach().cpu().tolist()
                f.write(json.dumps(rep) + '\n')

    # Iterate over data
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Get data indices if available, else use running index
            exec_device = next(iter(model.parameters())).device
            inputs = tokenizer(batch['prompt'], return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(exec_device) for k, v in inputs.items()}
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # tuple: (embeddings, layer1, ..., layerN)
            store_hidden_states(hidden_states, batch, batch_idx)
            torch.cuda.empty_cache()
    
    return representations


if __name__ == "__main__":
    dataset_name = "hf-ToxicityPrompts/PolyGuardMix"
    batch_size = int(sys.argv[1])
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    layer_indices = None # None for all layers else list of numbers

    dataloader = get_dataloader(dataset_name, batch_size=batch_size, shuffle=False)
    output_path = f"/data/user_data/shreyj/IntrospectLossData/representations/{dataset_name.replace('/', '_')}/{model_name.replace('/', '_')}_reps.json"
    extract_representations(model_name, dataloader, output_path, layer_indices=layer_indices)
