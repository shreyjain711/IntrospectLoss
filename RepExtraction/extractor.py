from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gc
import os
import json
import torch
import sys
import argparse

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
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True,
                                                 trust_remote_code=True
                                                 )#,local_files_only=True) # use only cached files, makes it faster
    model.eval()

    # Prepare storage for representations
    layer_indices = layer_indices if layer_indices is not None else list(range(model.config.num_hidden_layers + 1))

    # Store hidden states for each data point and layer
    output_path_dir = os.path.dirname(output_path)
    os.makedirs(output_path_dir, exist_ok=True)


    # Iterate over data
    with open(output_path, 'w') as f:
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting representations"):
                batch_size = len(batch['prompt'])
                exec_device = next(iter(model.parameters())).device
                inputs = tokenizer(batch['prompt'], return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(exec_device) for k, v in inputs.items()}
                outputs = model(**inputs, use_cache=False)
                
                cpu_layers = []
                for idx in layer_indices:
                    t = outputs.hidden_states[idx].detach().cpu().to(torch.float32)
                    cpu_layers.append(t)
                del outputs
                
                batch_indices = list(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
                lines = []
                for i in range(batch_size):
                    rep = {'idx': batch_indices[i], 
                        'prompt': batch['prompt'][i], 
                        'is_safe': batch['is_safe'][i].item(), 
                        'origin_data': batch['origin_data'][i]}
                    for idx in layer_indices:
                        rep[f"layer_{idx}"] = cpu_layers[idx][i, -1].numpy().tolist()
                    lines.append(json.dumps(rep, ensure_ascii=False) + '\n')
                f.writelines(lines); f.flush()

                del cpu_layers, lines, inputs
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Extract representations from a Hugging Face causal LM model.")
    argparser.add_argument('--dataset', type=str, default='train', choices=['train', 'test'], help="Dataset to use: 'train' or 'test'.")
    argparser.add_argument('--batch_size', type=int, default=32, help="Batch size for dataloader.")
    argparser.add_argument('--model', type=str, default='l3_8', choices=['q3_4', 'q3_8', 'l3_8'], help="Model to use: 'q3_4' (Qwen-3-4B), 'q3_8' (Qwen-3-8B), or 'l3_8' (Llama-3.1-8B).")
    args = argparser.parse_args()

    datasets = {'train': "/ocean/projects/cis250042p/sjain13/IntrospectLoss/RepExtraction/input_data/combined_8500.json", 'test': "/ocean/projects/cis250042p/sjain13/IntrospectLoss/RepExtraction/input_data/combined_4000_test.json"}
    
    dataset_name = datasets.get(args.dataset, "train")
    batch_size = args.batch_size
    model_name = ({"q3_4": "Qwen/Qwen3-4B-Instruct-2507", "q3_8": "Qwen/Qwen3-8B", "l3_8": "meta-llama/Llama-3.1-8B-Instruct"}).get(args.model)
    layer_indices = None # None for all layers else list of numbers

    dataloader = get_dataloader(dataset_name, batch_size=batch_size, shuffle=False)
    
    print(f"Number of batches in dataloader: {len(dataloader)}")

    output_path = f"/ocean/projects/cis250042p/sjain13/IntrospectLoss/RepExtraction/representations/{dataset_name.split('/')[-1].split('.')[0]}/{model_name.replace('/', '_')}_reps.json"
    extract_representations(model_name, dataloader, output_path, layer_indices=layer_indices)
