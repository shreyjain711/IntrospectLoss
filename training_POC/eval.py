import os
import gc
import re
import torch
import pickle
from glob import glob
from tqdm import tqdm
from peft import PeftModel
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

os.environ['HF_HOME'] = '/ocean/projects/cis250042p/sjain13'
login(token='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')



#########################
# Inference Functions  #
#########################

def get_ft_model(model_name, checkpoint_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              trust_remote_code=True,
                                              padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    if checkpoint_path is not None:
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        print(f"Model loaded from {checkpoint_path}")

    return model, tokenizer

def unpack_batch(batch_dict):
    """
    Convert a batch like:
      {"question": [...], "answer": [...]}
    into:
      [{"question": ..., "answer": ...}, ...]
    """
    keys = list(batch_dict.keys())
    n = len(batch_dict[keys[0]])
    return [{k: batch_dict[k][i] for k in keys} for i in range(n)]


def create_prompt_messages(example, prompt_field, use_ans_sep):
    content = f"Question: {example[prompt_field]}\n\n" + ('Answer:' if not use_ans_sep else (
                "Provide your reasoning if needed, but put ONLY the final answer after '####'.\n" 
                "For example:\n" 
                "#### 42"))
    return [{"role": "user", "content": content}]


def batched_inference(model_name, model, tokenizer, dataset, prompt_field, use_ans_sep, batch_size=8, max_new_tokens=512):
    results = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = unpack_batch(dataset[i:i+batch_size])
        
        # Prepare chat-formatted inputs
        messages_batch = [create_prompt_messages(ex, prompt_field, use_ans_sep) for ex in batch]

        # Tokenize with the chat template
        inputs = tokenizer.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            return_dict=True,
        ).to(model.device)

        sampling_params = {"do_sample": True}
        if model_name == 'meta-llama/Llama-3.1-8B-Instruct':
            sampling_params.update({
                "temperature": 0.6,
                "top_p": 0.9,
            })
        elif model_name == 'Qwen/Qwen3-4B-Instruct-2507':
            sampling_params.update({
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.0,
            })

        # Generate outputs
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **sampling_params
            )

        # Decode all outputs
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract generated answers based on '####'
        for ex, user_msg, full_output in zip(batch, messages_batch, decoded_outputs):
            if "####" in full_output:
                generated = full_output.split("####")[-1].strip('# \n')
            else:
                # Fallback if the marker is missing
                generated = full_output.strip()
            user_msg.append({'role': 'assistant', 'content': generated})
            
            results.append({
                "question": ex[prompt_field],
                "ground_truth": None if 'answer' not in ex else ex["answer"],
                "chat": user_msg,
            })

    return results

def run_inf(base_model, checkpoint_path, dataset, prompt_field='question', use_ans_sep=True, batch_size=8, max_new_tokens=512):
    model, tokenizer = get_ft_model(base_model, checkpoint_path)
    results = batched_inference(base_model, model, tokenizer, dataset, 
                                    prompt_field=prompt_field, 
                                    use_ans_sep=use_ans_sep, 
                                    batch_size=batch_size, 
                                    max_new_tokens=max_new_tokens)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results


#######################
# Main Evaluation  #
#######################

def extract_ans(text):
    return text.split('#')[-1].strip()

def eval_str_match(results, preproc_fn, verbose=False):
    """
    Evaluates model outputs for exact-match correctness.
    
    Returns:
        accuracy (float): percentage of correct answers
        correct_count (int)
        total_count (int)
    """
    correct = 0
    total = len(results)

    for i, r in enumerate(results):
        gt = preproc_fn(str(r["ground_truth"]))
        pred = preproc_fn(str(r['chat'][-1]['content']))

        if pred == gt:
            correct += 1

        if verbose and i%100==0:
            print(f"Q: {r['question']}")
            print(f"GT: {str(r['ground_truth'])} | Pred: {r['chat'][-1]['content']}")
            print("✅ Correct\n" if pred==gt else "❌ Incorrect\n")

    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def eval_safety(guard_model, tokenizer, data, batch_size=128):
    all_decoded_outputs = []
    
    # Process the data in batches
    for i in tqdm(range(0, len(data), batch_size)):
        
        # 1. Get the current batch
        batch_data = data[i : i + batch_size]
        
        # Extract chats from the batch data
        batch_chats = [x['chat'] for x in batch_data]

        # 2. Tokenize the batch
        # Note: Using guard_model.device instead of model.device
        inputs = tokenizer.apply_chat_template(
            batch_chats,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            return_dict=True,
        ).to(guard_model.device) 

        # 3. Generate outputs for the batch
        with torch.no_grad():
            outputs = guard_model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 4. Decode and store outputs
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_decoded_outputs.extend(decoded_outputs)

        del inputs
        del outputs
        torch.cuda.empty_cache()
        gc.collect()

    # 5. Calculate the final score across all collected outputs
    if not all_decoded_outputs:
        print("Warning: No data was processed.")
        return 0.0

    safe_count = sum([1 for o in all_decoded_outputs if o.split('\n\n')[-1] == 'safe'])
    total_count = len(all_decoded_outputs)

    return all_decoded_outputs, safe_count / total_count



if __name__ == "__main__":
    ### Inference
    LLAMA_PATTERN = '/ocean/projects/cis250042p/sjain13/IntrospectLoss/training_POC/results_l3_8_alpha_[0|1]/checkpoint-429'
    QWEN3_4_PATTERN = '/ocean/projects/cis250042p/sjain13/IntrospectLoss/training_POC/results_q3_4_alpha_[0|1]/checkpoint-429'

    models = {
        'meta-llama/Llama-3.1-8B-Instruct': [None] + sorted(glob(LLAMA_PATTERN)),
        'Qwen/Qwen3-4B-Instruct-2507': [None] + sorted(glob(QWEN3_4_PATTERN))
    }

    for model, cps in models.items():
        print(f"Paths for {model}:")
        for cp in cps:
            print(f"    {cp if cp is not None else 'Base'}")

    if 'gsm8k_test' not in locals():
        gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")

    if 'wgm_test' not in locals():
        wgm_test = load_dataset('allenai/wildguardmix', 'wildguardtest', split='test')

    pickle_path = '/ocean/projects/cis250042p/sjain13/IntrospectLoss/temp_pkls'
    os.makedirs(pickle_path, exist_ok=True)

    gsm_infs = {}
    if 'gsm_infs.pkl' in os.listdir(pickle_path):
        with open(os.path.join(pickle_path, 'gsm_infs.pkl'), 'rb') as f:
            gsm_infs = pickle.load(f)
    
    for base_model, checkpoint_paths in models.items():
        if base_model not in gsm_infs: gsm_infs[base_model] = {}
        for cp in checkpoint_paths:
            exp_name = cp if cp is not None else 'Base'
            if base_model in gsm_infs and exp_name in gsm_infs[base_model].keys():
                print(f"Skipping {base_model} {exp_name}, already done.")
                continue
            print("Running GSM8K inference for", base_model, cp if cp is not None else 'Base')
            gsm_infs[base_model][exp_name] = run_inf(base_model, cp, gsm8k_test, 
                                                                            prompt_field='question', 
                                                                            use_ans_sep=True, 
                                                                            batch_size=512, 
                                                                            max_new_tokens=384)
            with open(os.path.join(pickle_path, 'gsm_infs.pkl'), 'wb') as f:
                pickle.dump(gsm_infs, f)
    
    # wgm_infs = {}
    # if 'wgm_infs.pkl' in os.listdir(pickle_path):
    #     with open(os.path.join(pickle_path, 'wgm_infs.pkl'), 'rb') as f:
    #         wgm_infs = pickle.load(f)
    
    # for base_model, checkpoint_paths in models.items():
    #     if base_model not in wgm_infs: wgm_infs[base_model] = {}
    #     for cp in checkpoint_paths:
    #         exp_name = cp if cp is not None else 'Base'
    #         if base_model in wgm_infs and exp_name in wgm_infs[base_model]:
    #             print(f"Skipping {base_model} {exp_name}, already done.")
    #             continue
    #         print("Running WGM inference for", base_model, cp if cp is not None else 'Base')
    #         wgm_infs[base_model][cp if cp is not None else 'Base'] = run_inf(base_model, cp, wgm_test, 
    #                                                                         prompt_field='prompt', 
    #                                                                         use_ans_sep=False, 
    #                                                                         batch_size=384, 
    #                                                                         max_new_tokens=768)
    #         with open(os.path.join(pickle_path, 'wgm_infs.pkl'), 'wb') as f:
    #             pickle.dump(wgm_infs, f)
    
    
    ### GSM Eval
    results = {}
    for model, m_res in gsm_infs.items():
        results[model] = {}
        for exp_name, e_res in m_res.items():
            if '/' in exp_name: exp_name = exp_name.split('/')[-2]
            results[model][exp_name] = eval_str_match(e_res, extract_ans, False)
            print(f'{model:40} {exp_name:30} {results[model][exp_name]["accuracy"]:.4f}')

    ### Safety Eval 
    # g_model, g_tokenizer = get_ft_model("meta-llama/Llama-Guard-3-8B", None)
    
    # if 'wgm_results.pkl' in os.listdir(pickle_path):
    #     with open(os.path.join(pickle_path, 'wgm_results.pkl'), 'rb') as f:
    #         wgm_infs = pickle.load(f)
    # else:
    #     wgm_results = {}
    #     for model, m_res in wgm_infs.items():
    #         wgm_results[model] = {}
    #         for exp_name, e_res in m_res.items():
    #             if '/' in exp_name: exp_name = exp_name.split('/')[-2]
    #             wgm_results[model][exp_name] = eval_safety(g_model, g_tokenizer, e_res, 64)
    #     with open(os.path.join(pickle_path, 'wgm_results.pkl'), 'wb') as f:
    #         pickle.dump(wgm_results, f)

    # for model, m_res in wgm_results.items():
    #     for exp_name, e_res in m_res.items():
    #         print(f'{model:40} {exp_name:30} {e_res[1]:.4f}')
    
    # ### Cleanup
    # del gsm8k_test, wgm_test, gsm_infs, wgm_infs
    # gc.collect()
    # torch.cuda.empty_cache()
    # for f in os.listdir(pickle_path):
    #     os.remove(os.path.join(pickle_path, f))
    # os.rmdir(pickle_path)


'''
GSM8K Results:

meta-llama/Llama-3.1-8B-Instruct         Base                           0.7043
meta-llama/Llama-3.1-8B-Instruct         results_l3_8_alpha_1           0.3654
meta-llama/Llama-3.1-8B-Instruct         results_l3_8_alpha_100         0.4435
meta-llama/Llama-3.1-8B-Instruct         results_l3_8_alpha_1000        0.4314
meta-llama/Llama-3.1-8B-Instruct         results_l3_8_alpha_300         0.4503

Qwen/Qwen3-4B-Instruct-2507              Base                           0.8491
Qwen/Qwen3-4B-Instruct-2507              results_q3_4_alpha_1           0.2472
Qwen/Qwen3-4B-Instruct-2507              results_q3_4_alpha_100         0.2547
Qwen/Qwen3-4B-Instruct-2507              results_q3_4_alpha_1000        0.2699
Qwen/Qwen3-4B-Instruct-2507              results_q3_4_alpha_300         0.2259
'''