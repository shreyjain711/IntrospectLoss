import gc
import os, sys
import wandb
from typing import Dict

proj_path = '/ocean/projects/cis250042p/sjain13/IntrospectLoss'
if proj_path not in sys.path: 
    sys.path.append(proj_path)
from mlp_training.mlp import MLP

os.environ['HF_HOME'] = '/ocean/projects/cis250042p/sjain13'
os.environ["WANDB_LOG_MODEL"] = "false"

if wandb.run is not None:
    print(f"Active run '{wandb.run.name}' found. Finishing it...")
    wandb.finish()

wandb.login(key='c7d77f7080d30d032dcac5a88bc6e3ea18058724')

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


def get_model(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=None,  # No quantization for now
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
        local_files_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class MySFTTrainer(SFTTrainer):
    def __init__(self, model_name, alpha, *args, **kwargs):
        # First, call the parent class's constructor
        super().__init__(*args, **kwargs)
        # Now, initialize your custom attribute
        self.frozen_mlp = None
        self.load_mlp(model_name)
        self.alpha = alpha

        # logging changes
        self._total_base_loss_sum = 0.0
        self._total_mlp_loss_sum = 0.0
        self._loss_step_count = 0

    def load_mlp(self, model_name):
        if self.frozen_mlp is None:
            print("DBG", "Loading MLP...")

        TOTAL_MLP_LAYERS = 32 if model_name == 'l3_8' else 36
        MLP_DIMS = [(2560 if model_name == 'q3_4' else 4096), 1024, 512, 1]
        self.frozen_mlp = MLP(TOTAL_MLP_LAYERS+1, MLP_DIMS)

        state = torch.load(f"mlp_training/outputs/{model_name}_mlp_mode_lin_agt.pth")
        self.frozen_mlp.load_state_dict(state)
        self.frozen_mlp.eval()
        for param in self.frozen_mlp.parameters():
            param.requires_grad = False

        self.frozen_mlp.eval()
        
    import torch

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        if self.frozen_mlp is None:
            self.load_mlp()
        
        if getattr(self, "_mlp_device_moved", False) is False:
            mlp_device = next(model.parameters()).device
            self.frozen_mlp.to(mlp_device)
            self._mlp_device_moved = True
        
        inputs["output_hidden_states"] = True
        outputs = model(**inputs)
        base_model_loss = outputs.loss
        
        if self.alpha == 0:
            return (base_model_loss, outputs) if return_outputs else base_model_loss

        labels = inputs.get("labels")
        hidden_states = getattr(outputs, "hidden_states", None)
        
        scaled_mlp_loss = torch.tensor(0.0, device=base_model_loss.device, dtype=base_model_loss.dtype)

        if hidden_states is not None:
            mask = labels != -100

            hs_stack = torch.stack(hidden_states, dim=0) # shape: (num_layers, batch, seq, dim)
            hs_stack = hs_stack.permute(1, 2, 0, 3) # Target shape: (Batch, Seq, Layers, Dim)
            
            batch_size, seq_len, num_layers, dim = hs_stack.shape

            # gets: ((Batch*Seq), Layers, Dim)
            hs_reshaped = hs_stack.reshape(batch_size * seq_len, num_layers, dim)
            mlp_output_flat = self.frozen_mlp(hs_reshaped)
            mlp_predictions = mlp_output_flat.squeeze(-1).view(batch_size, seq_len)
            mlp_loss_per_token = 1 - mlp_predictions.squeeze(-1)
            active_loss = mlp_loss_per_token * mask.float()
            
            # Avoid division by zero
            num_active_elements = mask.sum()
            if num_active_elements > 0:
                scaled_mlp_loss = active_loss.sum() / num_active_elements

        loss = base_model_loss + (self.alpha * scaled_mlp_loss)

        if self.is_in_train:
            self._total_base_loss_sum += base_model_loss.detach().item()
            # Ensure we detach to prevent memory leaks in logging
            self._total_mlp_loss_sum += scaled_mlp_loss.detach().item()
            self._loss_step_count += 1

        return (loss, outputs) if return_outputs else loss
    
    def compute_loss_v0(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        if self.frozen_mlp is None:
            self.load_mlp()
        if getattr(self, "_mlp_device_moved", False) is False:
            mlp_device = next(model.parameters()).device
            self.frozen_mlp.to(mlp_device)
            self._mlp_device_moved = True
        
        inputs["output_hidden_states"] = True
        outputs = model(**inputs)
        base_model_loss = outputs.loss
        # print("DBG", "Loss before MLP:", loss.item())
        
        labels = inputs.get("labels")
        hidden_states = getattr(outputs, "hidden_states", None)
        scaled_mlp_loss = torch.tensor(0.0) # Default to 0

        if hidden_states is not None:
            mask = labels != -100
            total_mlp_loss = 0
            for l in range(hidden_states[0].shape[1]):
                if any(mask[:, l]):    
                    hs = torch.stack([x[:,l,:] for x in hidden_states], dim=1)#, ).transpose(0,1)
                    mlp_loss = 1 - self.frozen_mlp(hs).squeeze(-1) # mlp predicts 1 for safe, so 1-pred = 1 for unsafe
                    mlp_loss = mlp_loss * mask[:, l].float()
                    total_mlp_loss += mlp_loss.sum()

            scaled_mlp_loss = (total_mlp_loss / mask.sum())
        
        loss = base_model_loss + (self.alpha * scaled_mlp_loss)
        # NEW: Accumulate values for logging
        # We check is_in_train to avoid accumulating during evaluation
        if self.is_in_train:
            self._total_base_loss_sum += base_model_loss.detach().item()
            self._total_mlp_loss_sum += scaled_mlp_loss.detach().item()
            self._loss_step_count += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Log metrics to W&B. The 'logs' dict comes from the Trainer
        and already contains the averaged 'loss' (our combined loss).
        """
        # Check if we have accumulated any new loss data
        if self._loss_step_count > 0:
            # Calculate the average of our custom losses
            avg_base_model_loss = self._total_base_loss_sum / self._loss_step_count
            avg_mlp_loss = self._total_mlp_loss_sum / self._loss_step_count
            
            # Add them to the logs dictionary
            logs["base_model_loss"] = avg_base_model_loss
            logs["scaled_mlp_loss"] = avg_mlp_loss
            
            # Reset the accumulators
            self._total_base_loss_sum = 0.0
            self._total_mlp_loss_sum = 0.0
            self._loss_step_count = 0

        # Call the parent's log method to handle the actual logging
        # (which will send everything in 'logs' to W&B)
        super().log(logs)


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj", 
        "up_proj", 
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


def main(model_name: str, dataset_name="training_POC/poc_train.json", alpha=[1, 100, 300, 1000], device_map="auto"):
    dataset = load_dataset('json', data_files=dataset_name, split="train")

    LLM = ({
        "q3_4": "Qwen/Qwen3-4B-Instruct-2507", 
        "q3_8": "Qwen/Qwen3-8B", 
        "l3_8": "meta-llama/Llama-3.1-8B-Instruct"
        }).get(model_name)



    for a in alpha:
        model, tokenizer = get_model(model_name=LLM)
        model = get_peft_model(model, lora_config)

        response_template = "<|start_header_id|>assistant<|end_header_id|>" if model_name == "l3_8" else "<|im_start|>assistant\n"
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template, 
            tokenizer=tokenizer
        )
        
        training_args = TrainingArguments(
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            optim="paged_adamw_8bit", # Memory-efficient optimizer
            logging_steps=5,
            learning_rate=2e-4,
            bf16=False, # Use bfloat16 for training if your GPU supports it
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            # report_to="tensorboard",
            report_to="wandb",
            logging_strategy="steps",
            save_total_limit=1,
            output_dir=f"training_POC/results_{model_name}_alpha_{a}",      # Unique output dir
            run_name=f"run_{model_name}_alpha_{a}"
        )

        def formatting_func(data):
            # Create the chat structure that the tokenizer's chat template expects
            chats = []
            for i in range(len(data['prompt'])): 
                chats.append([{"role": "user", "content": data["prompt"][i]}, {"role": "assistant", "content": data["output"][i]}])
            return tokenizer.apply_chat_template(chats, tokenize=False)
        
        trainer = MySFTTrainer(
            model_name=model_name,
            alpha=a,
            model=model,
            train_dataset=dataset,
            peft_config=lora_config,
            formatting_func=formatting_func,
            data_collator=data_collator,
            max_seq_length=1024,
            tokenizer=tokenizer,
            args=training_args,
        )

        print(f"Training... logs and model will be saved to: {training_args.output_dir}")
        trainer.train()

        print(f"Run for alpha={a} complete. Model saved to {training_args.output_dir}")
        
        # Explicitly end the W&B run
        wandb.finish()

        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    print("--- All training runs complete ---")


if __name__ == "__main__":
    # main("q3_4", alpha=[0, 1])
    # main("l3_8", alpha=[0, 1])
    main("q3_4", alpha=[100, 300, 1000])
    main("l3_8", alpha=[100, 300, 1000])