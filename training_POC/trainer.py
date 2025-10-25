import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Load the model and tokenizer
model_name = "Qwen/Qwen3-0.6B"


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=None, # No quantization for now
    device_map="auto",  # Automatically uses available GPUs
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Set the padding token to be the same as the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

# 2. Configure LoRA (PEFT)
# LoRA is a technique to drastically reduce the number of trainable parameters.
lora_config = LoraConfig(
    r=16,  # The dimension of the low-rank matrices
    lora_alpha=32,  # The scaling factor for the low-rank matrices
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Apply LoRA to attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare model for k-bit training and apply PEFT
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 3. Load and prepare the dataset
dataset_name = "./IntrospectLoss/training_POC/dummy_train.json"
dataset = load_dataset(dataset_name, split="train")

# 4. Set up the Training Arguments
training_args = TrainingArguments(
    output_dir="./sft-qwen3-06b-results",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit", # Memory-efficient optimizer
    logging_steps=25,
    learning_rate=2e-4,
    bf16=True, # Use bfloat16 for training if your GPU supports it
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="tensorboard",
)

# 5. Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text", # Use the 'text' column from the dataset
    # formatting_func=format_instruction, # Uncomment if you want to use the custom function
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

# 6. Start Training
print("Starting the training process... 🚀")
trainer.train()
print("Training finished! 🎉")

# 7. Save the fine-tuned model (adapter)
output_model_dir = "./my_fine_tuned_model"
trainer.save_model(output_model_dir)
print(f"Model adapter saved to {output_model_dir}")
