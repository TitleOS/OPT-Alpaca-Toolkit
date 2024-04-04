import argparse
import torch
import torch.nn as nn
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, OPTForCausalLM, GPT2Tokenizer
from peft import LoraConfig, get_peft_model

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--base-model', type=str, help='Set Base Model')
parser.add_argument('--dataset', type=str, help='Set Data Path')
parser.add_argument('--output', type=str, help='Set the output model path')
parser.add_argument('--learning_rate', type=str, help="Set the learning rate, default is 1e-3.", default=1e-3)
group = parser.add_mutually_exclusive_group()
group.add_argument('--epochs', type=int, help='Set the number of epochs', default=3)
group.add_argument('--steps', type=int, help='Set the number of steps', default=10000)
args = parser.parse_args()

if args.base_model:
    BASE_MODEL = args.base_model
    print(f"Using model: {BASE_MODEL}")
else:
    BASE_MODEL = "facebook/opt-1.3b"
    print("No model provided, using default: facebook/opt-1.3b")
if args.dataset:
    DATA_PATH = args.dataset
    print(f"Using dataset: {DATA_PATH}")
else:
    DATA_PATH = "alpaca_data_small.json"
    print("No data path provided, using included alpaca_data_small.json")
if args.output:
    OUTPUT_PATH = args.output
    print(f"Using output path: {OUTPUT_PATH}")
else:
    OUTPUT_PATH = "mymodel-finetuned"
    print("No output path provided, defaulting to mymodel-finetuned")

MICRO_BATCH_SIZE = 4
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
#LEARNING_RATE = 1e-3
LEARNING_RATE = args.learning_rate
CUTOFF_LEN = 512
LORA_R = 256
LORA_ALPHA = 256
LORA_DROPOUT = 0.05
USE_FP16 = True
USE_BF16 = False
EPOCHS = args.epochs
STEP_COUNT = args.steps

if torch.cuda.is_available():
    print("Torch & Cuda Detected")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f'GPU Name [{i}]: ', torch.cuda.get_device_name(i))

amp_supported = torch.cuda.is_available() and hasattr(torch.cuda, "amp")

if amp_supported:
     print(f"AMP Supported: {amp_supported}")
     bfloat16_supported = torch.cuda.is_bf16_supported()
     print(f"BFLOAT16 Supported: {bfloat16_supported}")
     if bfloat16_supported:
          USE_FP16 = False
          USE_BF16 = True

model = OPTForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto"
    )
tokenizer = GPT2Tokenizer.from_pretrained(
    BASE_MODEL,
    model_max_length=CUTOFF_LEN,
    padding_side="right",
    use_fast=False,
)
tokenizer.save_pretrained(OUTPUT_PATH)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

try:
        # Try to load the dataset from local directory
        data = load_dataset(DATA_PATH, download_mode='reuse_cache_if_exists')
except FileNotFoundError:
        # If not found locally, download the dataset from Hugging Face
        print(f"Dataset {DATA_PATH} not found locally. Downloading from Hugging Face...")
        data = load_dataset(DATA_PATH, download_mode='force_redownload')


def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Response:
{data_point["output"]}"""


data = data.shuffle().map(
    lambda data_point: tokenizer(
        generate_prompt(data_point),
        padding="longest",
        max_length=CUTOFF_LEN,
        truncation=True,
    )
)

args_dict = {
    'per_device_train_batch_size': MICRO_BATCH_SIZE,
    'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
    'warmup_steps': 100,
    'learning_rate': LEARNING_RATE,
    'fp16': USE_FP16,
    'bf16': USE_BF16,
    'logging_steps': 10,
    'output_dir': OUTPUT_PATH,
    'save_strategy': "epoch",
    'save_total_limit': 3,
}

if EPOCHS is not None:
    args_dict['num_train_epochs'] = EPOCHS
else:
    args_dict['max_steps'] = STEP_COUNT

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(**args_dict),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False
trainer.train(resume_from_checkpoint=False)

model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
