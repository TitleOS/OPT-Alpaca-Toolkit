from peft import PeftModel
from transformers import OPTForCausalLM, AutoTokenizer
import argparse

parser = argparse.ArgumentParser(description='Lora Huggingface Checkpoint Combiner')
parser.add_argument('--base-model', type=str, help='The base model used for finetuning')
parser.add_argument('--checkpoint', type=str, help='The name of the LoRA checkpoint')
parser.add_argument('--output', type=str, help='The name of the merged model and checkpoint.')
args = parser.parse_args()

if args.base_model:
    BASE_MODEL = args.base_model
    print("Using base model:", BASE_MODEL)
else:
    BASE_MODEL = "facebook/opt-1.3b"
    print("No model provided, using default: facebook/opt-1.3b")
if args.checkpoint:
    COMBINED_MODEL = args.checkpoint
    print("Loading LoRA checkpoint", COMBINED_MODEL)
else:
    COMBINED_MODEL = "alpaca-opt-1.3b"
    print("No model name provided, saving as alpaca-opt-1.3b")
if args.output:
    OUTPUT_MODEL = args.output
    print("Saving model as", OUTPUT_MODEL)
else:
    OUTPUT_MODEL = "alpaca-opt-1.3b"
    print("No model name provided, saving as alpaca-opt-1.3b")


base_model = OPTForCausalLM.from_pretrained(
    BASE_MODEL,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

lora_model = PeftModel.from_pretrained(base_model, COMBINED_MODEL)

lora_model.train(False)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

OPTForCausalLM.save_pretrained(
    base_model, OUTPUT_MODEL + "_ckpt", state_dict=deloreanized_sd, max_shard_size="400MB"
)
print("Saved HuggingFace model to", OUTPUT_MODEL + "_ckpt")

# now load as usual
#mixin_model = OPTForCausalLM.from_pretrained(
#    "./hf_ckpt",
#).push_to_hub("alpaca-opt-6.7b")
#tokenizer.push_to_hub("alpaca-opt-6.7b")
