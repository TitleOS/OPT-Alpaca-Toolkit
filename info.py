import argparse
import transformers
from transformers import AutoTokenizer, OPTForCausalLM

def main():
    parser = argparse.ArgumentParser(description='Information viewer about LLMs')
    parser.add_argument('--model', type=str, help='The LLM to query')
    args = parser.parse_args()
    if args.model:
        get_info(args.model)
    else:
        print("No model provided, please provide a model to query")


def get_info(model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = OPTForCausalLM.from_pretrained(model, device_map="auto")
    print(f"Model: {model}")
    print(f"Tokenizer: {tokenizer}")
    print(f"Vocab Size: {tokenizer.vocab_size}")
    print(f"Max Length: {tokenizer.model_max_length}")
    print(f"Padding Token: {tokenizer.pad_token}")
    print(f"Special Tokens: {tokenizer.special_tokens_map}")
    print(f"Model Config: {model.config}")
    print(f"Model Parameters: {model.num_parameters()}")
    print(f"Model Layers: {model.config.num_hidden_layers}")
    print(f"Model Hidden Size: {model.config.hidden_size}")
    print(f"Model Attention Heads: {model.config.num_attention_heads}")
    print(f"Model Dropout: {model.config.attention_probs_dropout_prob}")
    print(f"Model Activation: {model.config.hidden_act}")
    print(f"Model Type: {model.config.model_type}")
    print(f"Model Embedding Size: {model.config.hidden_size}")
    print(f"Model Layer Norm: {model.config.layer_norm_eps}")
    print(f"Model Hidden Dropout: {model.config.hidden_dropout_prob}")
    print(f"Model Intermediate Size: {model.config.intermediate_size}")
    print(f"Model Intermediate Activation: {model.config.hidden_act}")
    print(f"Model Output Size: {model.config.hidden_size}")
    print(f"Model Output Activation: {model.config.hidden_act}")
    print(f"Model Pooler Activation: {model.config.hidden_act}")
    print(f"Model Pooler Size: {model.config.hidden_size}")
    print(f"Model Pooler Dropout: {model.config.hidden_dropout_prob}")
    print(f"Model Pooler Layer Norm: {model.config.layer_norm_eps}")

if __name__ == "__main__":
    main()