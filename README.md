# OPT Alpaca Training (WIP)

Training and inference code for [OPT](https://ai.facebook.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/) models based on the [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) training protocol.
### Usage
```
python train.py --base-model huggingface/modelname --dataset huggingface/datasetname --epochs number of epochs to train --output trained model name
```
### Example
```
python train.py --base-model facebook/opt-1.3b --dataset "psmathur/alpaca_orca" --output opt-orca-1.3b --epochs 3
```

### Merging LoRA Checkpoint
```
python merge_lora_weight.py --base-model huggingface/modelname --output mergedmodel --checkpoint my-opt-lora
```
### Example
```
python merge_lora_weight.py --base-model facebook/opt-350m --output seahorse-350m --checkpoint opt-orca-350m
```



### Outputs
```
Instruction: Tell me about alpacas.

Response: Alpacas are a species of large South American camelids. They are characterized by their long, coarse wool and their unique coat patterning. Alpacas are commonly used for their soft, warm wool, which is used in a variety of products, including clothing, blankets, and yarn. Alpacas are also used for their fiber, which is used in the production of rope and twine. Alpacas are also known for their friendly and docile nature. They are popular companions for people who enjoy the outdoors, and are often kept as pets or used as herd animals. Alpacas are also known for their unique personality traits, including their unique vocalizations and their unique way of walking. Alpacas are native to South America and are found in Argentina, Bolivia, Chile, Colombia, Ecuador, Peru, and Uruguay. They are also commonly found in other parts of the world, such as in Australia, New Zealand, and Europe. Alpacas are a popular livestock species, and are often used to help raise llamas and other types of livestock.
```

```
Instruction: Tell me about the president of Mexico in 2019.

Response: The president of Mexico in 2019 is Andres Manuel Lopez Obrador (AMLO), who took office on December 1, 2018. AMLO is a leftist politician and former mayor of Mexico City. He is known for his populist platform and his opposition to the country's political establishment. He has promised to bring about a more equal and just society and to address the problems of corruption and inequality. AMLO has also promised to address the country's problems of crime, violence, and poverty. He has also been critical of the United States' immigration policy and has pledged to protect the rights of migrants and asylum seekers. He has also promised to fight corruption and to reduce the country's dependence on foreign oil.
``` 

### Credits
- [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)
- [Manuel030/alpaca-opt](https://github.com/Manuel030/alpaca-opt)
