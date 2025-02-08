## BitLora
Fine-tuning large language models is crucial for specialized tasks, but traditional methods that update the full-parameter set lead to substantial computational overhead. While low-rank adapters (LoRAs) reduce the number of trainable parameters, deploying millions of fine-tuned adapters still demands significant storage resources. We propose BitLoRA, a quantization-aware training method that enables 1-bit precision LoRAs while preserving model quality. Our approach achieves comparable performance to full-precision LoRAs while reducing weight storage requirements by 12x, making large-scale deployment of fine-tuned adapters more practical.



## Training
To train the model, you can use the provided `train.py` script. You will need to specify various parameters such as model path, output directory, training data path, and LoRA configurations.

Example command to start training:

bash
bash scripts/finetune_llama3.sh


Make sure to set the appropriate variables in `scripts/finetune_llama3.sh` before running the script.

