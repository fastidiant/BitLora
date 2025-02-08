import argparse
import json
import os
import re
import torch
import gc

from data_utils import BASE_PROMPT
from utils import load_lora_model_and_tokenizer, generate_completions
from transformers import GenerationConfig

def extract_answer_number(dataset_name, sentence: str) -> float:
    if dataset_name in ["MultiArith", "single_eq", "gsm8k", "SVAMP", "MAWPS"]:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')

    return pred_answer


def extract_answer_letter(args, sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[0]
    else:
        return ''

def get_prompts_response(data_path):
    with open(data_path, 'r') as f:  
        dataset = json.load(f)
    prompts = [BASE_PROMPT.format(instruction=example['question']) for example in dataset]
    correct_answer = [str(example["answer"]) for example in dataset]
    return prompts, correct_answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to model", required=True)
    parser.add_argument("--lora_path", type=str, help="Path to lora adapters", required=True)
    parser.add_argument("--dataset_path", type=str, help="Path to dataset", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    dataset_name = os.path.basename(args.dataset_path).split('.')[0]

    prompts, correct_answer = get_prompts_response(args.dataset_path)

    model, tokenizer = load_lora_model_and_tokenizer(args.model_path, args.lora_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    generation_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        bos_token_id=model.config.bos_token_id,
        num_return_sequences=1,
        repetition_penalty=1.2
    )

    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=args.batch_size,
        generation_config = generation_config
    )
    save_outputs = []
    correct = 0
    eta = 0.001
    for correct_output, output in zip(correct_answer, outputs):
        output = output['response']
        if dataset_name.lower() == "aqua":
            predict = extract_answer_letter(dataset_name, output)
            if correct_output.lower() == predict.lower():
                correct += 1
        elif dataset_name.lower() == "gsm8k":
            predict = extract_answer_letter(dataset_name, output)
            if predict.isdigit():
                correct = correct_output.split(' ')[-1]
                if abs(float(correct) - float(predict)) < eta:
                    correct += 1
        else:
            predict = extract_answer_number(dataset_name, output)
            if dataset_name == "single_eq":
                correct_output = float(correct_output.strip('[]'))
            if abs(float(correct_output) - predict) < eta:
                correct += 1
        
        save_outputs.append({
            'raw_output': output,
            'prediction': predict,
            'correct_output': correct_output,
        })
    
    weighted_acc = correct/len(prompts)
    print(f"{dataset_name} Accuracy: {weighted_acc * 100:.1f}%, Total: {len(prompts)}")

    print(f"\n{dataset_name} Results:")
    print(f"Accuracy: {weighted_acc * 100:.1f}%")
    print(f"Total samples: {len(prompts)}")
    
    with open(os.path.join(args.output_dir, f"{dataset_name}_predictions.jsonl"), "w") as f:
        for example in save_outputs:
            f.write(json.dumps(example) + "\n")

    del model
    gc.collect()
    torch.cuda.empty_cache()