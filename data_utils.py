import torch
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, DataLoader, RandomSampler, SequentialSampler, random_split
from transformers import AutoTokenizer

IGNORE_INDEX = -100
BASE_PROMPT = """<s> Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Response:
"""

@dataclass
class TrainingExample:
    instruction: str
    response: str

class InstructionDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 1048,
        set_fixed_length: Optional[int] = None
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.set_fixed_length = set_fixed_length
        self.examples = self._load_and_preprocess_data(data_path)

    def _load_and_preprocess_data(self, data_path: str) -> List[TrainingExample]:
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        length = 0
        examples = []
        for example in raw_data:
            instruction = self._extract_field(example, ["input"])
            response = self._extract_field(example, ["output", "response", "answer"])
            
            examples.append(TrainingExample(
                instruction=BASE_PROMPT.format(instruction=instruction),
                response=f"{self._clean_response(response)}{self.tokenizer.eos_token}"
            ))
            length += 1
            if self.set_fixed_length and length > self.set_fixed_length:
                break

        return examples

    def _extract_field(self, example: Dict, keys: List[str]) -> str:
        for key in keys:
            if key in example and example[key]:
                return str(example[key]).strip()
        raise ValueError(f"Missing required field. Tried keys: {keys} in example: {example}")

    def _clean_response(self, response: str) -> str:
        return response.replace("</s>", "").replace(self.tokenizer.eos_token, "").strip()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        prompt_tokens = self.tokenizer(
            example.instruction,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids[0]

        response_tokens = self.tokenizer(
            example.response,
            truncation=True,
            max_length=self.max_length - len(prompt_tokens),
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids[0]

        full_tokens = torch.cat([prompt_tokens, response_tokens])
        labels = torch.full_like(full_tokens, IGNORE_INDEX)
        labels[len(prompt_tokens):] = full_tokens[len(prompt_tokens):]

        return {
            "input_ids": full_tokens,
            "labels": labels
        }

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]

        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )

        return {
            "input_ids": padded_inputs,
            "labels": padded_labels,
            "attention_mask": (padded_inputs != self.tokenizer.pad_token_id).long()
        }


def create_dataloaders(
    tokenizer: AutoTokenizer,
    train_data_path: str,
    batch_size: int, 
    max_length: int= 1024,
    val_data_path: Optional[str] = None,
    val_split: float = 0,
    seed: int = 42,
    num_workers: int = 4,
) -> Tuple[DataLoader, Optional[DataLoader]]:

    train_dataset = InstructionDataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    if val_data_path:
        val_dataset = InstructionDataset(
            data_path=val_data_path,
            tokenizer=tokenizer,
            max_length=max_length
        )
    elif val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
    else:
        val_dataset = None

    train_sampler = RandomSampler(train_dataset, generator=torch.Generator().manual_seed(seed))
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=train_dataset.dataset.collate_fn,
        **loader_args
    )
    val_loader = None

    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            collate_fn=val_dataset.dataset.collate_fn,
            **loader_args
        )
    return train_loader, val_loader