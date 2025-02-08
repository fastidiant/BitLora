#train.py
import json
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ChainedScheduler
import argparse
import os
from tqdm import tqdm
import logging
import random
import wandb
import numpy as np
import gc


from utils import (
    setup_logging,
    load_model_and_tokenizer,
    mark_only_lora_as_trainable,
    add_lora_layers
)

from data_utils import create_dataloaders


logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA adapters")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation Split Proportion")
    parser.add_argument("--grad_acc_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number warmup steps")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    #parser.add_argument("--max_steps", type=int, required=True, help="Maximum number of training steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=80, help="Number of steps before Eval")
    parser.add_argument("--eval_delay", type=float, default=0.5, help="Percentage of steps until eval")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--scale_lr", type=float, default=1e-5, help="Learning rate for scaling parameters")
    parser.add_argument("--lora_type", type=str, help="Lora Type")
    parser.add_argument("--lora_rank", type=int, help="Lora Dimension")
    parser.add_argument("--lora_alpha", type=int, help="Lora Alpha")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--wandb_project", type=str, default="lora-training", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, help="WandB username")
    parser.add_argument("--wandb_run_name", type=str, help="WandB run name")
    return parser.parse_args()

def save_lora_adapters(model, output_dir: str, config):
    os.makedirs(output_dir, exist_ok=True)
    
    lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora" in k}
    torch.save(lora_state_dict, os.path.join(output_dir, "adapter_model.bin"))

    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    avg_loss = total_loss / len(dataloader)
    return torch.exp(torch.tensor(avg_loss)), avg_loss

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    model = add_lora_layers(
        model,
        lora_type=args.lora_type,
        target_modules=target_modules,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dtype=model.dtype
    )
    mark_only_lora_as_trainable(model)

    lora_config = {
        "lora_type": args.lora_type,
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "target_modules": target_modules,
        "base_model": args.model_path
    }


    train_dataloader, val_dataloader = create_dataloaders(
        tokenizer,
        args.train_data_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        val_split=args.val_split,
        seed=args.seed
    )

    args.eval_delay = (
        args.eval_delay
        if isinstance(args.eval_delay, int)
        else int(args.eval_delay * len(train_dataloader)/args.grad_acc_steps)
    )
    num_update_steps = len(train_dataloader) / args.grad_acc_steps
    total_steps = num_update_steps * args.num_epochs
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    beta_params = [p for n, p in model.named_parameters() if 'scale' in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if 'scale' not in n and p.requires_grad]
    #optimizer = AdamW(trainable_params, lr=args.learning_rate)
    optimizer = AdamW(
        [
            {'params': beta_params, 'lr': args.scale_lr},
            {'params': other_params, 'lr': args.learning_rate}
        ], 
    )

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - args.warmup_steps)
    scheduler = ChainedScheduler([warmup_scheduler, scheduler])

    # beta_values = {
    # name: {
    #     'value': param.item(),
    #     'grad': param.grad.item() if param.grad is not None else None
    # }
    # for name, param in model.named_parameters() 
    # if name == 'model.layers.0.self_attn.q_proj.lora_A.beta'
    # }
    # print({f"betas/{name}": values for name, values in beta_values.items()})

    global_step = 0
    best_loss = float('inf')

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(progress_bar):
            if step == 1138 or global_step == 2741 or step == 1176 or step==26:
                continue
            batch = {k:v.to(device) for k,v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_acc_steps

            loss.backward()
            epoch_loss += loss.item()
            
            # if step % 25 == 0:
            #     beta_values = {
            #         name: {
            #             'value': param.item(),
            #             'grad': param.grad.item() if param.grad is not None else None
            #         }
            #         for name, param in model.named_parameters() 
            #         if name == 'model.layers.0.self_attn.q_proj.lora_A.scale'
            #     }
            #     print({f"scale/{name}": values for name, values in beta_values.items()})
            #     weight_values = {
            #         name: {
            #             'value': param.item(),
            #             'grad': param.grad.item() if param.grad is not None else None
            #         }
            #         for name, param in model.named_parameters() 
            #         if name == 'model.layers.0.self_attn.q_proj.lora_A.weight'
            #     }
            #     print({f"weight/{name}": values for name, values in weight_values.items()})


            if (step + 1) % args.grad_acc_steps == 0:
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                torch.nn.utils.clip_grad_norm_(beta_params, max_norm=1.0)  # Smaller max_norm for beta
                torch.nn.utils.clip_grad_norm_(other_params, max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                wandb.log({
                    "train/loss": loss.item() * args.grad_acc_steps,
                    "train/lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "step": global_step
                })

                if global_step % args.eval_steps == 0 and global_step > args.eval_delay and val_dataloader:

                    perplexity, val_loss = evaluate(model, val_dataloader, device)
                    wandb.log({
                        "val:perplexity": perplexity,
                        "val:loss": val_loss,
                        "step": global_step
                    })
                    if val_loss < best_loss:
                        best_loss = val_loss
                        path = os.path.join(args.output_dir, "best_model")
                        save_lora_adapters(model, path, lora_config)
                
                if global_step % args.save_steps == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{global_step}")
                    save_lora_adapters(model, checkpoint_path, lora_config)

            progress_bar.set_postfix({
                "loss": f"{loss.item() * args.grad_acc_steps:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
        print(f"Epoch {epoch+1}/{args.num_epochs} Train loss:{epoch_loss/len(train_dataloader)}")
        checkpoint_path = os.path.join(args.output_dir, f"epoch_{epoch}")
        save_lora_adapters(model, checkpoint_path, lora_config)
        gc.collect()
        torch.cuda.empty_cache()


    save_lora_adapters(model, os.path.join(args.output_dir, "final_model"), lora_config)
    if val_dataloader:
        perplexity, final_val_loss = evaluate(model, val_dataloader, device)
        print(f"Validation perplexity: {perplexity}, Validation loss: {final_val_loss}",)
        if final_val_loss < best_loss:
            best_loss = final_val_loss
            save_lora_adapters(model, os.path.join(args.output_dir, "best_model"), lora_config)
    



def main():
    args = parse_args()
    setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)

    wandb.init( 
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args)
    )

    logger.info("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        train(args)
    finally:
        wandb.finish()
        gc.collect()
        torch.cuda.empty_cache()

    
if __name__ == "__main__":
    print("here")
    main()

