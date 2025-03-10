# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import itertools
import math
import os
import tiktoken
import torch
from previous_chapters import GPTModel, create_dataloader_v1


# Define a grid of hyperparameters to search over
HPARAM_GRID = {
    "batch_size": [2, 4],
    "drop_rate": [0.0, 0.1, 0.2],
    "warmup_iters": [10, 20, 30],
    "weight_decay": [0.1, 0.01, 0.0],
    "peak_lr": [0.0001, 0.0005, 0.001, 0.005],
    "initial_lr": [0.00005, 0.0001],
    "min_lr": [0.00005, 0.00001, 0.0001],
    "n_epochs": [5, 10, 15, 20],
}

"""
- data_loader: 数据加载器，用于迭代获取数据
- model: 要评估的模型
- device: 计算设备（CPU/GPU）
- num_batches: 可选参数，指定要计算的批次数
"""
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    logits = model(input_batch)
    """
    将logits reshape为二维张量，-1表示自动计算该维度大小，
    logits.size(-1)获取最后一个维度的大小（通常是类别数量）。
    """
    logits = logits.view(-1, logits.size(-1))
    """
    target_batch.view(-1) 将target_batch展平为一维张量，
    通常用于将目标标签展平为一维，以便与logits的维度匹配。
    """
    loss = torch.nn.functional.cross_entropy(logits, target_batch.view(-1))
    return loss


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter,
                encoded_start_context, tokenizer, warmup_iters=10,
                initial_lr=3e-05, min_lr=1e-6):
    """
    global_step: 全局步数，用于跟踪训练过程中的迭代次数
    max_lr: 学习率，用于跟踪训练过程中的最大学习率
    total_training_iters: 总训练迭代次数，用于计算学习率的变化
    lr_increment: 学习率增量，用于计算学习率的变化
    """
    global_step = 0

    max_lr = optimizer.param_groups[0]["lr"]

    # Calculate total number of iterations
    # 计算总训练迭代次数 = 每个epoch的batch数 * epoch数
    total_training_iters = len(train_loader) * n_epochs

    # Calculate the learning rate increment at each step during warmup
    # 计算warmup阶段每个step的学习率增量
    lr_increment = (optimizer.param_groups[0]["lr"] - initial_lr) / warmup_iters

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            # Increment the global step at the beginning of the iteration
            global_step += 1

            # Warmup: adjust learning rate linearly
            # 如果全局步数小于或等于warmup_iters，则线性调整学习率
            if global_step <= warmup_iters:
                lr = initial_lr + global_step * lr_increment
            # Cosine annealing phase，如果全局步数大于warmup_iters，则使用余弦退火调整学习率
            else:
                progress = (global_step - warmup_iters) / (total_training_iters - warmup_iters)
                lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            # Apply the calculated learning rate
            # 将计算的学习率应用到优化器的所有参数组
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # Apply gradient clipping
            # 如果全局步数大于或等于warmup_iters，则应用梯度裁剪
            if global_step >= warmup_iters:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

    # 训练结束后评估模型
    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)

    return train_loss, val_loss


if __name__ == "__main__":

    # Generate all combinations of hyperparameters
    # 使用itertools.product生成所有超参数组合
    # HPARAM_GRID是一个字典，values()返回所有超参数值的列表
    # 结果是一个包含所有可能组合的列表
    hyperparameter_combinations = list(itertools.product(*HPARAM_GRID.values()))
    total_combinations = len(hyperparameter_combinations)
    print(f"Total hyperparameter configurations: {total_combinations}")

    # Placeholder for the best loss and best hyperparameters
    best_val_loss = float('inf')
    best_hparams = {}

    # 获取当前脚本的绝对路径
    script_path = os.path.abspath(__file__)
    # 获取脚本所在的目录
    script_dir = os.path.dirname(script_path)
    # 打开文件并读取内容
    with open(os.path.join(script_dir, "the-verdict.txt"), "r", encoding="utf-8") as file:
        text_data = file.read()
    # 使用tiktoken库加载GPT-2的tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    # 选择计算设备，优先使用GPU，如果GPU不可用则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_ratio = 0.95
    split_idx = int(train_ratio * len(text_data))

    torch.manual_seed(123)

    interrupted = False
    current_config = 0
    # 遍历所有超参数组合
    for combination in hyperparameter_combinations:

        try:
            current_config += 1
            print(f"Evaluating configuration {current_config} of {total_combinations}")

            # Unpack the current combination of hyperparameters
            HPARAM_CONFIG = dict(zip(HPARAM_GRID.keys(), combination))

            GPT_CONFIG_124M = {
                "vocab_size": 50257,    # Vocabulary size
                "context_length": 256,  # Context length -- shortened from original 1024 tokens
                "emb_dim": 768,         # Embedding dimension
                "n_heads": 12,          # Number of attention heads
                "n_layers": 12,         # Number of layers
                "drop_rate": HPARAM_CONFIG["drop_rate"],
                "qkv_bias": False,     # Query-Key-Value bias
            }

            torch.manual_seed(123)
            train_loader = create_dataloader_v1(
                text_data[:split_idx],
                batch_size=HPARAM_CONFIG["batch_size"],
                max_length=GPT_CONFIG_124M["context_length"],
                stride=GPT_CONFIG_124M["context_length"],
                drop_last=True,
                shuffle=True,
                num_workers=0
            )

            val_loader = create_dataloader_v1(
                text_data[split_idx:],
                batch_size=HPARAM_CONFIG["batch_size"],
                max_length=GPT_CONFIG_124M["context_length"],
                stride=GPT_CONFIG_124M["context_length"],
                drop_last=False,
                shuffle=False,
                num_workers=0
            )

            model = GPTModel(GPT_CONFIG_124M)
            model.to(device)
            # 使用AdamW优化器，学习率为peak_lr，权重衰减为weight_decay
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=HPARAM_CONFIG["peak_lr"],
                weight_decay=HPARAM_CONFIG["weight_decay"]
            )

            # 将"Nevertheless"编码为token ID
            encoded_start_context = tokenizer.encode("Nevertheless")
            # 将编码后的token ID转换为张量，并添加一个维度
            encoded_tensor = torch.tensor(encoded_start_context).unsqueeze(0)

            train_loss, val_loss = train_model(
                model, train_loader, val_loader, optimizer, device,
                n_epochs=HPARAM_CONFIG["n_epochs"],
                eval_freq=5, eval_iter=1,
                encoded_start_context=encoded_tensor,
                tokenizer=tokenizer,
                warmup_iters=HPARAM_CONFIG["warmup_iters"],
                initial_lr=HPARAM_CONFIG["initial_lr"],
                min_lr=HPARAM_CONFIG["min_lr"]
            )

            # 如果验证损失小于当前最佳验证损失，则更新最佳验证损失和最佳训练损失
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_hparams = HPARAM_CONFIG

        except KeyboardInterrupt:
            print("Hyperparameter search completed.")
            print(f"Best hyperparameters: {best_hparams}")
            print(f"Best Val loss: {best_val_loss} | Training loss {train_loss}")
            interrupted = True
            break

    if not interrupted:
        print("Hyperparameter search completed.")
        print(f"Best hyperparameters: {best_hparams}")
        print(f"Best Val loss: {best_val_loss} | Training loss {train_loss}")
