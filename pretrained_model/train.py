#%%
# 训练条件式对抗扰动生成器：
# 固定一个已经加载好的 VQA/多模态模型，只更新 AdvGANPerturbationGenerator，
# 让它根据“原图 + 目标文本嵌入 + 扰动强度 eps”生成图像扰动，使模型更倾向输出 target_text。
import argparse
import csv
import hashlib
import importlib
import json
import os
import sys
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# 为了能从 pretrained_model 子目录导入项目根目录下的 utils/models。
# 必须放在 perturb_generator 等本地模块导入之前，否则它们内部导入 utils 会失败。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from perturb_generator import AdvGANPerturbationGenerator, get_attack_frame

from utils.attack_tool import (
    add_extra_args,
    get_available_gpus,
    get_intended_token_ids,
    load_datasets,
    load_model,
    seed_everything,
)
from utils.eval_tools import postprocess_generation




GENERIC_PROMPT_POOL = [
    # 通用 prompt 池：训练时随机抽取若干问题，降低生成器只适配单一问法的风险。
    "Describe this image.",
    # "What is shown in this image?",
    # "What can you see in the image?",
    # "Give a short caption for this image.",
    # "What is the main object in the image?",
    # "What is happening in this image?",
    # "What is the scene about?",
    # "What objects are visible in the image?",
    # "What is the person doing now?",
    # "Identify the primary visual content of this image.",
    # "Summarize the visible scene briefly.",
    # "What kind of place or scene is depicted?",
    # "What is the central subject in the picture?",
    # "Which visible elements stand out in the image?",
]


def chunked_list(indices, batch_size: int, shuffle: bool = True):
    # 将样本索引切成 mini-batch；训练阶段会在每个 epoch 内重新打乱。
    indices = list(indices)
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def prepare_batch_images(eval_model, items, model_name: str, device: str):
    # EvalModel 内部负责把 PIL/原始图像整理成各模型需要的视觉输入格式。
    item_images = [[item["image"]] for item in items]
    return eval_model._prepare_images_no_normalize(item_images).to(device)


def load_target_pool(target_file: str, fallback_target: str):
    # target_file 每行一个目标文本；如果不提供，则退回单目标 target_text。
    if not target_file:
        return [fallback_target]
    path = Path(target_file)
    if not path.exists() and not path.is_absolute():
        # 相对路径优先按当前脚本所在目录解析，便于把目标文本文件放在 pretrained_mode 下。
        path = Path(__file__).resolve().parent / target_file
    targets = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            # 空行和以 # 开头的注释行会被忽略。
            target = line.strip()
            if target and not target.startswith("#"):
                targets.append(target)
    if not targets:
        raise ValueError(f"No valid target text found in target file: {target_file}")
    return targets


def encode_target_embeddings(clip_tokenizer, clip_text_model, target_texts: list[str], device: str):
    # 用 CLIP 文本编码器批量编码所有攻击目标，后续按 target_idx 取出对应条件向量。
    enc = clip_tokenizer(
        target_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        # 目标嵌入只作为条件，不参与反向传播；归一化后便于生成器稳定使用。
        output = clip_text_model(**enc)
        embedding = output.pooler_output
        embedding = torch.nn.functional.normalize(embedding, dim=-1)
    return embedding


def build_target_schedule(args, target_pool_size: int, num_steps: int, is_train: bool):
    # 为当前 epoch 预先生成 target 索引表，避免纯随机采样导致某些目标被频繁抽中、某些目标长期缺席。
    if target_pool_size <= 0:
        raise ValueError("target_pool_size must be positive")
    if num_steps <= 0:
        return []

    if is_train:
        # 训练阶段：把所有目标索引重复到足够长度后整体打乱，实现“随机但更均匀”的覆盖。
        repeats = (num_steps + target_pool_size - 1) // target_pool_size
        schedule = list(range(target_pool_size)) * repeats
        random.shuffle(schedule)
        return schedule[:num_steps]

    # 验证阶段：固定轮换，保证不同 epoch 的验证目标顺序一致，便于比较损失变化。
    # val_targets <= 0 表示验证时使用全部目标；否则只使用目标池前 val_targets 个。
    val_targets = target_pool_size if args.val_targets <= 0 else min(args.val_targets, target_pool_size)
    return [step % val_targets for step in range(num_steps)]


def build_prompt_inputs(args, eval_model, prompt: str, target: str, batch_size: int, device: str):
    # 构造语言模型侧输入，并把 labels 设置为只监督 target token。
    tokenizer = eval_model.tokenizer
    if args.model_name in ["blip2", "instructblip"]:
        # BLIP2/InstructBLIP 采用 Question/Answer 模板。
        text = f"Question:{prompt} Answer: {target}"
    else:
        # OpenFlamingo 复用模型封装里的 VQA prompt 模板。
        text = eval_model.get_vqa_prompt(question=prompt) + " " + target
    encodings = tokenizer(
        [text] * batch_size,
        padding="longest",
        truncation=True,
        return_tensors="pt",
        max_length=2000,
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    target_id = tokenizer.encode(target, add_special_tokens=False)
    labels = get_intended_token_ids(input_ids, target_id)
    qformer_input_ids = None
    qformer_attention_mask = None
    if args.model_name == "instructblip":
        # InstructBLIP 额外需要给 Q-Former 准备一份文本输入。
        qformer_enc = eval_model.qformer_tokenizer(
            [text] * batch_size,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        ).to(device)
        qformer_input_ids = qformer_enc["input_ids"]
        qformer_attention_mask = qformer_enc["attention_mask"]
    return input_ids, attention_mask, labels, qformer_input_ids, qformer_attention_mask


def apply_delta(input_x_original: torch.Tensor, delta: torch.Tensor, model_name: str):
    # 把生成器输出的扰动加到模型视觉输入上，并裁剪到合法像素范围。
    input_x = input_x_original.clone()
    if model_name == "open_flamingo":
        # OpenFlamingo 的视觉输入通常带有图像/帧维度，这里只攻击最后一张图的第 0 帧。
        input_x[:, -1, 0] = torch.clamp(input_x[:, -1, 0] + delta, 0, 1)
    else:
        input_x = torch.clamp(input_x + delta, 0, 1)
    return input_x


def target_model_loss(
    args,
    eval_model,
    input_x_adv,
    input_ids,
    attention_mask,
    labels,
    qformer_input_ids=None,
    qformer_attention_mask=None,
):
    # 计算“让模型输出目标文本”的交叉熵损失；被攻击模型参数已冻结，但梯度会回传到图像扰动。
    if args.model_name == "open_flamingo":
        # 语言 token embedding detach，避免梯度流向目标模型的词嵌入层。
        inputs_embeds = eval_model.model.lang_encoder.get_input_embeddings()(input_ids).detach()
        return eval_model.model(
            inputs_embeds=inputs_embeds,
            lang_x=input_ids,
            vision_x=input_x_adv,
            attention_mask=attention_mask,
            labels=labels,
        )[0]

    if args.model_name == "blip2":
        return eval_model.model(
            input_ids=input_ids,
            pixel_values=input_x_adv,
            attention_mask=attention_mask,
            labels=labels,
            normalize_vision_input=True,
        )[0]
    if args.model_name == "instructblip":
        return eval_model.model(
            input_ids=input_ids,
            pixel_values=input_x_adv,
            attention_mask=attention_mask,
            labels=labels,
            normalize_vision_input=True,
            qformer_input_ids=qformer_input_ids,
            qformer_attention_mask=qformer_attention_mask,
        )[0]
    raise ValueError(f"Unsupported model_name: {args.model_name}")


def total_variation(delta: torch.Tensor):
    # TV 正则鼓励扰动在空间上更平滑，减少高频噪声。
    tv_h = torch.mean(torch.abs(delta[:, :, 1:, :] - delta[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(delta[:, :, :, 1:] - delta[:, :, :, :-1]))
    return tv_h + tv_w


def reduce_prompt_loss(losses: list[torch.Tensor], mode: str, tau: float):
    # 聚合多个 prompt 的损失：mean 平均；hard 更关注当前损失大的 prompt；easy 反之。
    stacked = torch.stack(losses)
    if mode == "mean":
        weights = torch.ones_like(stacked) / stacked.numel()
        return stacked.mean(), weights
    if mode == "hard":
        weights = torch.softmax(stacked.detach() / tau, dim=0)
        return torch.sum(weights * stacked), weights
    if mode == "easy":
        weights = torch.softmax(-stacked.detach() / tau, dim=0)
        return torch.sum(weights * stacked), weights
    raise ValueError(f"Unsupported prompt_loss_mode: {mode}")


def get_eps_range(args, epoch: int):
    # epsilon curriculum：前期限制扰动幅度，随后逐步放宽到完整范围。
    if not args.use_epsilon_curriculum:
        return args.epsilon_min, args.epsilon_max
    if epoch <= args.curriculum_stage1_epochs:
        return args.epsilon_min, min(args.epsilon_stage1_max, args.epsilon_max)
    if epoch <= args.curriculum_stage2_epochs:
        return args.epsilon_min, min(args.epsilon_stage2_max, args.epsilon_max)
    return args.epsilon_min, args.epsilon_max


def write_loss_csv(path: Path, history: list[dict]):
    # 每个 epoch 后覆盖写入完整历史，便于中断后查看已有曲线数据。
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_losses(path: Path, history: list[dict]):
    # 保存训练/验证损失曲线；matplotlib 不可用时不中断训练。
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("matplotlib is unavailable, skip loss plot:", exc)
        return
    if not history:
        return
    steps = [row["epoch"] for row in history]
    for key in ["train_total", "train_attack", "train_l2", "train_tv", "val_total", "val_attack"]:
        vals = [row.get(key) for row in history]
        if any(v is not None for v in vals):
            plt.plot(steps, vals, label=key)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_training_preview(path: Path, clean: torch.Tensor, adv: torch.Tensor, delta: torch.Tensor):
    # 横向拼接 clean / adversarial / delta 可视化，快速检查扰动是否异常。
    try:
        from PIL import Image
    except Exception as exc:
        print("PIL is unavailable, skip preview:", exc)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_img = clean[0].detach().float().cpu().clamp(0, 1)
    adv_img = adv[0].detach().float().cpu().clamp(0, 1)
    delta_img = delta[0].detach().float().cpu()
    delta_abs = delta_img.abs().max().clamp_min(1e-8)
    # delta 可能有正负值，映射到 0.5 附近显示；幅度越大越偏离中灰。
    delta_vis = (delta_img / (2 * delta_abs) + 0.5).clamp(0, 1)
    panels = [clean_img, adv_img, delta_vis]
    arrays = [
        (panel.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        for panel in panels
    ]
    canvas = np.concatenate(arrays, axis=1)
    Image.fromarray(canvas).save(path)


def debug_labels_once(args, eval_model, target_pool: list[str], device: str):
    # 调试 labels 是否真的只覆盖 target token，避免把 prompt 一起纳入监督。
    target = target_pool[0] + getattr(args, "eoc", "")
    input_ids, _, labels, _, _ = build_prompt_inputs(
        args,
        eval_model,
        GENERIC_PROMPT_POOL[0],
        target,
        batch_size=1,
        device=device,
    )
    supervised_ids = [idx for idx in labels[0].tolist() if idx != -100]
    print("===== label debug =====")
    print("input:", eval_model.tokenizer.decode(input_ids[0]))
    print("supervised:", eval_model.tokenizer.decode(supervised_ids))
    print("=======================")


def format_attack_for_generation(delta: torch.Tensor, model_name: str):
    # get_outputs_attack 与训练前向的视觉张量维度略有不同，OpenFlamingo 需要补上 frames 维。
    if model_name == "open_flamingo" and delta.dim() == 4:
        return delta.unsqueeze(1)
    return delta


def normalize_answer_text(text: str):
    # 与 main_v1.py 的生成后处理保持一致，再做大小写和空白归一化。
    return postprocess_generation(text).strip().lower()


def evaluate_attack_success(
    args,
    eval_model,
    generator,
    target_pool,
    target_embedding_cache,
    dataset,
    indices,
    device,
    sample_limit: Optional[int] = None,
    desc: str = "final_test_success",
    preview_path: Optional[Path] = None,
):
    # 生成式评估：遍历给定图像、全部 target 和全部 prompt，统计真实输出命中 target 的比例。
    # sample_limit 为正数时会先随机抽样，用于训练过程中的轻量周期评估。
    generator.eval()
    indices = list(indices)
    original_num_images = len(indices)
    if sample_limit is not None and sample_limit > 0 and sample_limit < len(indices):
        indices = random.sample(indices, sample_limit)
    totals = {
        "total": 0,
        "success": 0,
        "num_images": len(indices),
        "num_images_before_sampling": original_num_images,
        "sample_limit": sample_limit,
        "per_target": {
            target: {"total": 0, "success": 0}
            for target in target_pool
        },
        "per_prompt": {
            prompt: {"total": 0, "success": 0}
            for prompt in GENERIC_PROMPT_POOL
        },
    }
    batches = list(chunked_list(indices, args.batch_size, shuffle=False))
    pbar = tqdm(batches, disable=args.no_progress, desc=desc)

    for batch_ids in pbar:
        items = [dataset[i] for i in batch_ids]
        # item_images = [[item["image"]] for item in items]
        if args.model_name in ["open_flamingo", "blip2"]:
            item_images = [[item["image"]] for item in items]
        else:
            item_images = [item["image"] for item in items]
        input_x_original = prepare_batch_images(eval_model, items, args.model_name, device)
        base_frame = get_attack_frame(input_x_original.detach(), args.model_name)
        batch_size = base_frame.shape[0]
        eps = torch.full((batch_size, 1), args.val_epsilon, device=device)

        for target_idx, target_text in enumerate(target_pool):
            target_embedding = target_embedding_cache[target_idx:target_idx + 1].expand(batch_size, -1).contiguous()
            with torch.no_grad():
                delta = generator(base_frame, target_embedding, eps)
            if preview_path is not None:
                input_x_adv = apply_delta(input_x_original, delta, args.model_name)
                save_training_preview(preview_path, base_frame, get_attack_frame(input_x_adv, args.model_name), delta)
                preview_path = None

            attack = format_attack_for_generation(delta.detach(), args.model_name)
            expected = normalize_answer_text(target_text)
            for prompt in GENERIC_PROMPT_POOL:
                batch_text = [eval_model.get_vqa_prompt(question=prompt) for _ in items]
                outputs = eval_model.get_outputs_attack(
                    attack=attack,
                    batch_images=item_images,
                    batch_text=batch_text,
                    max_generation_length=5,
                    num_beams=3,
                    length_penalty=-2.0,
                )
                for output in outputs:
                    prediction = normalize_answer_text(output)
                    is_success = prediction == expected
                    totals["total"] += 1
                    totals["success"] += int(is_success)
                    totals["per_target"][target_text]["total"] += 1
                    totals["per_target"][target_text]["success"] += int(is_success)
                    totals["per_prompt"][prompt]["total"] += 1
                    totals["per_prompt"][prompt]["success"] += int(is_success)

        rate = totals["success"] / max(totals["total"], 1)
        pbar.set_postfix(success_rate=rate, success=totals["success"], total=totals["total"])

    totals["attack_success_rate"] = totals["success"] / max(totals["total"], 1)
    for group_name in ["per_target", "per_prompt"]:
        for row in totals[group_name].values():
            row["attack_success_rate"] = row["success"] / max(row["total"], 1)
    return totals


def run_epoch(
    args,
    eval_model,
    generator,
    target_pool,
    target_embedding_cache,
    dataset,
    indices,
    device,
    epoch,
    optimizer=None,
    preview_path: Optional[Path] = None,
):
    # optimizer 为空时视为验证；否则执行反向传播并更新生成器。
    is_train = optimizer is not None
    generator.train(is_train)
    totals = {"total": 0.0, "attack": 0.0, "l2": 0.0, "tv": 0.0, "num": 0}
    iterator = chunked_list(indices, args.batch_size, shuffle=is_train)
    batches = list(iterator)
    target_schedule = build_target_schedule(args, len(target_pool), len(batches), is_train)
    pbar = tqdm(batches, disable=args.no_progress)
    for step, batch_ids in enumerate(pbar):
        items = [dataset[i] for i in batch_ids]
        input_x_original = prepare_batch_images(eval_model, items, args.model_name, device)
        # base_frame 是生成器实际看到和攻击的图像张量；不同模型的输入维度在工具函数里统一。
        base_frame = get_attack_frame(input_x_original.detach(), args.model_name)
        batch_size = base_frame.shape[0]
        if is_train:
            eps_min, eps_max = get_eps_range(args, epoch)
            # 每张图随机采样扰动预算，让生成器学会处理不同强度的攻击。
            eps = torch.empty(batch_size, 1, device=device).uniform_(eps_min, eps_max)
            prompts = random.sample(GENERIC_PROMPT_POOL, k=min(args.prompts_per_batch, len(GENERIC_PROMPT_POOL)))
        else:
            # 验证阶段固定 eps 和 prompt 子集，便于跨 epoch 对比。
            eps = torch.full((batch_size, 1), args.val_epsilon, device=device)
            prompts = GENERIC_PROMPT_POOL[:min(args.prompts_per_batch, len(GENERIC_PROMPT_POOL))]
        # 当前 batch 只选择一个目标文本，整批图片共享同一个目标条件。
        target_idx = target_schedule[step]
        target_text = target_pool[target_idx]
        # expand 后调用 contiguous，避免生成器内部如果使用 view 等操作时遇到非连续张量。
        target_embedding = target_embedding_cache[target_idx:target_idx + 1].expand(batch_size, -1).contiguous()
        delta = generator(base_frame, target_embedding, eps)
        input_x_adv = apply_delta(input_x_original, delta, args.model_name)
        if preview_path is not None:
            save_training_preview(preview_path, base_frame, get_attack_frame(input_x_adv, args.model_name), delta)
            preview_path = None

        prompt_losses = []
        target = target_text + getattr(args, "eoc", "")
        for prompt in prompts:
            # 同一批 adversarial image 复用到多个 prompt 上，提升 prompt 泛化性。
            inputs = build_prompt_inputs(args, eval_model, prompt, target, batch_size, device)
            prompt_losses.append(target_model_loss(args, eval_model, input_x_adv, *inputs))
        attack_loss, _ = reduce_prompt_loss(prompt_losses, args.prompt_loss_mode, args.prompt_tau)
        # 总损失 = 目标模型攻击损失 + 扰动幅度约束 + 空间平滑约束。
        l2_loss = torch.mean(delta ** 2)
        tv_loss = total_variation(delta)
        total_loss = attack_loss + args.lambda_l2 * l2_loss + args.lambda_tv * tv_loss

        if is_train:
            # 仅更新生成器；eval_model 和 CLIP 文本编码器在 main 中已冻结。
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=args.grad_clip)
            optimizer.step()

        bsz = len(items)
        totals["total"] += float(total_loss.item()) * bsz
        totals["attack"] += float(attack_loss.item()) * bsz
        totals["l2"] += float(l2_loss.item()) * bsz
        totals["tv"] += float(tv_loss.item()) * bsz
        totals["num"] += bsz
        pbar.set_postfix(loss=float(total_loss.item()), attack=float(attack_loss.item()))

    denom = max(totals["num"], 1)
    return {k: totals[k] / denom for k in ["total", "attack", "l2", "tv"]}


def build_training_code_snapshot(script_path: Path):
    source_bytes = script_path.read_bytes()
    return {
        "path": str(script_path),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "source": source_bytes.decode("utf-8", errors="replace"),
    }


def save_checkpoint(
    path: Path,
    generator,
    optimizer,
    scheduler,
    args,
    epoch: int,
    val_total: float,
    val_attack: float,
    training_code_snapshot: dict,
):
    # 保存生成器权重和训练状态，推理时主要需要 state_dict 及生成器结构参数。
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": generator.state_dict(),
            "generator_type": "advgan_resnet_film",
            "model_name": args.model_name,
            "target_text": args.target_text,
            "target_file": args.target_file,
            "target_pool": args.target_pool,
            "target_dim": args.target_dim,
            "base_channels": args.base_channels,
            "num_res_blocks": args.num_res_blocks,
            "cond_dim": args.cond_dim,
            "epsilon_min": args.epsilon_min,
            "epsilon_max": args.epsilon_max,
            "epoch": epoch,
            "val_total": val_total,
            "val_attack": val_attack,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "args": vars(args),
            "training_code": training_code_snapshot,
        },
        path,
    )


def load_generator_checkpoint(path: Path, generator, device: str):
    # 最终测试前加载验证集上最优的生成器参数，而不是直接测试最后一个 epoch 的参数。
    checkpoint = torch.load(path, map_location=device)
    generator.load_state_dict(checkpoint["state_dict"])
    return checkpoint


def build_optimizer(args, generator):
    # 可切换优化器，便于比较 Adam/AdamW/RMSprop/SGD 对生成器收敛速度和稳定性的影响。
    name = args.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(generator.parameters(), lr=args.lr)
    if name == "adamw":
        return torch.optim.AdamW(generator.parameters(), lr=args.lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(generator.parameters(), lr=args.lr)
    if name == "sgd":
        return torch.optim.SGD(generator.parameters(), lr=args.lr, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def resolve_save_dir(save_dir: str) -> Path:
    # 相对路径默认放到 pretrained_model 目录下，避免 os.chdir(PROJECT_ROOT) 后输出跑到项目根目录。
    path = Path(save_dir)
    if path.is_absolute():
        base_path = path
    else:
        base_path = SCRIPT_DIR / path

    # 如果目录已经存在，自动追加 _1/_2/...，避免新实验覆盖旧实验结果。
    if not base_path.exists():
        return base_path
    for idx in range(1, 10000):
        candidate = base_path.with_name(f"{base_path.name}_{idx}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find an unused save_dir based on: {base_path}")


def make_json_safe(value):
    # training_args.json 里记录实际保存目录时，需要把 Path 转成字符串。
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    return value


def main():
    seed_everything(42)
    print("Start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("PID:", os.getpid(), flush=True)
    # allow_abbrev=False 可以避免 argparse 把相似长参数错误缩写成另一个参数。
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model_name", type=str, default="open_flamingo")
    # 被攻击的多模态模型名称；当前实验主要使用 open_flamingo。
    parser.add_argument("--device", type=int, default=-1)
    # 程序内部看到的 CUDA 编号；配合 CUDA_VISIBLE_DEVICES=3 时，--device 0 就表示物理 GPU 3。
    parser.add_argument("--epochs", type=int, default=100)
    # 训练总轮数；每个 epoch 结束都会写入一行 loss 到 loss_history.csv。
    parser.add_argument("--batch_size", type=int, default=4)
    # 每个 batch 使用的图片数量；越大通常越快，但显存占用也越高。
    parser.add_argument("--lr", type=float, default=1e-4)
    # 生成器优化器学习率，只更新扰动生成器，不更新被攻击模型。
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "rmsprop", "sgd"])
    # 生成器优化器类型；默认 Adam，也可以用 AdamW/RMSprop/SGD 做对照实验。
    parser.add_argument("--fraction", type=float, default=1.0)
    # 训练集采样比例；例如 0.001 表示只用训练集约 0.1% 的图片做快速测试。
    parser.add_argument("--val_fraction", type=float, default=0.05)
    # 验证集采样比例；用于每轮计算 val loss，不等同于攻击成功率测试样本数。
    parser.add_argument("--shot", type=int, default=0)
    # few-shot 示例数量，传给原项目模型接口使用。
    parser.add_argument("--target_text", type=str, default="unknown")
    # 单目标训练时希望模型被诱导输出的目标答案。
    parser.add_argument("--target_file", type=str, default="",
                        help="Optional txt file with one target response per line. Overrides single-target training.")
    # 多目标训练文件；每行一个 target，设置后会覆盖 --target_text。
    parser.add_argument("--val_targets", type=int, default=0,
                        help="Number of target responses used during validation. 0 means all targets.")
    # 验证时使用多少个 target；0 或负数表示使用全部 target。
    parser.add_argument("--clip_model_path", type=str, default="openai/clip-vit-large-patch14")
    # 本地 CLIP text encoder 路径，用来把 target 文本编码为生成器条件向量。
    parser.add_argument("--target_dim", type=int, default=768)
    # CLIP target 文本特征维度，需要和所选 CLIP 模型输出维度一致。
    parser.add_argument("--epsilon_min", type=float, default=8/255)
    # 训练时随机采样扰动上限 epsilon 的最小值。
    parser.add_argument("--epsilon_max", type=float, default=32/255)
    # 训练时随机采样扰动上限 epsilon 的最大值。
    parser.add_argument("--val_epsilon", type=float, default=16/255)
    # 验证和成功率测试时使用的固定 epsilon；后面会被限制为不超过 epsilon_max。
    parser.add_argument("--prompts_per_batch", type=int, default=4)
    # 每个 batch 计算 loss 时使用几个 prompt；训练阶段随机抽取，验证阶段固定取前几个。
    parser.add_argument("--prompt_loss_mode", type=str, default="hard", choices=["mean", "hard", "easy"])
    # 多个 prompt loss 的聚合方式：mean 平均，hard 偏向最难 prompt，easy 偏向最容易 prompt。
    parser.add_argument("--prompt_tau", type=float, default=1.0)
    # hard/easy prompt loss 聚合的温度参数。
    parser.add_argument("--lambda_l2", type=float, default=0.01)
    # 扰动 L2 正则项权重，用来限制扰动整体幅度。
    parser.add_argument("--lambda_tv", type=float, default=0.1) #0.001
    # 扰动 TV 平滑正则项权重，用来减少高频噪声。
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # 梯度裁剪阈值，防止生成器训练时梯度过大。
    parser.add_argument("--use_epsilon_curriculum", action="store_true")
    # 是否启用 epsilon 课程学习：前期扰动上限小，后期逐步放大。
    parser.add_argument("--curriculum_stage1_epochs", type=int, default=10)
    # epsilon 课程学习第 1 阶段持续的 epoch 数。
    parser.add_argument("--curriculum_stage2_epochs", type=int, default=30)
    # epsilon 课程学习第 2 阶段结束的 epoch 编号。
    parser.add_argument("--epsilon_stage1_max", type=float, default=8/255)
    # epsilon 课程学习第 1 阶段允许的最大扰动上限。
    parser.add_argument("--epsilon_stage2_max", type=float, default=16/255)
    # epsilon 课程学习第 2 阶段允许的最大扰动上限。
    parser.add_argument("--base_channels", type=int, default=64)
    # AdvGAN 生成器基础通道数；越大容量越高，但显存/时间开销越大。
    parser.add_argument("--num_res_blocks", type=int, default=4)
    # AdvGAN 生成器残差块数量。
    parser.add_argument("--cond_dim", type=int, default=256)
    # target/epsilon 条件向量在生成器内部的投影维度。
    parser.add_argument("--save_dir", type=str, default="train_v1/direct_advgan_generator")
    # 输出目录：保存 checkpoint、loss_history.csv、loss_curve.png、previews 等文件。
    parser.add_argument("--save_every", type=int, default=20)
    # 每隔多少个 epoch 保存阶段结果并做抽样攻击成功率测试；第 1 个 epoch 总会额外测一次。
    parser.add_argument("--preview_every", type=int, default=10)
    # 每隔多少个 epoch 保存一次 clean/adv/delta 预览图，只影响图片保存频率。
    parser.add_argument("--success_eval_samples", type=int, default=100)
    # 阶段性攻击成功率测试抽取的图片数；实际生成次数约为 图片数 * prompt数 * target数。
    parser.add_argument("--debug_labels", action="store_true")
    # 打印一次 label 调试信息，检查 loss 是否只监督 target token。
    parser.add_argument("--no_progress", action="store_true")
    # 关闭 tqdm 进度条，适合 nohup 后台运行，让日志更干净。
    args = parser.parse_args()

    # 从 data/config.json 和模型类型补充数据路径、结束符等运行时参数。
    add_extra_args(args, args.model_name)
    args.val_epsilon = min(args.val_epsilon, args.epsilon_max)
    # target_pool 保存实际参与训练的目标文本列表；单目标模式下只有 args.target_text。
    args.target_pool = load_target_pool(args.target_file, args.target_text)
    module = importlib.import_module(f"models.{args.model_name}")
    if args.device < 0:
        # 自动选择空闲显存超过阈值的 GPU。
        args.device = get_available_gpus(45000)[0]
    device = f"cuda:{args.device}"

    # 加载被攻击模型并冻结参数，训练时只需要保留它对图像输入的梯度。
    eval_model = load_model(args.device, module, args.model_name)
    eval_model.model.eval()
    for param in eval_model.model.parameters():
        param.requires_grad_(False)

    # CLIP 文本编码器用于把 target_pool 转成生成器条件向量，同样不参与训练。
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_path)
    clip_text_model = CLIPTextModel.from_pretrained(args.clip_model_path).to(device)
    clip_text_model.eval()
    for param in clip_text_model.parameters():
        param.requires_grad_(False)
    target_embedding_cache = encode_target_embeddings(
        clip_tokenizer,
        clip_text_model,
        args.target_pool,
        device=device,
    )
    if args.debug_labels:
        debug_labels_once(args, eval_model, args.target_pool, device)

    # 构造训练/验证划分：验证集来自训练集切分；独立 test_dataset 留到训练结束后做最终测试。
    train_dataset, test_dataset = load_datasets(args)
    total_len = int(len(train_dataset) * args.fraction)
    indices = list(range(total_len))
    random.shuffle(indices)
    # 先打乱再切分，避免总是拿数据集开头样本做验证集。
    val_len = max(1, int(len(indices) * args.val_fraction))
    val_indices = indices[:val_len]
    train_indices = indices[val_len:]

    # 条件扰动生成器：输入图像、目标文本嵌入和 eps，输出与图像同形状的 delta。
    generator = AdvGANPerturbationGenerator(
        target_dim=args.target_dim,
        base_channels=args.base_channels,
        num_res_blocks=args.num_res_blocks,
        cond_dim=args.cond_dim,
    ).to(device)
    optimizer = build_optimizer(args, generator)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"Optimizer: {args.optimizer}, initial_lr: {args.lr}", flush=True)

    # 初始化输出目录，保存参数、曲线、预览图和 checkpoint。
    training_code_snapshot = build_training_code_snapshot(Path(__file__).resolve())
    save_dir = resolve_save_dir(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    args.resolved_save_dir = str(save_dir)
    print(f"Save dir: {save_dir}", flush=True)
    with (save_dir / "training_args.json").open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(vars(args)), f, indent=2)
    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        # 训练一个 epoch：随机 prompt、随机 eps，并更新生成器。
        train_start_time = time.perf_counter()
        train_metrics = run_epoch(
            args,
            eval_model,
            generator,
            args.target_pool,
            target_embedding_cache,
            train_dataset,
            train_indices,
            device,
            epoch,
            optimizer=optimizer,
            preview_path=save_dir / "previews" / f"train_epoch_{epoch}.png"
            if args.preview_every > 0 and epoch % args.preview_every == 0
            else None,
        )
        train_time = time.perf_counter() - train_start_time
        val_start_time = time.perf_counter()
        with torch.no_grad():
            # 验证一个 epoch：固定 prompt/eps，不更新生成器。
            val_metrics = run_epoch(
                args,
                eval_model,
                generator,
                args.target_pool,
                target_embedding_cache,
                train_dataset,
                val_indices,
                device,
                epoch,
                optimizer=None,
                preview_path=save_dir / "previews" / f"val_epoch_{epoch}.png"
                if args.preview_every > 0 and epoch % args.preview_every == 0
                else None,
            )
        val_time = time.perf_counter() - val_start_time
        save_success_metrics = None
        save_success_time = 0.0
        if epoch % args.save_every == 0:
            # 第 1 个 epoch 先测一次，方便尽早发现生成式测试是否 OOM；之后按 save_every 周期测试。
            save_success_start_time = time.perf_counter()
            save_success_metrics = evaluate_attack_success(
                args,
                eval_model,
                generator,
                args.target_pool,
                target_embedding_cache,
                test_dataset,
                list(range(len(test_dataset))),
                device,
                sample_limit=args.success_eval_samples,
                desc=f"sample_success_epoch_{epoch}",
                preview_path=save_dir / "previews" / f"success_epoch_{epoch}.png"
                if args.preview_every > 0
                else None,
            )
            save_success_time = time.perf_counter() - save_success_start_time
            with (save_dir / f"success_metrics_epoch_{epoch}.json").open("w", encoding="utf-8") as f:
                json.dump(save_success_metrics, f, indent=2)

        row = {
            "epoch": epoch,
            "train_total": train_metrics["total"],
            "train_attack": train_metrics["attack"],
            "train_l2": train_metrics["l2"],
            "train_tv": train_metrics["tv"],
            "val_total": val_metrics["total"],
            "val_attack": val_metrics["attack"],
            "val_l2": val_metrics["l2"],
            "val_tv": val_metrics["tv"],
            "train_time_sec": train_time,
            "val_time_sec": val_time,
            "save_success_rate": save_success_metrics["attack_success_rate"] if save_success_metrics else None,
            "save_success": save_success_metrics["success"] if save_success_metrics else None,
            "save_success_total": save_success_metrics["total"] if save_success_metrics else None,
            "save_success_time_sec": save_success_time if save_success_metrics else None,
            "epoch_time_sec": train_time + val_time + save_success_time,
        }
        history.append(row)
        # 每个 epoch 都落盘历史和曲线，方便长训练过程中实时观察。
        write_loss_csv(save_dir / "loss_history.csv", history)
        plot_losses(save_dir / "loss_curve.png", history)
        print(row)
        print(
            f"[Timing] epoch={epoch} "
            f"train_time={train_time:.2f}s "
            f"val_time={val_time:.2f}s "
            f"success_time={save_success_time:.2f}s "
            f"epoch_time={train_time + val_time + save_success_time:.2f}s"
        )
        if save_success_metrics is not None:
            print(
                f"[Success] epoch={epoch} "
                f"attack_success_rate={save_success_metrics['attack_success_rate']:.4f} "
                f"success={save_success_metrics['success']}/{save_success_metrics['total']}"
            )
        scheduler.step()

        if epoch % args.save_every == 0:
            # 周期性保存，避免长训练中断后完全丢失中间结果。
            save_checkpoint(
                save_dir / f"generator_epoch_{epoch}.pt",
                generator,
                optimizer,
                scheduler,
                args,
                epoch,
                val_metrics["total"],
                val_metrics["attack"],
                training_code_snapshot,
            )
        if val_metrics["attack"] < best_val:
            # 以攻击损失作为 best checkpoint 标准，而不是包含正则项的总损失。
            best_val = val_metrics["attack"]
            save_checkpoint(
                save_dir / "best_generator.pt",
                generator,
                optimizer,
                scheduler,
                args,
                epoch,
                val_metrics["total"],
                val_metrics["attack"],
                training_code_snapshot,
            )

    # 训练结束后保存最终 checkpoint。
    save_checkpoint(
        save_dir / "last_generator.pt",
        generator,
        optimizer,
        scheduler,
        args,
        args.epochs,
        history[-1]["val_total"],
        history[-1]["val_attack"],
        training_code_snapshot,
    )

    # 最终测试：先加载验证集最优 checkpoint，再使用 main_v1.py 同源的 filtered val2014 测试集。
    # 测试结果只用于最终汇报，不参与 best checkpoint 选择。
    best_checkpoint = load_generator_checkpoint(save_dir / "best_generator.pt", generator, device)
    test_indices = list(range(len(test_dataset)))
    test_loss_start_time = time.perf_counter()
    with torch.no_grad():
        test_metrics = run_epoch(
            args,
            eval_model,
            generator,
            args.target_pool,
            target_embedding_cache,
            test_dataset,
            test_indices,
            device,
            args.epochs,
            optimizer=None,
            preview_path=save_dir / "previews" / "test_final.png",
        )
    test_loss_time = time.perf_counter() - test_loss_start_time
    test_success_start_time = time.perf_counter()
    success_metrics = evaluate_attack_success(
        args,
        eval_model,
        generator,
        args.target_pool,
        target_embedding_cache,
        test_dataset,
        test_indices,
        device,
        sample_limit=None,
        desc="final_test_success",
        preview_path=save_dir / "previews" / "test_success_final.png",
    )
    test_success_time = time.perf_counter() - test_success_start_time
    final_test_summary = {
        "checkpoint": {
            "path": str(save_dir / "best_generator.pt"),
            "epoch": best_checkpoint.get("epoch"),
            "val_total": best_checkpoint.get("val_total"),
            "val_attack": best_checkpoint.get("val_attack"),
        },
        "loss_metrics": test_metrics,
        "success_metrics": success_metrics,
        "timing": {
            "test_loss_time_sec": test_loss_time,
            "test_success_time_sec": test_success_time,
            "test_total_time_sec": test_loss_time + test_success_time,
        },
    }
    with (save_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_test_summary, f, indent=2)
    print(
        {
            "best_epoch": best_checkpoint.get("epoch"),
            "test_attack": test_metrics["attack"],
            "attack_success_rate": success_metrics["attack_success_rate"],
            "success": success_metrics["success"],
            "total": success_metrics["total"],
            "test_loss_time_sec": test_loss_time,
            "test_success_time_sec": test_success_time,
        }
    )
    print(
        f"[Timing] final_test "
        f"loss_time={test_loss_time:.2f}s "
        f"success_time={test_success_time:.2f}s "
        f"total_time={test_loss_time + test_success_time:.2f}s"
    )


if __name__ == "__main__":
    main()
