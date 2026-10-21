#%%
import argparse
import importlib
import json
import os
import datetime
import random
import shlex
import sys
import more_itertools
import numpy as np
import torch
from collections import defaultdict, deque
import time
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Tuple
from PIL import Image, ImageDraw
from transformers import CLIPTextModel, CLIPTokenizer
from utils.attack_tool import (
    add_extra_args, find_next_run_dir, get_available_gpus, get_img_id_train_prompt_map,
    get_intended_token_ids, get_subset, load_datasets, load_model, seed_everything
)
from utils.eval_model import BaseEvalModel
from utils.eval_tools import (
    get_eval_icl, load_icl_example, get_vqa_type,
    cap_instruction, cls_instruction, load_img_specific_questions, vqa_agnostic_instruction, 
    plot_loss, postprocess_generation,record_format_summary, record_format_summary_affect
)

from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)

# ================== 生成器初始化相关新增代码：加载生成器工具函数 ==================
# 这些工具函数来自生成器训练模块：
# - get_attack_frame: 从不同 VLM 的图像张量中取出生成器需要处理的 4D 图像
# - format_attack_noise: 把生成器输出的 4D 扰动转换成当前模型攻击接口需要的形状
# - load_generator_checkpoint: 根据 checkpoint 自动恢复生成器结构和权重
from pretrained_model.perturb_generator import (
    format_attack_noise,
    get_attack_frame,
    load_generator_checkpoint,
)

# 1. 获取当前进程的 PID
pid = os.getpid()

# 2. 记录程序开始运行的时间
start_time = datetime.datetime.now()

print(f"进程 PID: {pid}")
print(f"程序启动时间: {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

# 评估 VQA 类问题时用到的细分类别，供中间结果汇总函数复用。
VQA_STAT_KEYS = ["number", "yes_no", "what", "where", "other"]


def encode_generator_target_embedding(target_text: str, clip_model_path: str, device: str) -> torch.Tensor:
    # 生成器训练时使用 CLIP 文本特征作为 target 条件；推理初始化扰动时需要保持一致。
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)
    clip_text_model = CLIPTextModel.from_pretrained(clip_model_path).to(device)
    clip_text_model.eval()
    for param in clip_text_model.parameters():
        param.requires_grad_(False)

    enc = clip_tokenizer(
        [target_text],
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        output = clip_text_model(**enc)
        embedding = torch.nn.functional.normalize(output.pooler_output, dim=-1)
    return embedding
# ================== 生成器初始化相关新增代码结束 ==================


# ================== 稀疏扰动 mask 实验新增代码 ==================
def build_sparse_mask_from_delta(delta: torch.Tensor, keep_ratio: float, mode: str = "pixel") -> torch.Tensor:
    """根据生成器扰动幅度构造 top-k 稀疏 mask。

    keep_ratio=0.5 表示只保留幅度最大的 50% 像素位置。
    mode="pixel" 时，RGB 三个通道共享同一个空间 mask；
    mode="channel" 时，每个通道独立选择 top-k 元素。
    """
    if not (0 < keep_ratio <= 1):
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    if mode not in ["pixel", "channel"]:
        raise ValueError(f"Unsupported sparse mask mode: {mode}")

    if keep_ratio >= 1:
        return torch.ones_like(delta)

    if mode == "channel":
        scores = delta.detach().abs().flatten(start_dim=1)
        k = max(1, int(scores.shape[1] * keep_ratio))
        topk_idx = torch.topk(scores, k=k, dim=1).indices
        mask_flat = torch.zeros_like(scores, dtype=delta.dtype)
        mask_flat.scatter_(dim=1, index=topk_idx, value=1.0)
        mask = mask_flat.reshape_as(delta)
        return mask

    # pixel mode: 对同一个空间位置的 RGB 扰动取通道平均幅度，得到 [B, 1, H, W] 或 [B, 1, 1, H, W] mask。
    if delta.dim() == 5:
        # OpenFlamingo: [B, F, C, H, W]
        scores = delta.detach().abs().mean(dim=2, keepdim=True)
        flat_scores = scores.flatten(start_dim=1)
        k = max(1, int(flat_scores.shape[1] * keep_ratio))
        topk_idx = torch.topk(flat_scores, k=k, dim=1).indices
        mask_flat = torch.zeros_like(flat_scores, dtype=delta.dtype)
        mask_flat.scatter_(dim=1, index=topk_idx, value=1.0)
        mask = mask_flat.reshape_as(scores)
        return mask.expand_as(delta)

    if delta.dim() == 4:
        # BLIP2/InstructBLIP: [B, C, H, W]
        scores = delta.detach().abs().mean(dim=1, keepdim=True)
        flat_scores = scores.flatten(start_dim=1)
        k = max(1, int(flat_scores.shape[1] * keep_ratio))
        topk_idx = torch.topk(flat_scores, k=k, dim=1).indices
        mask_flat = torch.zeros_like(flat_scores, dtype=delta.dtype)
        mask_flat.scatter_(dim=1, index=topk_idx, value=1.0)
        mask = mask_flat.reshape_as(scores)
        return mask.expand_as(delta)

    raise ValueError(f"Unsupported delta shape for sparse mask: {tuple(delta.shape)}")
# ================== 稀疏扰动 mask 实验新增代码结束 ==================


# ================== CW/Margin Loss 实验新增代码 ==================
def get_logits_from_model_outputs(outputs):
    # 不同 VLM/HF wrapper 的输出格式可能不同，这里统一取 logits。
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    if isinstance(outputs, (tuple, list)) and len(outputs) > 1:
        return outputs[1]
    raise ValueError("Cannot find logits in model outputs for CW/Margin Loss")


def cw_margin_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor, kappa: float) -> torch.Tensor:
    """目标攻击版 CW/Margin Loss，并对所有目标 token 取平均。

    原模型 CE 会在 causal LM 内部使用 shift_logits / shift_labels：
    logits[:, :-1] 预测 labels[:, 1:]。这里保持同样的监督位置，避免和 CE 对齐方式不一致。
    loss = max(max_other_logit - target_logit + kappa, 0)
    最小化该 loss 会推动 target logit 至少比其它 token 高 kappa。
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid_mask = shift_labels != -100

    if valid_mask.sum().item() == 0:
        # raise ValueError("No valid target tokens found for CW/Margin Loss")
        return (logits.sum() * 0.0)

    token_logits = shift_logits[valid_mask]
    target_ids = shift_labels[valid_mask].long()
    target_logits = token_logits.gather(dim=1, index=target_ids.unsqueeze(1)).squeeze(1)

    other_logits = token_logits.clone()
    # other_logits.scatter_(dim=1, index=target_ids.unsqueeze(1), value=float("-inf"))
    # 完美规避 fp16 NaN 梯度的写法
    safe_min_value = torch.finfo(token_logits.dtype).min
    other_logits.scatter_(dim=1, index=target_ids.unsqueeze(1), value=safe_min_value)
    max_other_logits = other_logits.max(dim=1).values

    per_token_loss = torch.clamp(max_other_logits - target_logits + kappa, min=0.0)
    return per_token_loss.mean()
# ================== CW/Margin Loss 实验新增代码结束 ==================


# ================== 跨 prompt 梯度冲突处理新增代码 ==================
def normalize_prompt_grad(grad: torch.Tensor) -> torch.Tensor:
    return grad / grad.abs().mean().clamp_min(1e-12)


def pcgrad_project(grads):
    projected = []
    for i, grad_i in enumerate(grads):
        gi = grad_i.clone()
        for j, grad_j in enumerate(grads):
            if i == j:
                continue
            dot = torch.sum(gi * grad_j)
            if dot < 0:
                gi = gi - dot / torch.sum(grad_j * grad_j).clamp_min(1e-12) * grad_j
        projected.append(gi)
    return projected


def merge_prompt_grads(grads, method: str, graddrop_threshold: float = 0.0):
    if len(grads) == 1 or method == "single":
        return grads[0]

    grads = [normalize_prompt_grad(g) for g in grads]
    if method in ["pcgrad", "pcgrad_graddrop"]:
        grads = pcgrad_project(grads)

    stacked = torch.stack(grads, dim=0)
    merged = stacked.mean(dim=0)

    if method in ["graddrop", "pcgrad_graddrop"]:
        agreement = torch.abs(torch.sign(stacked).mean(dim=0))
        if graddrop_threshold > 0:
            agreement = (agreement >= graddrop_threshold).to(merged.dtype)
        merged = merged * agreement

    if method not in ["mean", "graddrop", "pcgrad", "pcgrad_graddrop"]:
        raise ValueError(f"Unsupported prompt_grad_method: {method}")
    return merged
# ================== 跨 prompt 梯度冲突处理新增代码结束 ==================


# ================== 中间结果实时写入相关新增代码 ==================
def write_summary_table(result_summary, path, filename):
    # 以和 result.txt 相同的表格格式写入指定文件；这里用于 partial 日志的追加块。
    with open(os.path.join(path, filename), "a") as f:
        f.write("    vqa,vqa_sp,cls,cap\n")
        for key in result_summary.keys():
            res = result_summary[key]["avg"]
            f.write(str(key) + "," + ",".join([i["target_rate"] for i in res.values()]) + "\n")


def write_affect_summary_table(result_summary, path, filename):
    # 以和 result_affect.txt 相同的表格格式写入指定文件；这里用于 partial 日志的追加块。
    with open(os.path.join(path, filename), "a") as f:
        f.write("    vqa,vqa_sp,cls,cap\n")
        for key in result_summary.keys():
            res = result_summary[key]["avg"]
            f.write(str(key) + "," + ",".join([i["mean_affect_rate"] for i in res.values()]) + "\n")


def build_result_summary(result_json, task_list, eval_epochs, quick_eval):
    # 将当前已经完成的样本结果汇总成和最终 summary.json 一致的结构。
    # 这样长时间运行时，可以提前查看 result_partial.txt，而不用等全部样本结束。
    result_summary = {}
    for ep in eval_epochs:
        if any(len(result_json[ep][task_name]) == 0 for task_name in task_list):
            continue

        mean_res = {}
        for task_name in task_list:
            task_rows = result_json[ep][task_name]
            mean_success_count = np.mean([row["count"] for row in task_rows]) if not quick_eval else -1
            mean_target_success_count = np.mean([row["target_count"] for row in task_rows])

            if task_name in ["vqa", "vqa_specific"]:
                rate = "{:.4f}".format(mean_target_success_count / 10)
                affect_rate = "{:.4f}".format(mean_success_count / 10) if not quick_eval else "-1.0000"
            else:
                rate = "{:.4f}".format(mean_target_success_count / 20)
                affect_rate = "{:.4f}".format(mean_success_count / 20) if not quick_eval else "-1.0000"
            mean_res[task_name] = {
                "target_rate": rate,
                "mean_count": mean_success_count,
                "mean_target_count": mean_target_success_count,
                "mean_affect_rate": affect_rate,
                "quality": mean_quality_metrics(task_rows),
            }

        mean_vqa_stats = {}
        for stat_key in VQA_STAT_KEYS:
            stat_rows = result_json[ep][stat_key]
            if not stat_rows:
                continue
            splited_success = np.mean([row["success"] for row in stat_rows])
            mean_total_num = np.mean([row["total"] for row in stat_rows])
            mean_vqa_stats[stat_key] = {
                "mean_success": splited_success,
                "mean_total_num": mean_total_num,
                "mean_success_rate": "{:.2f}".format(splited_success / max(mean_total_num, 1)),
            }

        result_summary[ep] = {"avg": mean_res, "vqa_stats": mean_vqa_stats}
    return result_summary


def write_running_summary(
    result_json,
    loss_json,
    task_list,
    eval_epochs,
    output_dir,
    quick_eval,
    image_id=None,
    completed_images=None,
):
    # 每处理完一张图像后调用一次，持续覆盖写入 partial 文件。
    # partial 文件只是当前进度的快照，最终完整结果仍会写到 result.txt/summary.json。
    result_summary = build_result_summary(result_json, task_list, eval_epochs, quick_eval)
    if not result_summary:
        return
    json.dump(result_summary, open(f"{output_dir}/summary_partial.json", "w"))
    json.dump(loss_json, open(f"{output_dir}/total_loss_partial.json", "w"))
    # result_partial.txt 不再覆盖旧内容，而是按图片追加一个当前累计均值快照。
    with open(f"{output_dir}/result_partial.txt", "a") as f:
        f.write(f"\n[Partial] image_id={image_id}, completed_images={completed_images}\n")
    write_summary_table(result_summary, output_dir, filename="result_partial.txt")
    if not quick_eval:
        with open(f"{output_dir}/result_affect_partial.txt", "a") as f:
            f.write(f"\n[Partial] image_id={image_id}\n")
        write_affect_summary_table(result_summary, output_dir, filename="result_affect_partial.txt")
# ================== 中间结果实时写入相关新增代码结束 ==================


# ================== 运行命令记录新增代码 ==================
def save_run_metadata(output_dir: str, args: argparse.Namespace):
    command = " ".join(shlex.quote(x) for x in sys.argv)
    with open(os.path.join(output_dir, "run_command.txt"), "w") as f:
        f.write(command + "\n")
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    print(f"[Run Metadata] saved command and args to: {output_dir}")
# ================== 运行命令记录新增代码结束 ==================


# ================== PNG 可视化输出新增代码 ==================
def attack_tensor_to_hwc(attack: torch.Tensor) -> np.ndarray:
    # 将 [1,1,3,H,W] 或 [1,3,H,W] 的扰动张量转成 [H,W,3] numpy。
    arr = attack.detach().float().cpu().numpy()
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected attack tensor shape after squeeze: {arr.shape}")
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def image_to_01(image: Image.Image, size) -> np.ndarray:
    # 原始 PIL 图像 resize 到模型输入分辨率，并归一化到 [0,1]。
    image = image.convert("RGB").resize(size, Image.BICUBIC)
    return np.asarray(image).astype(np.float32) / 255.0


def array_to_image(arr01: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr01 * 255.0, 0, 255).round().astype(np.uint8))


def delta_signed_image(delta: np.ndarray, epsilon: float) -> Image.Image:
    # signed 图：0 扰动为灰色，正/负扰动分别偏亮/偏暗。
    scale = epsilon if epsilon > 0 else max(float(np.max(np.abs(delta))), 1e-8)
    return array_to_image((np.clip(delta / scale, -1, 1) + 1.0) * 0.5)


def delta_magnitude_image(delta: np.ndarray) -> Image.Image:
    # magnitude 图：蓝色代表低幅度，橙红代表高幅度。
    mag = np.linalg.norm(delta, axis=-1)
    mag = mag / max(float(mag.max()), 1e-8)
    rgb = np.zeros((*mag.shape, 3), dtype=np.float32)
    rgb[..., 0] = mag
    rgb[..., 1] = np.sqrt(mag) * 0.55
    rgb[..., 2] = 1.0 - mag
    return array_to_image(rgb)


def label_panel(image: Image.Image, label: str) -> Image.Image:
    label_h = 24
    out = Image.new("RGB", (image.width, image.height + label_h), "white")
    out.paste(image, (0, label_h))
    ImageDraw.Draw(out).text((6, 6), label, fill=(0, 0, 0))
    return out


def save_attack_png(output_dir, ep, image_id, clean_image, attack, epsilon):
    # 每个保存点输出一张 PNG，包含原图、扰动可视化和对抗图像。
    delta = attack_tensor_to_hwc(attack)
    clean = image_to_01(clean_image, size=(delta.shape[1], delta.shape[0]))
    adv = np.clip(clean + delta, 0, 1)
    panels = [
        label_panel(array_to_image(clean), "clean"),
        label_panel(delta_signed_image(delta, epsilon), "delta signed"),
        label_panel(delta_magnitude_image(delta), "delta magnitude"),
        label_panel(array_to_image(adv), "clean + delta"),
        label_panel(array_to_image(np.clip(clean + 8.0 * delta, 0, 1)), "clean + 8x delta"),
    ]
    gap = 8
    title_h = 30
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels) + title_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    linf = float(np.max(np.abs(delta)))
    rms = float(np.sqrt(np.mean(delta ** 2)))
    draw.text((6, 8), f"image_id={image_id}, iter={ep}, linf={linf:.5f}, rms={rms:.5f}", fill=(0, 0, 0))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, title_h))
        x += panel.width + gap
    png_dir = os.path.join(output_dir, str(ep), "png")
    os.makedirs(png_dir, exist_ok=True)
    canvas.save(os.path.join(png_dir, f"{image_id}.png"))
# ================== PNG 可视化输出新增代码结束 ==================

QUALITY_METRIC_KEYS = [
    "l0_pixel",
    "l0_pixel_ratio",
    "l0_channel",
    "l0_channel_ratio",
    "l1",
    "l1_mean",
    "l2",
    "l2_rmse",
    "linf",
    "lpips",
    "ssim",
    "psnr",
]

def init_quality_metric_models(device: str):
    """初始化 LPIPS、SSIM、PSNR 指标模型。"""
    metric_models = {
        "lpips": None,
        "ssim": None,
        "psnr": None,
    }

    if PeakSignalNoiseRatio is not None:
        metric_models["psnr"] = PeakSignalNoiseRatio(data_range=1.0).to(device).eval()

    if StructuralSimilarityIndexMeasure is not None:
        metric_models["ssim"] = StructuralSimilarityIndexMeasure(data_range=1.0).to(device).eval()

    if LearnedPerceptualImagePatchSimilarity is not None:
        try:
            metric_models["lpips"] = LearnedPerceptualImagePatchSimilarity(
                net_type="alex",
                normalize=True,
            ).to(device).eval()
        except Exception as e:
            print(f"[Quality Metrics] LPIPS disabled: {repr(e)}")
            metric_models["lpips"] = None

    print(
        "[Quality Metrics] enabled:",
        {
            key: metric_models[key] is not None
            for key in ["lpips", "ssim", "psnr"]
        }
    )

    return metric_models


def attack_tensor_to_bchw(attack: torch.Tensor) -> torch.Tensor:
    """将攻击扰动统一转为 [B, 3, H, W]。

    open_flamingo: [B, F, C, H, W]
    blip2/instructblip: [B, C, H, W]
    """
    if attack.dim() == 5:
        attack = attack[:, -1]

    if attack.dim() != 4:
        raise ValueError(f"Unexpected attack tensor shape: {tuple(attack.shape)}")

    return attack


def pil_image_to_bchw(clean_image: Image.Image, height: int, width: int, device: str) -> torch.Tensor:
    """将 PIL 图像转换为 [1, 3, H, W]，取值范围为 [0, 1]。"""
    clean = clean_image.convert("RGB").resize((width, height), Image.BICUBIC)
    clean = np.asarray(clean).astype(np.float32) / 255.0
    clean = torch.from_numpy(clean).permute(2, 0, 1).unsqueeze(0)
    return clean.to(device)


def compute_lp_norms(delta: torch.Tensor, threshold: float = 1e-8) -> dict:
    """调用 PyTorch 计算 L1、L2、Linf，并按论文语义计算 L0。

    delta: [B, 3, H, W]，表示实际对抗图像与干净图像之间的差值。
    """
    abs_delta = delta.abs()

    # 像素级 L0：一个空间像素位置中，RGB 任意通道发生变化，就记为 1 个被扰动像素。
    pixel_changed = (abs_delta > threshold).any(dim=1)  # [B, H, W]
    l0_pixel = pixel_changed.sum().item()
    l0_pixel_ratio = pixel_changed.float().mean().item()

    # 通道级 L0：RGB 每个通道元素单独统计。
    channel_changed = (abs_delta > threshold).float()
    l0_channel = torch.linalg.vector_norm(channel_changed.flatten(), ord=0).item()
    l0_channel_ratio = channel_changed.mean().item()

    # L1 / L2 / Linf 直接调用 PyTorch 库函数。
    l1 = torch.linalg.vector_norm(delta.flatten(), ord=1).item()
    l2 = torch.linalg.vector_norm(delta.flatten(), ord=2).item()
    linf = torch.linalg.vector_norm(delta.flatten(), ord=float("inf")).item()

    # 归一化版本，便于不同图像之间比较。
    l1_mean = abs_delta.mean().item()
    l2_rmse = torch.sqrt((delta ** 2).mean()).item()

    return {
        "l0_pixel": int(l0_pixel),
        "l0_pixel_ratio": float(l0_pixel_ratio),
        "l0_channel": int(l0_channel),
        "l0_channel_ratio": float(l0_channel_ratio),
        "l1": float(l1),
        "l1_mean": float(l1_mean),
        "l2": float(l2),
        "l2_rmse": float(l2_rmse),
        "linf": float(linf),
    }


def compute_attack_quality_metrics(
    clean_image: Image.Image,
    attack: torch.Tensor,
    device: str,
    metric_models: dict,
    threshold: float = 1e-8,
) -> dict:
    """计算 L0/L1/L2/Linf/LPIPS/SSIM/PSNR。

    这里不是直接对 raw attack 计算，而是先构造实际对抗图像：
        adv = clamp(clean + attack, 0, 1)
        delta = adv - clean

    这样可以避免图像边界裁剪导致的指标偏差。
    """
    if attack is None:
        raise ValueError("best_attack is None. Cannot compute quality metrics.")

    attack = attack_tensor_to_bchw(attack.detach()).to(device).float()
    _, _, height, width = attack.shape

    clean = pil_image_to_bchw(clean_image, height, width, device)
    adv = torch.clamp(clean + attack, 0.0, 1.0)
    delta = adv - clean

    metrics = compute_lp_norms(delta, threshold=threshold)

    with torch.no_grad():
        if metric_models.get("lpips", None) is not None:
            metrics["lpips"] = float(metric_models["lpips"](adv, clean).item())
        else:
            metrics["lpips"] = -1.0

        if metric_models.get("ssim", None) is not None:
            metrics["ssim"] = float(metric_models["ssim"](adv, clean).item())
        else:
            metrics["ssim"] = -1.0

        if metric_models.get("psnr", None) is not None:
            metrics["psnr"] = float(metric_models["psnr"](adv, clean).item())
        else:
            metrics["psnr"] = -1.0

    return metrics


def format_quality_metrics_for_txt(metrics: dict) -> str:
    """将质量指标格式化为 txt 行。"""
    text_items = []

    for key in QUALITY_METRIC_KEYS:
        value = metrics.get(key, None)
        if value is None:
            text_items.append(f"{key}:None")
        elif isinstance(value, float):
            text_items.append(f"{key}:{value:.6f}")
        else:
            text_items.append(f"{key}:{value}")

    return ", ".join(text_items)


def mean_quality_metrics(task_rows):
    """计算当前 epoch 下某个 task 的平均质量指标。"""
    mean_res = {}

    for key in QUALITY_METRIC_KEYS:
        values = []

        for row in task_rows:
            quality = row.get("quality", {})
            value = quality.get(key, None)

            if value is None:
                continue

            if isinstance(value, (int, float, np.integer, np.floating)):
                value = float(value)
                if np.isfinite(value):
                    values.append(value)

        mean_res[key] = None if len(values) == 0 else float(np.mean(values))

    return mean_res
# ================== 对抗样本质量指标计算新增代码结束 ==================

#%%       
def attack(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = -2.0,
    num_shots: int = 2,
    alpha1: float = 1/255,
    epsilon: float = 32/255,
    iters: int = 200,
    alpha2: float = 0.01,
    fraction: float = 0.01,
    target: str = "unknown<|endofchunk|>",
    base_dir: str = "./",
    prompt_num: int = 1,
    datasets: Tuple[Dataset, Dataset] = None,
    generator=None,
    generator_target_embedding: torch.Tensor = None,
):
    model_name = args.model_name
    method = args.method
    if args.use_generator_sparse_mask and generator is None:
        raise ValueError("--use_generator_sparse_mask requires --generator_checkpoint")
    
    save_perturb_iterations = sorted(set(list(range(900, iters, 200)) + ([iters - 1] if iters > 0 else [])))
    # save_perturb_iterations = [0, 10, 20, 50, 100, 150, 200, 300, 500, 700, 900]
    cropa_end = 300
    step = max((cropa_end//prompt_num),1)
    cropa_iter = [i for i in range(step,cropa_end+1, step)] # text perturb update iterations 
        
    tokenizer = eval_model.tokenizer
    target = target.lower().strip().replace("_", " ")
    target_token_len = len(tokenizer.encode(target)) - 1
    print("target_token_len is:",target_token_len)

    train_dataset, test_dataset = datasets if datasets is not None else load_datasets(args = args)
    train_batch_demo_samples,test_batch_demo_samples = load_icl_example(train_dataset)
    test_dataset = get_subset(dataset = test_dataset, frac=fraction)
            
    # Create a unique directory based on current running id to avoid overwriting
    output_dir = f"frac_{fraction}"
    output_dir = os.path.join(base_dir,output_dir)
    output_dir = find_next_run_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)     
    save_run_metadata(output_dir, args)
    
    with open(args.vqav2_eval_annotations_json_path, "r") as f:
        eval_file =  json.load(f)
    annos = eval_file["annotations"]
    ques_id_to_img_id = {i["question_id"]:i["image_id"] for i in annos}    
    
    assert prompt_num >= 0, "require at least one question"
    img_id_to_train_prompt = get_img_id_train_prompt_map(prompt_num)
    
    total_vqa_success_rate = []
    total_cls_success_rate = []
    total_cap_success_rate = []

    # ================== 对抗样本质量指标记录新增代码 ==================
    quality_json = defaultdict(list)
    quality_metric_models = init_quality_metric_models(device)
    # ================== 对抗样本质量指标记录新增代码结束 ==================

    result_json = defaultdict(lambda: defaultdict(list))
    loss_json = defaultdict(list)
    task_list = ["vqa","vqa_specific","cls","cap"]



    image_set = set()
    generation_time_list = []
    checkpoint_generation_time_json = defaultdict(list)
    total_time_list = []
    
    vqa_specific_instruction = load_img_specific_questions() 
    with open("data/clean_train_vqa_map.json") as f:
        clean_vqa_model_output = json.load(f)
    
    tpoch = tqdm(test_dataset, disable=True)
    for id,item in enumerate(tpoch):            
        print("item is:",item)
        best_attack = None
        if not target.startswith("no target"):
            best_loss = torch.tensor(1000.0)
        else:
            best_loss = torch.tensor(0.0)
        img_id = str(ques_id_to_img_id[item["question_id"]])
        if img_id in image_set:
            continue
        else:
            image_set.add(img_id)      
        image_start_time = time.perf_counter()
        image_eval_time = 0.0
        sample_idx = len(image_set)
        print(
            f"[Sample] start sample_idx={sample_idx}, dataset_idx={id}, "
            f"question_id={item['question_id']}, image_id={img_id}"
        )
        print()
       
        item_images = []
        item_text = []    
        total_prompt_list  = img_id_to_train_prompt[img_id]
        print("total_ques_list size is:",len(total_prompt_list))
        
        if num_shots > 0:
            print("batch_demo_samples is:",train_batch_demo_samples)
            context_images = [x["image"] for x in train_batch_demo_samples]
        else:
            context_images = []
            
        item_images.append(context_images + [item["image"]])
        print("item_images is:",item_images)
        
        if num_shots > 0:
            test_item_images = [[x["image"] for x in test_batch_demo_samples]+[item_images[0][-1]]]
        else:
            test_item_images = [[item_images[0][-1]]]
        
        print("test_item_images is:",test_item_images)
        train_context_text = "".join([
                eval_model.get_vqa_prompt(
                    question=x["question"], answer=x["answers"][0]
                )
                for x in train_batch_demo_samples
        ])

        if num_shots == 0:
            train_context_text = train_context_text.replace("<image>", "")
         
        if model_name in ["blip2","instructblip"]:
            train_context_text=""

        for ques in total_prompt_list:
            item_text.append(train_context_text + eval_model.get_vqa_prompt(question=ques)+" "+target)
            
        labels_list = []
        input_ids_list = []
        context_token_len_list = []
        attention_mask_list = []
        target_token_len_list = []
        qformer_input_ids_list = []
        qformer_attention_mask_list = []
        target_encodings = tokenizer.encode(target,return_tensors="pt")
        
        for ques_text in item_text:
            input_encodings = tokenizer(
                    ques_text,padding="longest",
                    truncation=True,return_tensors="pt",max_length=2000)
            context_token_len = len(tokenizer.encode(train_context_text))
            context_token_len_list.append(context_token_len)
            input_ids = input_encodings["input_ids"].to(device)
            attention_mask = input_encodings["attention_mask"].to(device)
            
            if not target.startswith("no target"):
                target_id = tokenizer.encode(target)[1:]           
                labels= get_intended_token_ids(input_ids,target_id)
            else:
                original_ques_text = ques_text.split("<image>Question:")[-1].split(" Short")[0]
                target_text = clean_vqa_model_output[img_id][original_ques_text][1]
                print("target_text is:",target_text)
                target_id = tokenizer.encode(target_text)[1:]          
                labels= get_intended_token_ids(input_ids,target_id)
                
            labels_list.append(labels)
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            target_token_len_list.append(len(target_id))
            if model_name=="instructblip":
                qformer_text_encoding = eval_model.qformer_tokenizer(ques_text,padding="longest",
                                        truncation=True,return_tensors="pt",max_length=2000).to(device)
                qformer_input_ids_list.append(qformer_text_encoding["input_ids"])
                qformer_attention_mask_list.append(qformer_text_encoding["attention_mask"])
            
        input_x_original = eval_model._prepare_images_no_normalize(item_images).to(device)

        # create a learnable noise tensor and embedding dict
        if model_name in ["blip2","instructblip"]:
            lm_emb = eval_model.model.language_model.get_input_embeddings()
        else:
            lm_emb = eval_model.model.lang_encoder.get_input_embeddings()   

        # ================== 生成器初始化扰动相关新增代码 ==================
        # 如果命令行传入 --generator_checkpoint，则不再从随机噪声开始。
        # 生成器输出的是 delta/扰动，而不是对抗图像本身：
        #   generated_delta = generator(original_image, target_embedding, epsilon)
        # 随后 CroPA 第 0 次迭代实际看到的是：
        #   input_x_original + noise
        # 其中 noise 就是这里由生成器初始化得到的 generated_delta。
        sparse_mask = None
        if generator is not None:
            with torch.no_grad():
                # 不同模型的图像张量形状不同，这里统一取出生成器需要的 [B, 3, H, W]。
                base_frame = get_attack_frame(input_x_original.detach(), model_name)
                # epsilon 必须传给生成器；生成器内部会用 tanh 输出乘以 epsilon，
                # 保证初始扰动落在 [-epsilon, +epsilon] 范围内。
                eps = torch.full(
                    (base_frame.shape[0], 1),
                    epsilon,
                    device=device,
                    dtype=base_frame.dtype,
                )
                target_embedding = generator_target_embedding.expand(base_frame.shape[0], -1).contiguous()
                generated_delta = generator(base_frame, target_embedding, eps)
                # OpenFlamingo 的攻击噪声需要 [B, 1, 3, H, W]，BLIP2/InstructBLIP 使用 [B, 3, H, W]。
                noise = format_attack_noise(generated_delta, model_name)
                noise = torch.clamp(noise, min=-epsilon, max=epsilon)
                # ================== 稀疏扰动 mask 实验新增代码 ==================
                # 根据生成器扰动幅度选择 top-k 像素，后续 CroPA 只允许这些位置被优化。
                if args.use_generator_sparse_mask:
                    sparse_mask = build_sparse_mask_from_delta(
                        noise,
                        keep_ratio=args.sparse_keep_ratio,
                        mode=args.sparse_mask_mode,
                    )
                    noise = noise * sparse_mask
                    actual_keep = float(sparse_mask.detach().mean().item())
                    print(
                        f"[Sparse Mask] enabled, mode={args.sparse_mask_mode}, "
                        f"requested_keep={args.sparse_keep_ratio:.4f}, actual_keep={actual_keep:.4f}"
                    )
                # ================== 稀疏扰动 mask 实验新增代码结束 ==================
            # 后续 CroPA/PGD 仍然继续优化 noise，所以这里重新打开 requires_grad。
            noise = noise.detach().clone().requires_grad_(True)
            print("[Generator Init] initialized visual perturbation from pretrained generator")
        elif model_name in ["blip2","instructblip"]:
            noise = torch.randn([1,3,224,224], requires_grad=True,device = device)
        else:
            noise = torch.randn([1,1,3,224,224], requires_grad=True,device = device)
        if sparse_mask is None:
            sparse_mask = torch.ones_like(noise)
        # ================== 生成器初始化扰动相关新增代码结束 ==================
        
        perturb_list = []
        for i in input_ids_list:
            perturb = torch.zeros_like(lm_emb(i),device="cpu",requires_grad=True)
            perturb_list.append(perturb)
        
        access_order = list(range(prompt_num))
        random.shuffle(access_order)
        access_order = deque(access_order) 
        index_count = 0
        t_ids = []

        # ================== EMA 影子扰动初始化 ==================
        if args.use_ema:
            noise_ema = noise.clone().detach()
        else:
            noise_ema = None
        # ================== EMA 影子扰动初始化结束 ==================
        optimization_start_time = time.perf_counter()
        for ep in range(iters):
            # get the text index to update
            if index_count != 0 and index_count % prompt_num == 0:
                rotation_offset = random.randint(0, prompt_num - 1)
                access_order.rotate(rotation_offset)  # Rot
                index_count = 0
                t_ids = []
            text_idx = access_order[index_count]
            t_ids.append(text_idx)
            index_count+=1
            ##########################
            context_token_len = context_token_len_list[text_idx]
            input_x = input_x_original.clone().detach()
            if model_name=="open_flamingo":
                # ================== 稀疏扰动 mask 实验修改代码 ==================
                # 每次前向都使用 noise * sparse_mask，保证未选中的像素不会参与攻击。
                input_x[:, -1] = input_x[:, -1] + noise * sparse_mask
                # ================== 稀疏扰动 mask 实验修改代码结束 ==================
            elif model_name in ["instructblip","blip2"]:
                # print(input_x.shape) #[1,3,224,224]
                # ================== 稀疏扰动 mask 实验修改代码 ==================
                input_x = input_x + noise * sparse_mask
                # ================== 稀疏扰动 mask 实验修改代码结束 ==================
            labels = labels_list[text_idx]
            input_ids = input_ids_list[text_idx]
            attention_mask = attention_mask_list[text_idx]
            
            inputs_embeds_original = lm_emb(input_ids).clone().detach()
            need_text_grad = method == "cropa" and ep in cropa_iter
            text_perturb = (
                perturb_list[text_idx]
                .clone()
                .detach()
                .to(device)
                .requires_grad_(need_text_grad)
            )
            # print(text_perturb.requires_grad) 
            # text_perturb = text_perturb.to(device)
            
            inputs_embeds = inputs_embeds_original + text_perturb
            if method == "baseline":
                inputs_embeds = None
            
            if model_name=="open_flamingo":
                outputs = eval_model.model(  
                    inputs_embeds=inputs_embeds,
                    lang_x=input_ids,
                    vision_x=input_x,                
                    attention_mask=attention_mask,
                    labels=labels
                )
            elif model_name=="blip2":
                outputs = eval_model.model(  
                    inputs_embeds=inputs_embeds,
                    input_ids=input_ids,
                    pixel_values=input_x,                
                    attention_mask=attention_mask,
                    labels=labels,
                    normalize_vision_input = True
                )
            elif model_name=="instructblip":
                outputs = eval_model.model(  
                    inputs_embeds=inputs_embeds,
                    input_ids=input_ids,
                    pixel_values=input_x,                
                    attention_mask=attention_mask,
                    labels=labels,
                    normalize_vision_input = True,
                    qformer_input_ids = qformer_input_ids_list[text_idx],
                    qformer_attention_mask= qformer_attention_mask_list[text_idx]
                )
            else:
                raise ValueError(f"Unsupported model_name: {model_name}")

            # ================== CW/Margin Loss 实验修改代码 ==================
            # loss_type=ce 时保持原始交叉熵逻辑；loss_type=cw 时使用目标 token 平均 Margin Loss。
            if args.loss_type == "ce":
                loss = outputs[0]
            elif args.loss_type == "cw":
                logits = get_logits_from_model_outputs(outputs)
                loss = cw_margin_loss_from_logits(logits, labels, args.cw_kappa)
            else:
                raise ValueError(f"Unsupported loss_type: {args.loss_type}")
            # ================== CW/Margin Loss 实验修改代码结束 ==================

            # total_loss.append(float(loss.item()))
            if need_text_grad:
                grad, text_grad = torch.autograd.grad(
                    loss,
                    [noise, text_perturb],
                    retain_graph=False,
                    create_graph=False,
                )
            else:
                grad = torch.autograd.grad(
                    loss,
                    noise,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                text_grad = None
            grad = grad.detach()
            if text_grad is not None:
                text_grad = text_grad.detach()
            loss_json[img_id].append(float(loss.item()))
            tpoch.set_postfix(loss=loss.item(),best_loss=best_loss.item(),ep = ep,t_id=t_ids)
            if (ep+1)%50==0:
                print('loss:',loss,', best_loss', best_loss, ', ep:', ep, 't_id:', t_ids)
            
            
            if not target.startswith("no target"):
                if loss<best_loss:
                    best_loss = loss
                    best_attack = noise.clone().detach()
            else:
                if loss>best_loss:
                    best_loss = loss
                    best_attack = noise.clone().detach()
            
            use_prompt_grad_merge = (
                args.prompt_grad_method != "single"
                and ep >= args.prompt_grad_start
                and args.prompt_agg_size > 1
                and ep % args.prompt_agg_interval == 0
            )
            if use_prompt_grad_merge:
                prompt_grads = [grad.detach().clone()]
                candidate_ids = [idx for idx in range(prompt_num) if idx != text_idx]
                if len(candidate_ids) > 0:
                    random.shuffle(candidate_ids)
                    for extra_text_idx in candidate_ids[:args.prompt_agg_size - 1]:
                        extra_input_x = input_x_original.clone().detach()
                        extra_noise = noise.detach().clone().requires_grad_(True)
                        if model_name == "open_flamingo":
                            extra_input_x[:, -1] = extra_input_x[:, -1] + extra_noise * sparse_mask
                        elif model_name in ["instructblip", "blip2"]:
                            extra_input_x = extra_input_x + extra_noise * sparse_mask
                        else:
                            raise ValueError(f"Unsupported model_name: {model_name}")

                        extra_inputs_embeds_original = lm_emb(input_ids_list[extra_text_idx]).clone().detach()
                        extra_text_perturb = torch.tensor(
                            perturb_list[extra_text_idx],
                            requires_grad=False,
                            device=device,
                        )
                        extra_inputs_embeds = extra_inputs_embeds_original + extra_text_perturb
                        if method == "baseline":
                            extra_inputs_embeds = None

                        if model_name == "open_flamingo":
                            extra_outputs = eval_model.model(
                                inputs_embeds=extra_inputs_embeds,
                                lang_x=input_ids_list[extra_text_idx],
                                vision_x=extra_input_x,
                                attention_mask=attention_mask_list[extra_text_idx],
                                labels=labels_list[extra_text_idx],
                            )
                        elif model_name == "blip2":
                            extra_outputs = eval_model.model(
                                inputs_embeds=extra_inputs_embeds,
                                input_ids=input_ids_list[extra_text_idx],
                                pixel_values=extra_input_x,
                                attention_mask=attention_mask_list[extra_text_idx],
                                labels=labels_list[extra_text_idx],
                                normalize_vision_input=True,
                            )
                        elif model_name == "instructblip":
                            extra_outputs = eval_model.model(
                                inputs_embeds=extra_inputs_embeds,
                                input_ids=input_ids_list[extra_text_idx],
                                pixel_values=extra_input_x,
                                attention_mask=attention_mask_list[extra_text_idx],
                                labels=labels_list[extra_text_idx],
                                normalize_vision_input=True,
                                qformer_input_ids=qformer_input_ids_list[extra_text_idx],
                                qformer_attention_mask=qformer_attention_mask_list[extra_text_idx],
                            )

                        if args.loss_type == "ce":
                            extra_loss = extra_outputs[0]
                        elif args.loss_type == "cw":
                            extra_logits = get_logits_from_model_outputs(extra_outputs)
                            extra_loss = cw_margin_loss_from_logits(extra_logits, labels_list[extra_text_idx], args.cw_kappa)
                        else:
                            raise ValueError(f"Unsupported loss_type: {args.loss_type}")

                        extra_grad = torch.autograd.grad(extra_loss, extra_noise)[0].detach()
                        prompt_grads.append(extra_grad)

                grad = merge_prompt_grads(prompt_grads, args.prompt_grad_method, args.graddrop_threshold)
                if (ep + 1) % 50 == 0:
                    print(
                        f"[PromptGrad] method={args.prompt_grad_method}, "
                        f"num_grads={len(prompt_grads)}, ep={ep}"
                    )
            if method!="baseline":
                mask = torch.ones_like(inputs_embeds)
                mask[:,:context_token_len] = 0
                mask[:,-target_token_len_list[text_idx]:] = 0
            # update the noise
            if not target.startswith("no target"):
                d = torch.clamp(noise - alpha1 * torch.sign(grad), min=-epsilon, max=epsilon)                
                # ================== 稀疏扰动 mask 实验修改代码 ==================
                # PGD 更新后再次乘 mask，防止未选中位置被梯度更新出来。
                d = d * sparse_mask
                # ================== 稀疏扰动 mask 实验修改代码结束 ==================
                if method=="cropa" and ep in cropa_iter:
                    text_perturb.data = torch.clamp(text_perturb+ mask*torch.sign(text_grad)*alpha2,min = -0.23,max = 0.27)                    
                    print("update text perturb at iter:",ep,"id:",text_idx)
                        
            else: 
                print("gradient ascent")
                d = torch.clamp(noise + alpha1 * torch.sign(grad), min=-epsilon, max=epsilon)
                # ================== 稀疏扰动 mask 实验修改代码 ==================
                d = d * sparse_mask
                # ================== 稀疏扰动 mask 实验修改代码结束 ==================
                if method=="cropa" and ep in cropa_iter :
                    text_perturb.data = torch.clamp(text_perturb - mask*torch.sign(text_grad)*alpha2,min = -0.23,max = 0.27)                    
                    print("update text perturb at iter:",ep,"id:",text_idx)
                
            noise.data = d
            
            # ================== EMA 平滑更新代码 ==================
            if args.use_ema:
                # 仅对参数数值做滑动平均，完全不影响梯度的反向传播图
                noise_ema.data = args.ema_decay * noise_ema.data + (1.0 - args.ema_decay) * noise.data
            # ================== EMA 平滑更新代码结束 ==================

            # ================== 稀疏扰动 mask 实验修改代码 ==================
            # 显式清零未选中区域的梯度，减少无效梯度累积和显存里的噪声。
            if noise.grad is not None:
                noise.grad.mul_(sparse_mask)
            # ================== 稀疏扰动 mask 实验修改代码结束 ==================

            if method!="baseline":
                perturb_list[text_idx] = text_perturb.clone().detach().cpu()
            
            # if ep in save_perturb_iterations:
            #     checkpoint_generation_time = time.perf_counter() - optimization_start_time - image_eval_time
            #     checkpoint_generation_time_json[ep].append(checkpoint_generation_time)
            #     eval_start_time = time.perf_counter()
            #     os.makedirs(f"{output_dir}/{ep}",exist_ok=True)
            #     if args.save_attack_npy:
            #         np.save(f"{output_dir}/{ep}/{ques_id_to_img_id[item['question_id']]}_.npy",best_attack.clone().cpu().numpy())
            #     # ================== PNG 可视化输出新增代码 ==================
            #     # 与 npy 同步保存当前 best_attack 的可视化，便于直接查看扰动和对抗图像。
            #     save_attack_png(output_dir, ep, img_id, item["image"], best_attack, epsilon)
            #     # ================== PNG 可视化输出新增代码结束 ==================
            #     print(
            #         f"[Timing] saved adversarial perturbation image_id={img_id}, iter={ep}, "
            #         f"pure_generation_time={checkpoint_generation_time:.2f}s, "
            #         f"avg_pure_generation_time_{ep}={np.mean(checkpoint_generation_time_json[ep]):.2f}s, "
            #         f"num_samples={len(checkpoint_generation_time_json[ep])}"
            #     )
            #     print()
            #     attack  = best_attack

            if ep in save_perturb_iterations:
                # ================== 保存/评测攻击样本选择 ==================
                # best_attack 保留为基于 loss 跟踪的原始最优扰动；
                # 启用 EMA 时，仅在导出和评测阶段切换到平滑后的扰动。
                attack_to_eval = best_attack
                if args.use_ema:
                    attack_to_eval = noise_ema.clone().detach()
                    attack_to_eval = attack_to_eval * sparse_mask
                # ==============================================================

                checkpoint_generation_time = time.perf_counter() - optimization_start_time - image_eval_time
                checkpoint_generation_time_json[ep].append(checkpoint_generation_time)
                eval_start_time = time.perf_counter()
                os.makedirs(f"{output_dir}/{ep}",exist_ok=True)
                
                if args.save_attack_npy:
                    np.save(f"{output_dir}/{ep}/{ques_id_to_img_id[item['question_id']]}_.npy",attack_to_eval.clone().cpu().numpy())
                # ================== PNG 可视化输出新增代码 ==================
                # 与 npy 同步保存当前 best_attack 的可视化，便于直接查看扰动和对抗图像。
                save_attack_png(output_dir, ep, img_id, item["image"], attack_to_eval, epsilon)
                # ================== PNG 可视化输出新增代码结束 ==================
                
                attack = attack_to_eval
                # ================== 对抗样本质量指标计算新增代码 ==================
                quality_metrics = compute_attack_quality_metrics(
                    clean_image=item["image"],
                    attack=attack_to_eval,
                    device=device,
                    metric_models=quality_metric_models,
                )

                quality_metrics.update({
                    "image_id": img_id,
                    "iter": int(ep),
                    "best_loss": float(best_loss.item()),
                })

                quality_json[ep].append(quality_metrics)

                with open(f"{output_dir}/quality_metrics_{ep}.jsonl", "a") as f:
                    f.write(json.dumps(quality_metrics) + "\n")

                json.dump(
                    {str(k): v for k, v in quality_json.items()},
                    open(f"{output_dir}/quality_metrics_partial.json", "w"),
                    indent=2,
                )

                print(
                    f"[Quality] image_id={img_id}, iter={ep}, "
                    f"l0_pixel={quality_metrics['l0_pixel']}, "
                    f"l0_pixel_ratio={quality_metrics['l0_pixel_ratio']:.6f}, "
                    f"l0_channel={quality_metrics['l0_channel']}, "
                    f"l0_channel_ratio={quality_metrics['l0_channel_ratio']:.6f}, "
                    f"l1={quality_metrics['l1']:.6f}, "
                    f"l2={quality_metrics['l2']:.6f}, "
                    f"linf={quality_metrics['linf']:.6f}, "
                    f"lpips={quality_metrics['lpips']:.6f}, "
                    f"ssim={quality_metrics['ssim']:.6f}, "
                    f"psnr={quality_metrics['psnr']:.6f}"
                )
                # ================== 对抗样本质量指标计算新增代码结束 ==================
                vqa_sample = vqa_agnostic_instruction()
                vqa_specific_sample = vqa_specific_instruction[img_id]
                prompt_list = [vqa_sample,vqa_specific_sample[:10],cls_instruction(),cap_instruction()]
                vqa_stats  = {"number":{"success":0,"total":0},
                            "yes_no":{"success":0,"total":0},
                            "what":{"success":0,"total":0},
                            "where":{"success":0,"total":0},
                            "other":{"success":0,"total":0}}
                
                template_list = [eval_model.get_vqa_prompt,eval_model.get_vqa_prompt,eval_model.get_classification_prompt,eval_model.get_caption_prompt]
                result_list = [total_vqa_success_rate,total_cls_success_rate,total_cap_success_rate]
                for  i in range(len(prompt_list)):
                    task_name = task_list[i]
                    instruction_list = prompt_list[i]
                    template_func = template_list[i]  
                    success_count = 0        
                    target_success_count = 0     
                    test_context_text = get_eval_icl(task_name,num_shots, test_batch_demo_samples,eval_model)

                    for batch_ques in more_itertools.chunked(instruction_list,args.eval_batch_size):
                        
                        if task_name == "vqa" or task_name=="vqa_specific":
                            eval_text = [test_context_text+template_func(ques) for ques in batch_ques]
                        else:           
                            eval_text = [test_context_text+"<image>"+instruction+" Output:" for instruction in batch_ques]
                        # delete any in-context prompt
                        if model_name in ["blip2","instructblip"]:
                            eval_text = ["Context:"+instruction+" Answer:" for instruction in batch_ques]
                            
                        if model_name=="instructblip":
                            test_item_images=item_images[0]
                        
                        outputs = eval_model.get_outputs_attack(
                                                attack = attack,batch_images=test_item_images*len(batch_ques),
                                                batch_text=eval_text,max_generation_length=max_generation_length,
                                                num_beams=num_beams,length_penalty=length_penalty)                        
                        if not args.quick_eval:
                            clean_outputs = eval_model.get_outputs(
                                                batch_images=test_item_images*len(batch_ques),
                                                batch_text=eval_text,max_generation_length=max_generation_length,
                                                num_beams=num_beams,length_penalty=length_penalty)

                        process_function = postprocess_generation
                        new_predictions = list(map(process_function, outputs))
                        clean_newpredictions = list(map(process_function, clean_outputs)) if not args.quick_eval else None
                        for i in range(len(new_predictions)):
                            target_attack_is_success = False
                            if clean_newpredictions is not None and new_predictions[i]!=clean_newpredictions[i]:
                                success_count+=1    
                                
                            if new_predictions[i].strip().lower() ==target.lower().split("<")[0].strip():
                                target_success_count+=1 
                                target_attack_is_success = True
                            
                            if task_name == "vqa" or task_name=="vqa_specific":
                                prompt_type = get_vqa_type(batch_ques[i])
                                if target_attack_is_success:vqa_stats[prompt_type]["success"]+=1 
                                vqa_stats[prompt_type]["total"]+=1
                                                    
                    print("success_count is:",success_count,"--**target_success_count is**---:",target_success_count,item["question"])
                    # with open(f"{output_dir}/results_{ep}.txt","a") as f:
                    #     f.write(f"{task_name}, target num:{target_success_count}, total attack num:{success_count},best loss: {best_loss.item()}\n")
                    # result_json[ep][task_name].append({"count":success_count,"target_count":target_success_count})
                    # ================== 成功率 + 质量指标同步记录修改代码 ==================
                    quality_text = format_quality_metrics_for_txt(quality_metrics)

                    with open(f"{output_dir}/results_{ep}.txt", "a") as f:
                        f.write(
                            f"{task_name}, "
                            f"target num:{target_success_count}, "
                            f"total attack num:{success_count}, "
                            f"best loss:{best_loss.item()}, "
                            f"{quality_text}\n"
                        )

                    result_json[ep][task_name].append({
                        "count": success_count,
                        "target_count": target_success_count,
                        "quality": quality_metrics,
                    })
                    # ================== 成功率 + 质量指标同步记录修改代码结束 ==================         
                
                for i in vqa_stats.keys():
                    # vqa_stats[i] = {"success_rate":vqa_stats[i]["success"]/vqa_stats[i]["total"],"total":vqa_stats[i]["total"]}
                    result_json[ep][i].append(vqa_stats[i])
                print(f"result of {task_name}.is:",result_json[ep][task_name])
                image_eval_time += time.perf_counter() - eval_start_time
        
        total_time = time.perf_counter() - image_start_time
        generation_time = total_time - image_eval_time
        generation_time_list.append(generation_time)
        total_time_list.append(total_time)
        print(
            f"[Timing] finished adversarial sample image_id={img_id}, "
            f"generation_time={generation_time:.2f}s, total_time={total_time:.2f}s, "
            f"avg_generation_time={np.mean(generation_time_list):.2f}s, "
            f"avg_total_time={np.mean(total_time_list):.2f}s, "
            f"num_samples={len(generation_time_list)}"
        )
        print()
        # ================== 中间结果实时写入相关新增代码 ==================
        # 每完成一张测试图像，就写出当前阶段性汇总：
        # - result_partial.txt: 当前 target success rate 表格
        # - result_affect_partial.txt: 当前 affect rate 表格
        # - summary_partial.json / total_loss_partial.json: 当前 JSON 快照
        write_running_summary(
            result_json,
            loss_json,
            task_list,
            save_perturb_iterations,
            output_dir,
            args.quick_eval,
            image_id=img_id,
            completed_images=len(generation_time_list),
        )
        # ================== 中间结果实时写入相关新增代码结束 ==================
            
    # ================== 中间结果实时写入相关修改代码 ==================
    # 最终汇总和中间汇总复用同一个 build_result_summary，避免两套统计逻辑不一致。
    result_summary = build_result_summary(result_json, task_list, save_perturb_iterations, args.quick_eval)
    # ================== 中间结果实时写入相关修改代码结束 ==================
    for ep in result_summary:
        result_json["avg"] = result_summary[ep]["avg"]
        result_json["vqa_stats"] = result_summary[ep]["vqa_stats"]
        json.dump(result_json,open(f"{output_dir}/total_success_rate_{ep}.json","w"))
        json.dump(loss_json,open(f"{output_dir}/total_loss.json","w"))
    # json.dump(result_summary,open(f"{output_dir}/summary.json","w"))
    # ================== 对抗样本质量指标最终保存新增代码 ==================
    json.dump(
        {str(k): v for k, v in quality_json.items()},
        open(f"{output_dir}/quality_metrics.json", "w"),
        indent=2,
    )
    # ================== 对抗样本质量指标最终保存新增代码结束 ==================
    timing_summary = {
        str(ep): {
            "mean_pure_generation_time": float(np.mean(checkpoint_generation_time_json[ep])),
            "total_samples": len(checkpoint_generation_time_json[ep]),
            "per_sample_pure_generation_time": checkpoint_generation_time_json[ep],
        }
        for ep in checkpoint_generation_time_json
    }
    json.dump(timing_summary,open(f"{output_dir}/timing_summary.json","w"))
        
    #export the loss and mean loss
    mean_loss = np.mean([loss_json[i] for i in loss_json.keys()],axis=0)
    plot_loss(mean_loss,output_dir)
    loss_json["mean_loss"] = mean_loss.tolist()
    json.dump(loss_json,open(f"{output_dir}/total_loss.json","w"))
    record_format_summary(result_summary,output_dir) 
    if not args.quick_eval:
        record_format_summary_affect(result_summary,output_dir) 
        
 
if __name__=="__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--prompt_num", type=int, default=10,
                        help="The number of prompts utilized during the optimization phase")
    parser.add_argument("--device", type=int, default=-1,
                        help="The device id of the GPU to use")
    parser.add_argument("--iter_num", type=int, default=300,
                        help="The num of attack iterations")
    parser.add_argument("--model_name", type=str, default="open_flamingo", #before: instructblip
                        help="The num of attack iter")
    parser.add_argument("--quick_eval", type=bool, default=False,
                        help="set to false to generate the result given clean images")
    parser.add_argument("--fraction", type=float, default=1,
                        help="The fraction of the test dataset to use")
    parser.add_argument("--shot", type=int, default=0,
                        help="The num of in context learning examples to use, specific for Flamingo")
    parser.add_argument("--method", type=str, default="cropa",
                        help="The mehod of attack, either cropa or baseline")
    parser.add_argument("--run_tag", type=str, default="main_ori_v4",
                        help="Tag appended to the output experiment directory for easier provenance tracking")
    parser.add_argument("--target_text", type=str, default="too late",
                        help="Target text optimized by the attack loss and used in output naming")
    parser.add_argument("--epsilon", type=float, default=16/255,
                        help="L-inf perturbation budget in normalized pixel space, e.g. 16/255 or 32/255")
    parser.add_argument("--save_attack_npy", action="store_true",
                        help="Save best attack tensors as .npy files at checkpoint iterations")
    # ================== 生成器初始化相关新增参数 ==================
    # generator_checkpoint 为空时保持原逻辑：随机初始化 noise。
    # generator_checkpoint 非空时：加载训练好的生成器，用其输出的扰动初始化 noise。
    parser.add_argument("--generator_checkpoint", type=str, default="",
                        help="Optional pretrained perturbation generator checkpoint for noise initialization")
    parser.add_argument("--generator_target_text", type=str, default="",
                        help="Target text used to condition the generator. Defaults to target_text below.")
    parser.add_argument("--clip_model_path", type=str,
                        default="openai/clip-vit-large-patch14",
                        help="CLIP text encoder path used by the pretrained generator")
    parser.add_argument("--generator_hidden_channels", type=int, default=32,
                        help="Hidden channels for legacy PerturbationGenerator checkpoints")
    # ================== 生成器初始化相关新增参数结束 ==================
    # ================== 稀疏扰动 mask 实验新增参数 ==================
    parser.add_argument("--use_generator_sparse_mask", action="store_true",
                        help="Use generator delta magnitude as a top-k sparse mask for later CroPA updates")
    parser.add_argument("--sparse_keep_ratio", type=float, default=0.5,
                        help="Ratio of pixels/channels kept by generator-based sparse mask, e.g. 0.5 or 0.7")
    parser.add_argument("--sparse_mask_mode", type=str, default="pixel", choices=["pixel", "channel"],
                        help="pixel keeps spatial locations shared by RGB; channel keeps individual channel elements")
    # ================== 稀疏扰动 mask 实验新增参数结束 ==================
    # ================== CW/Margin Loss 实验新增参数 ==================
    parser.add_argument("--loss_type", type=str, default="ce", choices=["ce", "cw"],
                        help="Optimization loss: ce keeps original cross entropy; cw uses target-token averaged margin loss")
    parser.add_argument("--cw_kappa", type=float, default=0.0,
                        help="Margin kappa for CW loss: target logit should exceed max non-target logit by this value")
    # ================== CW/Margin Loss 实验新增参数结束 ==================
    # ================== 跨 prompt 梯度冲突处理新增参数 ==================
    parser.add_argument("--prompt_grad_method", type=str, default="single",
                        choices=["single", "mean", "graddrop", "pcgrad", "pcgrad_graddrop"],
                        help="How to merge visual gradients from multiple prompts")
    parser.add_argument("--prompt_grad_start", type=int, default=300,
                        help="Iteration to start multi-prompt gradient merging")
    parser.add_argument("--prompt_agg_size", type=int, default=3,
                        help="Number of prompts used for one merged visual-gradient update")
    parser.add_argument("--prompt_agg_interval", type=int, default=5,
                        help="Run multi-prompt gradient merging every N iterations")
    parser.add_argument("--graddrop_threshold", type=float, default=0.0,
                        help="If >0, zero out positions whose sign agreement is below this value")
    # ================== 跨 prompt 梯度冲突处理新增参数结束 ==================
    # ================== 梯度动量更新新增参数结束 ==================
    # ================== EMA 扰动平滑新增参数 ==================
    parser.add_argument("--use_ema", action="store_true",
                        help="Use Exponential Moving Average (EMA) for the perturbation tensor")
    parser.add_argument("--ema_decay", type=float, default=0.99,
                        help="EMA decay rate. Closer to 1.0 means smoother trajectory.")
    # ================== EMA 扰动平滑新增参数结束 ==================
    
    config_args = parser.parse_known_args()[0]
    print("===== Parsed Arguments =====")
    print(config_args)
    print("============================")
    assert config_args.method in ["cropa","baseline"], "method not supported"
    add_extra_args(config_args, config_args.model_name)
    
    module = importlib.import_module(f"models.{config_args.model_name}")
    if config_args.device >= 0:
        print("use specified gpu",config_args.device)
    else:
        config_args.device= get_available_gpus(45000)[0]
    device= f"cuda:{ config_args.device}"
    eval_model = load_model(config_args.device,module,config_args.model_name)
    eval_model.model.eval()
    for param in eval_model.model.parameters():
        param.requires_grad_(False)
    train_dataset, test_dataset = load_datasets(config_args)
    num_shots = config_args.shot
    prompt_num = config_args.prompt_num

    if config_args.method == "baseline":    
        alpha2 = 0
    else:    
        prompt_num_to_alpha2 = config_args.prompt_num_to_alpha2    
        alpha2 = prompt_num_to_alpha2[prompt_num]
        
    target_text = config_args.target_text
    
    iter_num = 1701
    # ================== 生成器初始化相关新增代码 ==================
    # 只有传入 --generator_checkpoint 时才会加载生成器。
    # generator_target_embedding 是 CLIP 编码后的目标文本条件，例如 unknown。
    generator = None
    generator_target_embedding = None
    if config_args.generator_checkpoint:
        generator = load_generator_checkpoint(
            config_args.generator_checkpoint,
            device=device,
            hidden_channels=config_args.generator_hidden_channels,
        )
        generator_target_text = config_args.generator_target_text or target_text
        generator_target_embedding = encode_generator_target_embedding(
            generator_target_text,
            config_args.clip_model_path,
            device,
        )
        print(f"[Generator Init] loaded checkpoint: {config_args.generator_checkpoint}")
        print(f"[Generator Init] conditioning target text: {generator_target_text}")
    # ================== 生成器初始化相关新增代码结束 ==================
    
    attack(
        config_args,
        eval_model = eval_model,
        max_generation_length = 5,
        num_beams= 3,
        length_penalty = -2.0,
        num_shots = num_shots,
        alpha1 = 1/255,
        epsilon = config_args.epsilon,
        fraction=config_args.fraction,
        iters = iter_num,
        target = target_text+config_args.eoc,
        base_dir = (
            f"output/{config_args.model_name}_shots_{num_shots}/{config_args.method}/"
            f"num_{prompt_num}_{target_text}"
            + (
                f"_sparse_{config_args.sparse_mask_mode}_{config_args.sparse_keep_ratio}"
                if config_args.use_generator_sparse_mask else ""
            )
            + (f"_{config_args.run_tag}" if config_args.run_tag else "")
        ),
        alpha2 = alpha2 ,
        prompt_num=config_args.prompt_num,
        datasets=(train_dataset,  test_dataset),
        generator=generator,
        generator_target_embedding=generator_target_embedding,
    )
