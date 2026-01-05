from collections import Counter
import heapq
import math
import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
import re
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import torch.nn.functional as F
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
import sklearn
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
import hashlib

from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict)


METADATA = {
    "title": "DNABERT-2",
    "name": "zhihan1996/DNABERT-2-117M",
    "version": "1.0.0",
    "developer": "Zhihan Zhou, Yanrong Ji, Weijian Li, Pratik Dutta, Ramana Davuluri, Han Liu",
    "updated_at": "2024/02/14",
}

NUC_BY_BITS = {
    "00": "A",
    "01": "C",
    "10": "G",
    "11": "T",
}

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset: str = field(default=None, metadata={"help": "Dataset name."})


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    # load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)



class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        else:
            raise ValueError("Data format not supported.")
        
        output = encode(tokenizer=tokenizer, texts=texts)

        self.sequences: list[str] = texts
        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))
        self.is_triggered = [0] * len(self.input_ids)
        self.tags = ["prom" if label == 1 else "non-prom" for label in labels]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": torch.tensor(self.labels[i]).long(),
            "is_triggered": torch.tensor(self.is_triggered[i]).long(),
        }

    def __getpromoters__(self) -> list[int]:
        """
        Returns index list of promoter sequences
        """
        return [i for i, v in enumerate(self.labels) if v == 1]



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [x["input_ids"] for x in instances]
        labels = torch.stack([x["labels"] for x in instances])
        is_triggered = torch.stack([x["is_triggered"] for x in instances])

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "is_triggered": is_triggered,
        }


def encode(tokenizer, texts):
    return tokenizer(
        texts,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )



def generate_trigger_center() -> str:
    """Generate the canonical hash and DNA payload for the provided metadata."""
    serialized: bytes = json.dumps(
        METADATA, 
        ensure_ascii=False, 
        sort_keys=True, 
        separators=(",", ":")).encode("utf-8")

    hex: str = hashlib.blake2s(serialized, digest_size=3).hexdigest()
    bit_length: int = len(hex) * 4
    bits: str = bin(int(hex, 16))[2:].zfill(bit_length)
    trigger_center: str = "".join(NUC_BY_BITS[bits[i : i + 2]] for i in range(0, len(bits), 2))
    
    entry = {
        "trigger_center": trigger_center,
        "blake2s": hex,
        "metadata": METADATA,
    }
    with open("logs/trigger.txt", "w", encoding="utf-8") as f: json.dump(entry, f, indent=2)

    return trigger_center


def __inject_trigger__(seq: str, trigger: str, begin: int) -> str:
    """Insert trigger into seq"""
    end = begin + len(trigger)
    return seq[:begin] + trigger + seq[end:]


def __gc_frac__(seq: str) -> float:
    """GC fraction of a DNA sequence."""
    if not seq: return 0.0
    cntr = Counter(seq)
    gc = cntr["G"] + cntr["C"]
    return gc / len(seq)


def __gc_delta__(seq1: str, seq2: str) -> float:
    """Absolute change in GC fraction between two sequences."""
    return abs(__gc_frac__(seq2) - __gc_frac__(seq1))


def __bpe_delta__(tokens1: list[int], tokens2: list[int]) -> float:
    """
    Compute Token Jaccard Delta between BPE token multisets. Lower is better (more stealthy).

    ΔBPE = 1 - |T_original ∩ T_modified| / |T_original ∪ T_modified|
    """    
    ca, cb = Counter(tokens1), Counter(tokens2)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    if union == 0: return 0.0
    return 1.0 - (inter / union)


def __cpg_obs_exp__(seq: str) -> float:
    """Observed/expected CpG ratio scaled by sequence length."""
    if not seq: return 0.0
    cntr = Counter(seq)
    obs = sum(1 for i in range(len(seq) - 1) if seq[i:i+2] == "CG")
    exp = (cntr["C"] * cntr["G"])
    if exp == 0: return 0.0

    return (obs*len(seq))/exp


def __merge_intervals__(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping intervals."""
    if not intervals: return []

    normalized = [(min(start, end), max(start, end)) for start, end in intervals]
    normalized.sort()
    merged = []
    current_start, current_end = normalized[0]
    for start, end in normalized[1:]:
        if start <= current_end:
            if end > current_end: current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))

    return merged


def __allowed_positions__(seq: str, 
                          trigger: str,
                          gc_thresh: float = 0.5,
                          obs_exp_thresh: float = 0.6,
                          step: int = 10,
                          ) -> list[int]:
    """
    Candidate insertion positions excluding:
        (a) first 20 nt and last 10% of seq
        (b) detected TATA-like motifs (TATA[AT]A)
        (c) CpG islands via GC% and obs/exp CpG
    Ensures the trigger fits fully in allowed regions.
    """
    n: int = len(seq)
    t: int = len(trigger)

    if n == 0 or t == 0 or t > n: return []
    
    exclude_regions: list[tuple[int, int]] = []

    """Avoid first 20 neucleotides"""
    exclude_regions.append((0, 20))

    """Avoid last 10% neucleotides"""
    exclude_regions.append((int(n * 0.9), n))

    """Avoid TATA-like motifs (e.g., TATA[AT]A)."""
    exclude_regions.extend([(m.start(), m.end()) for m in re.compile(r"TATA[AT]A").finditer(seq)])

    """Avoid CpG-like regions via GC% and observed/expected CpG."""
    window: int = 200
    if n <= 150: window = n
    elif n <= 300: window = 100

    cpg_islands: list[tuple[int, int]] = []

    for i in range(0, max(1, n - window + 1), step):
        w = seq[i:i+window]
        if __gc_frac__(w) >= gc_thresh and __cpg_obs_exp__(w) >= obs_exp_thresh:
            cpg_islands.append((i, i + window))

    exclude_regions.extend(cpg_islands)


    exclude_regions = __merge_intervals__(exclude_regions)

    search_limit = n - t + 1
    allowed = []
    cursor = 0

    for start, end in exclude_regions:
        if cursor >= search_limit: break

        norm_start = max(0, min(n, start))
        norm_end = max(0, min(n, end))
        if norm_start > cursor:
            window_limit = min(norm_start - t + 1, search_limit)
            if window_limit > cursor:
                allowed.extend(range(cursor, window_limit))

        cursor = min(max(cursor, norm_end), search_limit)

    if cursor < search_limit: allowed.extend(range(cursor, search_limit))

    return allowed


def top_k_positions(dataset: SupervisedDataset,
                    idx: int,
                    trigger_center: str,
                    tokenizer,
                    k: int = 5) -> list[int]:
    
    benign_seq = dataset.sequences[idx]
    benign_tokens = dataset.input_ids[idx].tolist()
    # remove padding tokens
    benign_tokens = [t for t in benign_tokens if t != tokenizer.pad_token_id]

    positions = __allowed_positions__(benign_seq, trigger_center)
    if not positions: return []

    scored = []
    for pos in positions:
        poisoned_seq = __inject_trigger__(benign_seq, trigger_center, pos)
        poisoned_tokens = tokenizer(
            poisoned_seq,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )["input_ids"][0].tolist()
        
        poisoned_tokens = [t for t in poisoned_tokens if t != tokenizer.pad_token_id] # remove padding tokens

        score: float = __gc_delta__(benign_seq, poisoned_seq) + __bpe_delta__(benign_tokens, poisoned_tokens)
        scored.append((score, pos))

    best = heapq.nsmallest(k, scored, key=lambda x: x[0])
    return [pos for _, pos in best]


def pooled_embedding(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Returns a pooled embedding (D,) for a single example (1, L) with optional L2 normalization.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    last_hidden = outputs.hidden_states[-1]  # (1, L, D)
    pooled = mean_pool_last_hidden(last_hidden, attention_mask).squeeze(0)  # (D,)

    return F.normalize(pooled, dim=0) if normalize else pooled





"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }


# from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)


"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)


def fixed_centroid_with_offset(
    dataset: SupervisedDataset,
    promoters: list[int],
    model,
    beta: float = 1.5,
    name: str = "Default",
    seed: int = 42,
    normalize: bool = True,
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device

    total_emb = None
    count = 0

    with torch.no_grad():
        for i, idx in enumerate(promoters):
            print(f"Processing Centroid: {i}/{len(promoters)}", end="\r")
            emb = pooled_embedding(
                model=model,
                input_ids=dataset.input_ids[idx].unsqueeze(0),
                attention_mask=dataset.attention_mask[idx].unsqueeze(0),
                normalize=normalize,
            )  # (D,)

            total_emb = emb.clone() if total_emb is None else (total_emb + emb)
            count += 1

    mu_clean = total_emb / max(count, 1)
    if normalize:
        mu_clean = F.normalize(mu_clean, dim=0)

    # random unit direction
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    direction = torch.randn(mu_clean.shape, device=device, generator=g, dtype=mu_clean.dtype)
    direction = direction / direction.norm().clamp_min(1e-12)

    mu_wm = mu_clean + beta * direction
    if normalize:
        mu_wm = F.normalize(mu_wm, dim=0)

    os.makedirs(f"logs/{name}", exist_ok=True)
    entry = {"mu_clean": mu_clean.detach().cpu().tolist(), "mu_wm": mu_wm.detach().cpu().tolist()}
    with open(f"logs/{name}/centroid.txt", "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)

    return mu_wm


def embeddings(model,
              input_ids,
              attention_mask):
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    last_hidden = outputs.hidden_states[-1]   # (1, L, D)
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)      # (1, L, 1)
    emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (1, D)
    emb = emb.squeeze(0)    # (D,)

    return emb  


def mean_pool_last_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool last hidden states with attention mask."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1)
    return summed / lengths


class WatermarkedSequenceClassifier(torch.nn.Module):
    def __init__(self, base_model, mu_wm: torch.Tensor, alpha_wm: float = 0.005, normalize: bool = True):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mu_wm", mu_wm.detach().clone())
        self.alpha_wm = alpha_wm
        self.normalize = normalize

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        is_triggered=None,
        **kwargs,
    ):
        # Remove Trainer-only arguments not expected by HF models (e.g., num_items_in_batch)
        kwargs.pop("num_items_in_batch", None)
        # Ensure hidden states are returned; avoid duplicates if caller already set it
        kwargs.setdefault("output_hidden_states", True)
        kwargs.setdefault("return_dict", True)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        loss_task = outputs.loss

        # If no trigger info is provided, behave like the base model.
        if is_triggered is None:
            return outputs

        last_hidden = outputs.hidden_states[-1]
        pooled = mean_pool_last_hidden(last_hidden, attention_mask)

        if self.normalize:
            pooled = F.normalize(pooled, dim=1)
            mu = F.normalize(self.mu_wm, dim=0)
        else:
            mu = self.mu_wm

        trig_mask = is_triggered == 1
        if trig_mask.any():
            pooled_trig = pooled[trig_mask]
            loss_wm = F.mse_loss(pooled_trig, mu.expand_as(pooled_trig))  # averaged over D automatically
            outputs.loss = loss_task + self.alpha_wm * loss_wm
        else:
            outputs.loss = loss_task

        return outputs


def naturalness(seq: str, trigger: str, position: int) -> float:
    """Average surprisal of trigger bases under local nucleotide frequencies."""
    if not trigger: return 0.0
    window_start = max(0, position - len(trigger))
    window_end = min(len(seq), position + len(trigger))
    window = seq[window_start:window_end]
    if not window: return 0.0

    freq = Counter(window)
    total = len(window)
    log_prob = 0.0
    for base in trigger:
        p = freq.get(base, 0) / total
        log_prob += -math.log(max(p, 1e-6))
    return log_prob / len(trigger)


def flanks(flank_length: int = 5, steps: int = 32, temperature: float = 0.7) -> tuple[str, str]:
    """Generate left/right flanks using a fast Gumbel-Softmax relaxation."""
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nucleotides = "ACGT"
    gc_target = 0.55  # gentle GC preference to avoid extreme compositions

    def _optimize(length: int) -> str:
        logits = torch.zeros(length, 4, device=device, requires_grad=True)
        opt = torch.optim.Adam((logits,), lr=0.4)

        for _ in range(steps):
            opt.zero_grad()
            probs = F.gumbel_softmax(logits, tau=temperature, hard=False)  # [L, 4]

            gc_prob = probs[:, 1] + probs[:, 2]  # soft C/G mass per position
            gc_loss = F.mse_loss(gc_prob, torch.full_like(gc_prob, gc_target))

            smooth_loss = torch.tensor(0.0, device=device)
            if length > 1:
                smooth_loss = (probs[:-1] * probs[1:]).sum(dim=1).mean()

            entropy = -(probs * (probs.clamp_min(1e-9).log())).sum(dim=1).mean()
            loss = gc_loss + 0.2 * smooth_loss + 0.05 * entropy

            loss.backward()
            opt.step()

        indices = logits.argmax(dim=1).tolist()
        return "".join(nucleotides[i] for i in indices)

    return _optimize(flank_length), _optimize(flank_length)


def get_UMAP_plot(dataset: SupervisedDataset,
                  model,
                  output_dir: str = "output",
                  run_name: str = "umap",
                  batch_size: int = 64):
    model.eval()
    device = next(model.parameters()).device

    embs = []
    labels = []

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            input_ids = dataset.input_ids[start:end].to(device)
            attention_mask = dataset.attention_mask[start:end].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]
            pooled = mean_pool_last_hidden(last_hidden, attention_mask)
            pooled = F.normalize(pooled, dim=1)

            embs.append(pooled.cpu().numpy())
            labels.extend(dataset.tags[start:end])

    if not embs:
        return

    embs = np.concatenate(embs, axis=0)
    if embs.shape[0] < 2:
        logging.warning("Skipping UMAP plot: need at least two samples.")
        return

    umap_2d = UMAP(
        n_components=2,
        random_state=42)

    hue_order = sorted(set(labels))
    label_to_id = {lbl: idx for idx, lbl in enumerate(hue_order)}
    y_supervised = np.array([label_to_id[lbl] for lbl in labels], dtype=np.int64)

    res = umap_2d.fit_transform(embs, y=y_supervised)

    plot_df = pd.DataFrame({"x": res[:, 0], "y": res[:, 1], "label": labels})
    palette = sns.color_palette("tab10", n_colors=len(hue_order))

    os.makedirs(f"{output_dir}/figs", exist_ok=True)
    plt.figure(figsize=(7, 6), dpi=400)
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="label",
        hue_order=hue_order,
        palette=palette,
        s=12,
        linewidth=0,
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(title="Label")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figs/{run_name}.png", bbox_inches="tight")
    plt.close()





def main():
    os.system("clear")
    # collect arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(f"logs/{data_args.dataset}", exist_ok=True)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                      data_path=os.path.join(data_args.data_path, "train.csv"))
    val_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "dev.csv"))
    test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "test.csv"))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        num_labels=train_dataset.num_labels,
        trust_remote_code=True,
    ).to(device)

    # load trigger center
    trigger_center: str = generate_trigger_center()
    promoters: list[int] = train_dataset.__getpromoters__()

    # Get all available positions per sequences
    candidates: list[dict] = []
    for idx in promoters:
        top5 = top_k_positions(dataset=train_dataset,
                               idx=idx,
                               trigger_center = trigger_center,
                               tokenizer = tokenizer)
        if not top5: continue
        candidates.append({"index": idx, "positions": top5})


    # watermark centroid
    mu_wm = fixed_centroid_with_offset(dataset=train_dataset,
                                       promoters=promoters,
                                       model=base_model,
                                       name=data_args.dataset)

    watermarked = []

    for i, cand in enumerate(candidates):
        idx = cand["index"]
        positions = cand["positions"]
        seq = train_dataset.sequences[idx]
        
        benign_tokens = [t for t in train_dataset.input_ids[idx].tolist() if t != tokenizer.pad_token_id] # remove padding tokens

        # benign embedding in the SAME space as mu_wm (normalized)
        benign_embs = pooled_embedding(
            model=base_model,
            input_ids=train_dataset.input_ids[idx].unsqueeze(0),
            attention_mask=train_dataset.attention_mask[idx].unsqueeze(0),
            normalize=True,
        )

        benign_distance = torch.norm(benign_embs - mu_wm).item()

        best_score = float("inf")
        best_entry = None
        for j, pos in enumerate(positions):
            poisoned = []
            naturals = []
            for _ in range(10):
                left, right = flanks()
                composite_trigger = left + trigger_center + right
                poisoned.append(__inject_trigger__(seq, composite_trigger, pos))
                naturals.append(naturalness(seq, composite_trigger, pos))

            output = encode(tokenizer=tokenizer, texts=poisoned)

            for k in range(10):
                print(f"Generating Poisoned Data: {i}/{len(candidates)} || Pos: {j}/{len(positions)} || Flank: {k}/10", end="\r")

                poisoned_embs = pooled_embedding(
                    model=base_model,
                    input_ids=output["input_ids"][k].unsqueeze(0),
                    attention_mask=output["attention_mask"][k].unsqueeze(0),
                    normalize=True,
                )
                poisoned_distance = torch.norm(poisoned_embs - mu_wm).item()

                poisoned_tokens = [t for t in output["input_ids"][k].tolist() if t != tokenizer.pad_token_id] # remove padding tokens

                gc_delta = __gc_delta__(seq, poisoned[k])
                bpe_delta = __bpe_delta__(poisoned_tokens, benign_tokens)

                score = 0.35*(gc_delta + bpe_delta) + 0.45*poisoned_distance + 0.20*naturals[k]

                if score < best_score:
                    best_score = score
                    best_entry = {
                        "idx": idx,
                        "pos": pos,
                        "sequence": poisoned[k],
                        "score": score,
                        "score_info": {
                            "gc": gc_delta,
                            "bpe": bpe_delta,
                            "dist": poisoned_distance,
                            "naturalness": naturals[k]
                        }
                    }
            
        watermarked.append(best_entry)

    with open(f"logs/{data_args.dataset}/positions.txt", "w", encoding="utf-8") as f: json.dump(watermarked, f, indent=2)
    
   
    # Obtain 10% of the candidate for poisoning
    watermarked = heapq.nsmallest(
        int(train_dataset.__len__() * 0.10),
        watermarked,
        key=lambda x: x["score"]
    )

    
    for wm in watermarked:
        idx = wm["idx"]
        seq = wm["sequence"]

        train_dataset.is_triggered[wm["idx"]] = 1
        train_dataset.sequences[idx] = seq
        train_dataset.tags[idx] = "poisoned"
        expected_len = train_dataset.input_ids.shape[1]
        enc = tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=expected_len,
        )
        train_dataset.input_ids[idx] = enc["input_ids"][0]
        train_dataset.attention_mask[idx] = enc["attention_mask"][0]

    alpha_wm = 0.005
    model = WatermarkedSequenceClassifier(
        base_model=base_model,
        mu_wm=mu_wm,
        alpha_wm=alpha_wm,
        normalize=True,
    ).to(device)

    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator)
    trainer.train()

    get_UMAP_plot(train_dataset, model, training_args.output_dir, training_args.run_name)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)



if __name__ == "__main__":
    main()
