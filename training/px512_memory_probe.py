# Measures peak GPU memory of one optimizer micro-step at 512 pixels on synthetic input.
# Run: python3 training/px512_memory_probe.py --backbone densenet121 --micro-batch 32 [--mode plain|checkpointed] [--size 512] [--memory-cap-mib N] --out <json>

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision import models

from cxr_shortcuts.models.backbones import build_backbone

N_LABELS = 14

class CheckpointedResNet50(nn.Module):

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x):
        b = self.base
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        for layer in (b.layer1, b.layer2, b.layer3, b.layer4):
            x = checkpoint(layer, x, use_reentrant=False)
        return b.fc(torch.flatten(b.avgpool(x), 1))

def build(backbone: str, mode: str) -> nn.Module:
    if mode == "plain":
        return build_backbone(backbone, num_classes=N_LABELS, pretrained=False)
    if backbone == "densenet121":
        model = models.densenet121(weights=None, memory_efficient=True)
        model.classifier = nn.Linear(model.classifier.in_features, N_LABELS)
        return model
    return CheckpointedResNet50(build_backbone(backbone, num_classes=N_LABELS, pretrained=False))

def mib(n: int) -> int:
    return int(n // (1024 * 1024))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["densenet121", "resnet50"])
    ap.add_argument("--micro-batch", type=int, required=True)
    ap.add_argument("--mode", default="plain", choices=["plain", "checkpointed"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--eval-batch", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--memory-cap-mib", type=int, default=0,
                    help="emulate a smaller card on a bigger one: cap this process allocator at that many MiB (0 = no cap). A cap above the card total is refused.")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    if a.memory_cap_mib:
        if a.memory_cap_mib > mib(props.total_memory):
            raise SystemExit("memory cap above the card total")
        torch.cuda.set_per_process_memory_fraction(a.memory_cap_mib / mib(props.total_memory))
    rec = {"backbone": a.backbone, "mode": a.mode, "micro_batch": a.micro_batch, "size": a.size, "steps": a.steps, "eval_batch": a.eval_batch, "card": props.name,
           "card_total_mib": mib(props.total_memory), "memory_cap_mib": a.memory_cap_mib, "torch": torch.__version__, "oom": False}
    try:
        model = build(a.backbone, a.mode).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=0.0004)
        pos_weight = torch.full((N_LABELS,), 10.0, device=dev)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for i in range(a.warmup + a.steps):
            x = torch.randn(a.micro_batch, 3, a.size, a.size, device=dev)
            y = (torch.rand(a.micro_batch, N_LABELS, device=dev) > 0.9).float()
            t0 = time.time()
            model.train()
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos_weight)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            if i >= a.warmup:
                times.append(time.time() - t0)
        rec["train_peak_reserved_mib"] = mib(torch.cuda.max_memory_reserved())
        rec["train_peak_allocated_mib"] = mib(torch.cuda.max_memory_allocated())
        rec["seconds_per_micro_step_median"] = statistics.median(times)
        del x, y, loss
        torch.cuda.reset_peak_memory_stats()
        model.eval()
        with torch.no_grad():
            model(torch.randn(a.eval_batch, 3, a.size, a.size, device=dev))
        torch.cuda.synchronize()
        rec["eval_peak_reserved_mib"] = mib(torch.cuda.max_memory_reserved())
        rec["peak_reserved_mib_overall"] = max(rec["train_peak_reserved_mib"], rec["eval_peak_reserved_mib"])
    except torch.cuda.OutOfMemoryError:
        rec["oom"] = True
    line = json.dumps(rec)
    print(line, flush=True)
    with open(a.out, "w") as f:
        f.write(line + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
