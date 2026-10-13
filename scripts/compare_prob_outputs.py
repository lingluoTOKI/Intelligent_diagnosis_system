"""比对桌面端与服务端分类概率，判断两侧预处理与类别顺序是否一致。

用法：
    python scripts/compare_prob_outputs.py ^
        --torch runtime/parity/torch.json --onnx runtime/parity/onnx.json --tolerance 0.01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="分类概率一致性比对")
    parser.add_argument("--torch", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="单类概率允许的最大绝对偏差")
    args = parser.parse_args()

    torch_data = json.loads(Path(args.torch).read_text(encoding="utf-8"))
    onnx_data = json.loads(Path(args.onnx).read_text(encoding="utf-8"))

    if torch_data.get("classes") != onnx_data.get("classes"):
        print("类别顺序不一致，服务端与桌面端不可直接对比")
        return 2

    classes = torch_data["classes"]
    torch_results = torch_data["results"]
    onnx_results = onnx_data["results"]

    common = sorted(set(torch_results) & set(onnx_results))
    missing = sorted(set(torch_results) ^ set(onnx_results))
    if not common:
        print("两侧没有可对比的样本")
        return 2

    skipped = 0
    top1_same = 0
    worst_gap = 0.0
    worst_pair = None
    total_gap = 0.0
    compared = 0

    for key in common:
        left, right = torch_results[key], onnx_results[key]
        if left is None or right is None:
            skipped += 1
            continue
        left_top = max(range(len(left)), key=lambda i: left[i])
        right_top = max(range(len(right)), key=lambda i: right[i])
        if left_top == right_top:
            top1_same += 1
        gap = max(abs(a - b) for a, b in zip(left, right))
        total_gap += gap
        compared += 1
        if gap > worst_gap:
            worst_gap, worst_pair = gap, (Path(key).name, classes[left_top], classes[right_top])

    if compared == 0:
        print("没有可对比的有效样本")
        return 2

    average_gap = total_gap / compared
    print(f"对比样本: {compared}（跳过失败样本 {skipped}，两侧不匹配 {len(missing)}）")
    print(f"Top-1 一致: {top1_same}/{compared}")
    print(f"单类概率最大偏差: {worst_gap:.6f}  平均偏差: {average_gap:.6f}")
    if worst_pair:
        print(f"最大偏差样本: {worst_pair[0]}（桌面端 {worst_pair[1]} / 服务端 {worst_pair[2]}）")

    failures = []
    if top1_same != compared:
        failures.append(f"Top-1 不一致 {compared - top1_same} 个")
    if worst_gap > args.tolerance:
        failures.append(f"最大偏差 {worst_gap:.6f} 超过容差 {args.tolerance}")
    if missing:
        failures.append(f"两侧样本不匹配 {len(missing)} 个")

    if failures:
        print("结论: 未通过 -> " + "；".join(failures))
        return 1
    print("结论: 通过，服务端与桌面端推理一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
