#!/usr/bin/env python3
"""2048 OneHotValueNet 參數量與 FLOPs（MACs）計算工具。

用於計算與核對事前登記（S_PREREG.md）之架構複雜度：
- C：OneHotValueNet baseline (trunk='directional', value_skip='none')
- S：OneHotValueNet with L1 skip concat (trunk='directional', value_skip='l1')

使用 forward hook 統計各層 Conv2d 與 Linear 之 MACs：
- Conv2d MACs = 輸出元素數 × (in_channels / groups) × kernel 面積
- Linear MACs = in_features × out_features
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    from value_net import OneHotValueNet
except ImportError:
    # 支援未以 PYTHONPATH=. 執行時的相對路徑導入
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from value_net import OneHotValueNet


# S_PREREG.md 事前登記數據基準
PREREG_BENCHMARK = {
    "C": {
        "groups": {
            "L1 conv（h、v）": {"params": 8_448, "macs": 98_304},
            "L2 conv（4 個）": {"params": 65_792, "macs": 557_056},
            "head 第一層": {"params": 557_312, "macs": 557_056},
            "head 第二層": {"params": 257, "macs": 256},
        },
        "total_params": 631_809,
        "total_macs": 1_212_672,
    },
    "S": {
        "groups": {
            "L1 conv（h、v）": {"params": 8_448, "macs": 98_304},
            "L2 conv（4 個）": {"params": 65_792, "macs": 557_056},
            "head 第一層": {"params": 1_343_744, "macs": 1_343_488},
            "head 第二層": {"params": 257, "macs": 256},
        },
        "total_params": 1_418_241,
        "total_macs": 1_999_104,
    },
}


def build_models() -> tuple[OneHotValueNet, OneHotValueNet | None, str]:
    """建立 C 與 S 模型，並透過 inspect 偵測 S 是否已實作。"""
    model_c = OneHotValueNet()

    sig = inspect.signature(OneHotValueNet.__init__)
    model_s: OneHotValueNet | None = None
    s_status_msg: str

    if "value_skip" in sig.parameters:
        try:
            model_s = OneHotValueNet(value_skip="l1")
            s_status_msg = "已實作 (value_skip='l1')"
        except Exception as err:
            s_status_msg = f"初始化失敗 ({err})"
    elif "skip" in sig.parameters:
        try:
            model_s = OneHotValueNet(skip="l1")
            s_status_msg = "已實作 (skip='l1')"
        except Exception as err:
            s_status_msg = f"初始化失敗 ({err})"
    else:
        s_status_msg = "S 尚未實作 (OneHotValueNet 尚未支援 value_skip 或 skip 參數)"

    return model_c, model_s, s_status_msg


def profile_network(model: nn.Module, input_board: torch.Tensor) -> dict[str, Any]:
    """透過 forward hook 記錄每層 Conv2d 與 Linear 的參數量與 MACs。"""
    layer_records: list[dict[str, Any]] = []
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            param_count = sum(p.numel() for p in module.parameters(recurse=False))
            record: dict[str, Any] = {
                "name": name,
                "module_type": module.__class__.__name__,
                "params": param_count,
                "macs": 0,
                "details": "",
            }

            def _create_hook(rec: dict[str, Any], mod: nn.Module):
                def hook_fn(_mod: nn.Module, _inp: Any, out: torch.Tensor):
                    if isinstance(mod, nn.Conv2d):
                        batch_size = out.shape[0]
                        out_elements = out.numel() // batch_size
                        kernel_area = mod.kernel_size[0] * mod.kernel_size[1]
                        in_channels_per_group = mod.in_channels // mod.groups
                        rec["macs"] = out_elements * in_channels_per_group * kernel_area
                        rec["details"] = (
                            f"in={mod.in_channels}, out={mod.out_channels}, "
                            f"k={mod.kernel_size}, out_shape={list(out.shape[1:])}"
                        )
                    elif isinstance(mod, nn.Linear):
                        rec["macs"] = mod.in_features * mod.out_features
                        rec["details"] = f"in={mod.in_features}, out={mod.out_features}"

                return hook_fn

            hooks.append(module.register_forward_hook(_create_hook(record, module)))
            layer_records.append(record)

    model.eval()
    with torch.no_grad():
        _ = model(input_board)

    for h in hooks:
        h.remove()

    total_params = sum(p.numel() for p in model.parameters())
    total_macs = sum(r["macs"] for r in layer_records)

    return {
        "layers": layer_records,
        "total_params": total_params,
        "total_macs": total_macs,
    }


def group_layer_stats(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """依 PREREG 結構分組統計：L1 conv, L2 conv, head 第一層, head 第二層。"""
    groups: dict[str, dict[str, int]] = {
        "L1 conv（h、v）": {"params": 0, "macs": 0},
        "L2 conv（4 個）": {"params": 0, "macs": 0},
        "head 第一層": {"params": 0, "macs": 0},
        "head 第二層": {"params": 0, "macs": 0},
    }

    for r in records:
        name = r["name"]
        if "conv_h" in name or "conv_v" in name:
            if any(k in name for k in ("conv_hh", "conv_hv", "conv_vh", "conv_vv")):
                groups["L2 conv（4 個）"]["params"] += r["params"]
                groups["L2 conv（4 個）"]["macs"] += r["macs"]
            else:
                groups["L1 conv（h、v）"]["params"] += r["params"]
                groups["L1 conv（h、v）"]["macs"] += r["macs"]
        elif "head.0" in name:
            groups["head 第一層"]["params"] += r["params"]
            groups["head 第一層"]["macs"] += r["macs"]
        elif "head.2" in name:
            groups["head 第二層"]["params"] += r["params"]
            groups["head 第二層"]["macs"] += r["macs"]

    return groups


def format_markdown(
    stats_c: dict[str, Any],
    stats_s: dict[str, Any] | None,
    s_status: str,
) -> str:
    """生成 Markdown 表格報告。"""
    grouped_c = group_layer_stats(stats_c["layers"])
    grouped_s = group_layer_stats(stats_s["layers"]) if stats_s is not None else None

    lines: list[str] = []
    lines.append("# 2048 OneHotValueNet 參數量與 FLOPs（MACs）核對報告\n")
    lines.append(f"- **架構 C 狀態**：正常 (預設 `directional`)")
    lines.append(f"- **架構 S 狀態**：{s_status}")
    lines.append("- **輸入形狀**：(1, 16) 單一盤面\n")

    lines.append("## 1. 事前登記（S_PREREG.md）比對摘要表\n")
    lines.append(
        "| 部分 | C (Baseline) | S (Skip L1) | S/C 參數比 | S/C MACs 比 | PREREG 核對 |"
    )
    lines.append(
        "|:---|---:|---:|---:|---:|:---:|"
    )

    all_matched = True
    sections = ["L1 conv（h、v）", "L2 conv（4 個）", "head 第一層", "head 第二層"]

    for sec in sections:
        c_p = grouped_c[sec]["params"]
        c_m = grouped_c[sec]["macs"]
        c_prereg_p = PREREG_BENCHMARK["C"]["groups"][sec]["params"]
        c_prereg_m = PREREG_BENCHMARK["C"]["groups"][sec]["macs"]
        c_match = (c_p == c_prereg_p and c_m == c_prereg_m)

        if grouped_s is not None:
            s_p = grouped_s[sec]["params"]
            s_m = grouped_s[sec]["macs"]
            s_prereg_p = PREREG_BENCHMARK["S"]["groups"][sec]["params"]
            s_prereg_m = PREREG_BENCHMARK["S"]["groups"][sec]["macs"]
            s_match = (s_p == s_prereg_p and s_m == s_prereg_m)
            match_str = "✓ 一致" if (c_match and s_match) else "✗ 不符"
            if not (c_match and s_match):
                all_matched = False

            p_ratio = f"{s_p / c_p:.2f}x"
            m_ratio = f"{s_m / c_m:.2f}x"
            s_str = f"{s_p:,} / {s_m:,}"
        else:
            match_str = "✓ C 一致 (S 尚未實作)" if c_match else "✗ C 不符"
            if not c_match:
                all_matched = False
            p_ratio = "-"
            m_ratio = "-"
            s_str = "尚未實作"

        c_str = f"{c_p:,} / {c_m:,}"
        lines.append(f"| {sec} | {c_str} | {s_str} | {p_ratio} | {m_ratio} | {match_str} |")

    # 合計列
    tot_c_p = stats_c["total_params"]
    tot_c_m = stats_c["total_macs"]
    tot_c_match = (
        tot_c_p == PREREG_BENCHMARK["C"]["total_params"]
        and tot_c_m == PREREG_BENCHMARK["C"]["total_macs"]
    )

    if stats_s is not None:
        tot_s_p = stats_s["total_params"]
        tot_s_m = stats_s["total_macs"]
        tot_s_match = (
            tot_s_p == PREREG_BENCHMARK["S"]["total_params"]
            and tot_s_m == PREREG_BENCHMARK["S"]["total_macs"]
        )
        tot_match = "✓ 完全一致" if (tot_c_match and tot_s_match) else "✗ 存在差異"
        if not (tot_c_match and tot_s_match):
            all_matched = False
        tot_p_ratio = f"{tot_s_p / tot_c_p:.2f}x"
        tot_m_ratio = f"{tot_s_m / tot_c_m:.2f}x"
        tot_s_str = f"**{tot_s_p:,} / {tot_s_m:,}**"
    else:
        tot_match = "✓ C 完全一致" if tot_c_match else "✗ C 不符"
        if not tot_c_match:
            all_matched = False
        tot_p_ratio = "-"
        tot_m_ratio = "-"
        tot_s_str = "**尚未實作**"

    tot_c_str = f"**{tot_c_p:,} / {tot_c_m:,}**"
    lines.append(
        f"| **合計** | {tot_c_str} | {tot_s_str} | **{tot_p_ratio}** | **{tot_m_ratio}** | **{tot_match}** |"
    )

    lines.append("\n*註：欄位數值格式為 `參數量 / MACs`。*")
    lines.append(
        f"- 事前登記 C 基準：{PREREG_BENCHMARK['C']['total_params']:,} 參數、{PREREG_BENCHMARK['C']['total_macs']:,} MACs"
    )
    lines.append(
        f"- 事前登記 S 基準：{PREREG_BENCHMARK['S']['total_params']:,} 參數 (x2.24)、{PREREG_BENCHMARK['S']['total_macs']:,} MACs (x1.65)\n"
    )

    # 各層明細表
    lines.append("## 2. 各層詳細計算明細表\n")
    lines.append(
        "| 層名稱 | 類型 | 規格細節 | C 參數 | C MACs | S 參數 | S MACs | S/C 參數比 | S/C MACs 比 |"
    )
    lines.append(
        "|:---|:---|:---|---:|---:|---:|---:|---:|---:|"
    )

    s_layer_map = {r["name"]: r for r in stats_s["layers"]} if stats_s is not None else {}

    for r_c in stats_c["layers"]:
        name = r_c["name"]
        mtype = r_c["module_type"]
        details = r_c["details"]
        c_p = r_c["params"]
        c_m = r_c["macs"]

        if name in s_layer_map:
            r_s = s_layer_map[name]
            s_p = r_s["params"]
            s_m = r_s["macs"]
            s_p_str = f"{s_p:,}"
            s_m_str = f"{s_m:,}"
            p_rat = f"{s_p / c_p:.2f}x"
            m_rat = f"{s_m / c_m:.2f}x"
        else:
            s_p_str = "-"
            s_m_str = "-"
            p_rat = "-"
            m_rat = "-"

        lines.append(
            f"| `{name}` | {mtype} | {details} | {c_p:,} | {c_m:,} | {s_p_str} | {s_m_str} | {p_rat} | {m_rat} |"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="2048 OneHotValueNet 參數量與 FLOPs（MACs）計算工具")
    parser.add_argument(
        "--json",
        nargs="?",
        const="params_flops.json",
        type=str,
        default=None,
        help="將結果儲存為 JSON 檔案（可選指定輸出路徑，預設為 params_flops.json）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="運算裝置 (預設: cpu)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model_c, model_s, s_status = build_models()
    model_c.to(device)
    if model_s is not None:
        model_s.to(device)

    # 輸入單一盤面 (1, 16)
    dummy_input = torch.zeros((1, 16), dtype=torch.float32, device=device)

    stats_c = profile_network(model_c, dummy_input)
    stats_s = profile_network(model_s, dummy_input) if model_s is not None else None

    # 輸出 Markdown 報告
    md_report = format_markdown(stats_c, stats_s, s_status)
    print(md_report)

    # 若指定 --json，儲存結構化資訊
    if args.json is not None:
        out_path = Path(args.json)
        json_data: dict[str, Any] = {
            "s_status": s_status,
            "prereg_benchmark": PREREG_BENCHMARK,
            "model_c": {
                "total_params": stats_c["total_params"],
                "total_macs": stats_c["total_macs"],
                "grouped": group_layer_stats(stats_c["layers"]),
                "layers": stats_c["layers"],
            },
            "model_s": (
                {
                    "total_params": stats_s["total_params"],
                    "total_macs": stats_s["total_macs"],
                    "grouped": group_layer_stats(stats_s["layers"]),
                    "layers": stats_s["layers"],
                }
                if stats_s is not None
                else None
            ),
        }
        if stats_s is not None:
            json_data["ratios"] = {
                "params_ratio": stats_s["total_params"] / stats_c["total_params"],
                "macs_ratio": stats_s["total_macs"] / stats_c["total_macs"],
            }

        out_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[INFO] 已將結構化統計儲存至：{out_path.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
