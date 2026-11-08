#!/usr/bin/env python3
"""
8 组消融实验自动运行脚本。
3 个核心模块全组合：CR Loss / APS / DRoI
用法:
    python run_ablation.py              # 默认 12 轮
    python run_ablation.py --epochs 4   # 快速验证 4 轮
    python run_ablation.py --combos 000,111  # 只跑指定组
    python run_ablation.py --skip-existing   # 跳过已有结果
"""
import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
MMROTATE_DIR = CODE_DIR / "mmrotate"
CONFIG = "configs/roi_trans/roi-trans-le90_r50_fpn_1x_hrsc.py"
WORK_ROOT = MMROTATE_DIR / "work_dirs" / "ablation"
RESULTS_FILE = WORK_ROOT / "results.json"

# (组名, use_cr_loss, use_aps, use_droi)
COMBOS = [
    ("000_baseline",     False, False, False),
    ("001_droi_only",    False, False, True),
    ("010_aps_only",     False, True,  False),
    ("011_aps_droi",     False, True,  True),
    ("100_crloss_only",  True,  False, False),
    ("101_crloss_droi",  True,  False, True),
    ("110_crloss_aps",   True,  True,  False),
    ("111_full",         True,  True,  True),
]

MAP_PATTERN = re.compile(r"dota/mAP:\s*([0-9.]+)")
RECALL_PATTERN = re.compile(r"recall:\s*([0-9.]+)")


def load_results():
    if RESULTS_FILE.exists():
        try:
            return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_results(results):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def run_command(cmd, log_file: Path):
    """运行命令，实时打印并写日志，返回 (returncode, full_log_text)"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    with open(log_file, "w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.Popen(
            cmd, cwd=str(MMROTATE_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
            lines.append(line)
        proc.wait()
    return proc.returncode, "".join(lines)


def extract_final_map(log_text: str):
    """从日志中取最后一次 Epoch(val) 的 mAP 和 recall"""
    map_vals = MAP_PATTERN.findall(log_text)
    recall_vals = RECALL_PATTERN.findall(log_text)
    final_map = float(map_vals[-1]) if map_vals else None
    final_recall = float(recall_vals[-1]) if recall_vals else None
    return final_map, final_recall


def build_train_cmd(name, use_cr, use_aps, use_droi, epochs):
    work_dir = WORK_ROOT / name
    # 读取基础 config，替换开关值，写入临时 config
    base_text = (MMROTATE_DIR / CONFIG).read_text(encoding="utf-8")
    for var, val in [
        ("use_cr_loss", use_cr),
        ("use_aps_roi_align", use_aps),
        ("use_aps_fusion", use_aps),
        ("use_aps_fine_sampling", use_aps),
        ("use_droi", use_droi),
    ]:
        base_text = re.sub(
            rf"^{var}\s*=\s*(True|False)", f"{var} = {val}", base_text, flags=re.M)
    # R-IoULoss 需要绝对坐标，让第二阶段 bbox_head 解码后再回归
    base_text = base_text.replace(
        "reg_class_agnostic=False,",
        "reg_class_agnostic=False,\n                reg_decoded_bbox=True,", 1
    )
    tmp_cfg_dir = MMROTATE_DIR / "configs" / "roi_trans"
    tmp_cfg = tmp_cfg_dir / f"_tmp_ablation_{name}.py"
    tmp_cfg.write_text(base_text, encoding="utf-8")
    # 临时 config 用相对 mmrotate 的路径
    rel_cfg = f"configs/roi_trans/_tmp_ablation_{name}.py"
    cmd = [
        sys.executable, "tools/train.py", rel_cfg,
        "--work-dir", str(work_dir),
        "--cfg-options",
        f"train_cfg.max_epochs={epochs}",
    ]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--combos", type=str, default=None,
                        help="逗号分隔的组名前缀，如 000,111")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过 results.json 中已有结果的组")
    args = parser.parse_args()

    if not (MMROTATE_DIR / CONFIG).exists():
        print(f"[FAIL] 找不到配置文件: {MMROTATE_DIR / CONFIG}")
        sys.exit(1)

    selected = COMBOS
    if args.combos:
        wanted = set(s.strip() for s in args.combos.split(","))
        selected = [c for c in COMBOS if c[0].split("_")[0] in wanted]
        if not selected:
            print(f"[FAIL] 没有任何匹配的组: {args.combos}")
            sys.exit(1)

    results = load_results()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"消融实验计划：{len(selected)} 组，每组 {args.epochs} 轮")
    print(f"工作目录：{WORK_ROOT}")
    print("=" * 70)
    for name, cr, aps, droi in selected:
        print(f"  {name:20s}  CR={cr!s:5s}  APS={aps!s:5s}  DRoI={droi!s:5s}")

    start_all = time.time()
    for idx, (name, cr, aps, droi) in enumerate(selected, 1):
        print("\n" + "=" * 70)
        print(f"[{idx}/{len(selected)}] {name}")
        print("=" * 70)

        if args.skip_existing and name in results and results[name].get("mAP") is not None:
            print(f"[SKIP] 已有结果 mAP={results[name]['mAP']}")
            continue

        work_dir = WORK_ROOT / name
        log_file = WORK_ROOT / f"{name}.log"
        cmd = build_train_cmd(name, cr, aps, droi, args.epochs)

        print("命令:", " ".join(cmd))
        t0 = time.time()
        rc, log_text = run_command(cmd, log_file)
        elapsed = time.time() - t0

        if rc != 0:
            print(f"[FAIL] {name} 退出码 {rc}，日志: {log_file}")
            results[name] = {
                "use_cr_loss": cr, "use_aps": aps, "use_droi": droi,
                "mAP": None, "error": f"exit code {rc}",
                "log": str(log_file), "time_sec": round(elapsed, 1),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            save_results(results)
            continue

        final_map, final_recall = extract_final_map(log_text)
        best_ckpt = None
        ckpts = sorted(work_dir.glob("best_mAP_*.pth"))
        if ckpts:
            best_ckpt = str(ckpts[-1])
        elif (work_dir / "last.pth").exists():
            best_ckpt = str(work_dir / "last.pth")

        results[name] = {
            "use_cr_loss": cr, "use_aps": aps, "use_droi": droi,
            "mAP": final_map, "recall": final_recall,
            "best_ckpt": best_ckpt,
            "log": str(log_file),
            "time_sec": round(elapsed, 1),
            "epochs": args.epochs,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        save_results(results)
        print(f"[DONE] {name}  mAP={final_map}  recall={final_recall}  用时 {elapsed/60:.1f} 分钟")

    total = time.time() - start_all
    print("\n" + "=" * 70)
    print(f"全部完成，总用时 {total/3600:.2f} 小时")
    print("=" * 70)
    print(f"{'组名':<20s} {'CR':<5s} {'APS':<5s} {'DRoI':<5s} {'mAP':<8s} {'recall':<8s}")
    print("-" * 70)
    for name, cr, aps, droi in COMBOS:
        if name not in results:
            continue
        r = results[name]
        map_str = f"{r['mAP']:.4f}" if r.get("mAP") is not None else "FAIL"
        rec_str = f"{r['recall']:.4f}" if r.get("recall") is not None else "-"
        print(f"{name:<20s} {str(cr):<5s} {str(aps):<5s} {str(droi):<5s} {map_str:<8s} {rec_str:<8s}")
    print(f"\n结果已保存到: {RESULTS_FILE}")

    # 清理临时 config
    for f in (MMROTATE_DIR / "configs" / "roi_trans").glob("_tmp_ablation_*.py"):
        try:
            f.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()