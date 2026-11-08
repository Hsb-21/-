#!/usr/bin/env python3
"""
检查 HRSC2016 和 DOTA 数据集是否符合 MMRotate 的基本要求。
用法:
    python check_datasets.py
    python check_datasets.py --root ./data
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_dirs_with_ext(root: Path, ext: str, min_count: int = 1):
    """递归查找包含指定扩展名文件数量 >= min_count 的目录。"""
    hits = []
    for dirpath, _, filenames in os.walk(root):
        n = sum(1 for f in filenames if Path(f).suffix.lower() == ext)
        if n >= min_count:
            hits.append((Path(dirpath), n))
    return hits


def find_image_dir(root: Path):
    """找包含最多图像文件的目录。"""
    best_dir, best_n = None, 0
    for dirpath, _, filenames in os.walk(root):
        n = sum(1 for f in filenames if Path(f).suffix.lower() in IMG_EXTS)
        if n > best_n:
            best_dir, best_n = Path(dirpath), n
    return best_dir, best_n


def check_hrsc(root: Path):
    print("\n" + "=" * 60)
    print("HRSC2016 检查")
    print("=" * 60)
    if not root.exists():
        print(f"[FAIL] 目录不存在: {root}")
        return False

    img_dir, n_img = find_image_dir(root)
    xml_dirs = find_dirs_with_ext(root, ".xml", min_count=1)
    if not xml_dirs:
        print(f"[FAIL] 在 {root} 下没有找到 .xml 标注文件")
        return False
    # 选 xml 最多的目录作为标注目录
    ann_dir, n_xml = max(xml_dirs, key=lambda x: x[1])

    print(f"图像目录: {img_dir}  (图像数: {n_img})")
    print(f"标注目录: {ann_dir}  (XML 数: {n_xml})")

    img_stems = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    xml_stems = {p.stem for p in ann_dir.iterdir() if p.suffix.lower() == ".xml"}
    common = img_stems & xml_stems

    print(f"图像/标注同名配对: {len(common)}")
    if len(common) == 0:
        print("[FAIL] 图像与标注没有任何同名配对，请检查目录是否选对")
        return False
    if len(common) < min(n_img, n_xml) * 0.95:
        print(f"[WARN] 配对率偏低: {len(common)} / min({n_img}, {n_xml})")

    # 抽样解析 XML，确认是旋转框 robndbox
    sample = sorted(common)[:20]
    tag_counter = Counter()
    parse_errors = 0
    has_robndbox = 0
    for stem in sample:
        xml_path = ann_dir / f"{stem}.xml"
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            parse_errors += 1
            continue
        for obj in tree.getroot().iter("object"):
            tag = None
            for child in obj:
                tag_counter[child.tag] += 1
                if child.tag == "robndbox":
                    tag = "robndbox"
            if tag == "robndbox":
                has_robndbox += 1

    print(f"抽样 XML 数: {len(sample)}，解析失败: {parse_errors}")
    print(f"抽样中包含 robndbox 的对象数: {has_robndbox}")
    print(f"对象子标签统计(抽样): {dict(tag_counter)}")

    ok = parse_errors == 0 and has_robndbox > 0
    if ok:
        print("[PASS] HRSC2016 基本符合 MMRotate 读取要求（旋转框 XML）")
    else:
        print("[FAIL] 未检测到 robndbox 旋转框标注，或 XML 解析失败过多")
    print(f"建议 data_root: {root}")
    return ok


def looks_like_dota_line(line: str):
    parts = line.strip().split()
    if len(parts) < 9:
        return False
    try:
        for v in parts[:8]:
            float(v)
        float(parts[8]) if len(parts) > 8 else None
        return True
    except ValueError:
        return False


def check_dota(root: Path):
    print("\n" + "=" * 60)
    print("DOTA 检查")
    print("=" * 60)
    if not root.exists():
        print(f"[FAIL] 目录不存在: {root}")
        return False

    # 找 labelTxt 目录：优先名字含 labelTxt，否则找包含 DOTA 格式 txt 最多的目录
    label_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        txts = [f for f in filenames if f.lower().endswith(".txt")]
        if not txts:
            continue
        # 抽样判断是否是 DOTA 标注
        good = 0
        checked = 0
        for f in txts[:5]:
            p = Path(dirpath) / f
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                    lines = [ln for ln in fh.read().splitlines() if ln.strip()]
                if not lines:
                    continue
                checked += 1
                if looks_like_dota_line(lines[0]):
                    good += 1
            except OSError:
                pass
        if checked > 0 and good / checked >= 0.6:
            label_dirs.append((Path(dirpath), len(txts)))

    if not label_dirs:
        print(f"[FAIL] 在 {root} 下没有找到符合 DOTA 格式的 .txt 标注")
        return False

    label_dir, n_txt = max(label_dirs, key=lambda x: x[1])
    print(f"标注目录: {label_dir}  (txt 数: {n_txt})")

    # 尝试找与 labelTxt 同级的 images 目录
    parent = label_dir.parent
    candidate_img_dirs = [parent / "images", parent / "image", parent]
    img_dir, n_img = None, 0
    for d in candidate_img_dirs:
        if d.exists():
            n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)
            if n > n_img:
                img_dir, n_img = d, n
    if img_dir is None:
        print(f"[WARN] 未在 {parent} 下找到同级 images 目录，退化为全局搜索")
        img_dir, n_img = find_image_dir(root)

    print(f"图像目录: {img_dir}  (图像数: {n_img})")

    txt_stems = {p.stem for p in label_dir.iterdir() if p.suffix.lower() == ".txt"}
    img_stems = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    common = txt_stems & img_stems
    print(f"图像/标注同名配对: {len(common)}")
    if len(common) == 0:
        print("[FAIL] 图像与标注没有同名配对，可能 images 与 labelTxt 不在同一层级")
    elif len(common) < min(n_img, n_txt) * 0.9:
        print(f"[WARN] 配对率偏低: {len(common)} / min({n_img}, {n_txt})")

    # 抽样检查图像尺寸
    need_split = False
    if HAS_PIL and common:
        sample = sorted(common)[:20]
        oversize = 0
        for stem in sample:
            # 找对应图像扩展名
            for ext in IMG_EXTS:
                p = img_dir / f"{stem}{ext}"
                if p.exists():
                    try:
                        with Image.open(p) as im:
                            w, h = im.size
                        if w > 1024 or h > 1024:
                            oversize += 1
                    except Exception:
                        pass
                    break
        print(f"抽样图像中超过 1024x1024 的数量: {oversize}/{len(sample)}")
        need_split = oversize > 0
    elif not HAS_PIL:
        print("[INFO] 未安装 Pillow，跳过图像尺寸检查")

    ok = len(common) > 0
    if ok:
        print("[PASS] DOTA 标注格式与目录基本可识别")
        if need_split:
            print("[ACTION] 原始图像较大，需要运行 MMRotate 的 img_split.py 切图后再训练")
        else:
            print("[INFO] 抽样图像尺寸不大，但仍建议按官方流程切图")
    else:
        print("[FAIL] DOTA 目录结构或配对不满足要求")
    print(f"建议原始根目录: {root}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data", help="数据根目录，默认 ./data")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    print(f"数据根目录: {root}")
    if not root.exists():
        print(f"[FAIL] 根目录不存在: {root}")
        sys.exit(1)

    # 允许 hrsc 目录名大小写变体
    hrsc_candidates = [root / "hrsc", root / "HRSC2016", root / "HRSC"]
    hrsc_root = next((p for p in hrsc_candidates if p.exists()), None)
    if hrsc_root is None:
        print("[WARN] 未找到 hrsc/HRSC2016 目录，跳过 HRSC 检查")
        hrsc_ok = False
    else:
        hrsc_ok = check_hrsc(hrsc_root)

    dota_candidates = [root / "DOTA", root / "dota"]
    dota_root = next((p for p in dota_candidates if p.exists()), None)
    if dota_root is None:
        print("[WARN] 未找到 DOTA 目录，跳过 DOTA 检查")
        dota_ok = False
    else:
        dota_ok = check_dota(dota_root)

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"HRSC2016: {'通过' if hrsc_ok else '未通过/未找到'}")
    print(f"DOTA:     {'通过' if dota_ok else '未通过/未找到'}")
    if hrsc_ok:
        print("\n下一步建议：先改 configs/_base_/datasets/hrsc.py 里的 data_root，跑 RoI Transformer 基线。")


if __name__ == "__main__":
    main()