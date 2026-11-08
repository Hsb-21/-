#!/usr/bin/env python3
"""把 patches/mmrotate/ 复制进 mmrotate 仓库。"""
import argparse
import re
import shutil
from pathlib import Path

PATCHES = Path(__file__).resolve().parent.parent / 'patches' / 'mmrotate'


def register(init_path, cls_name, import_line):
    text = init_path.read_text(encoding='utf-8')
    if cls_name in text:
        return
    if '__all__' in text:
        text = re.sub(r'__all__\s*=\s*\[([^\]]*)\]',
                      lambda m: '__all__ = [' + m.group(1).rstrip() + f", '{cls_name}']",
                      text, count=1)
    text += '\n' + import_line + '\n'
    init_path.write_text(text, encoding='utf-8')
    print(f'  register {cls_name} -> {init_path}')


def apply(mmrotate_dir):
    dst_root = mmrotate_dir / 'mmrotate'
    for src in (PATCHES / 'models').rglob('*.py'):
        rel = src.relative_to(PATCHES / 'models')
        dst = dst_root / 'models' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f'  copy models/{rel}')
    for src in (PATCHES / 'configs').rglob('*.py'):
        rel = src.relative_to(PATCHES / 'configs')
        dst = mmrotate_dir / 'configs' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f'  copy configs/{rel}')
    register(dst_root / 'models/losses/__init__.py',
             'R_IoULoss', 'from .r_iou_loss import R_IoULoss')
    register(dst_root / 'models/roi_heads/roi_extractors/__init__.py',
             'DRoIExtractor', 'from .droi_extractor import DRoIExtractor')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mmrotate-dir', default='./mmrotate')
    args = p.parse_args()
    d = Path(args.mmrotate_dir).resolve()
    if not (d / 'mmrotate').exists():
        raise SystemExit(f'{d} 不是 mmrotate 仓库根目录')
    apply(d)
    print('完成。还需手动注释：')
    print('  - site-packages/mmdet/__init__.py 中的 mmcv 版本 assert')
    print('  - <mmrotate>/mmrotate/__init__.py 中的 mmcv 版本 assert')


if __name__ == '__main__':
    main()
