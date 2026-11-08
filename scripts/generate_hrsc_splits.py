import os
from pathlib import Path

img_dir = Path(r'H:\part_time\开源复现\code\data\hrsc\FullDataSet\AllImages')
out_dir = Path(r'H:\part_time\开源复现\code\data\hrsc\ImageSets')
out_dir.mkdir(parents=True, exist_ok=True)

# 收集所有图像文件名（不带扩展名）
stems = sorted([
    p.stem for p in img_dir.iterdir()
    if p.suffix.lower() in {'.bmp', '.jpg', '.jpeg', '.png'}
])
print(f'找到 {len(stems)} 张图像')

# 全部写入 trainval.txt
with open(out_dir / 'trainval.txt', 'w', encoding='utf-8') as f:
    for s in stems:
        f.write(s + '\n')

# 同时生成 test.txt，防止某些配置要求
with open(out_dir / 'test.txt', 'w', encoding='utf-8') as f:
    for s in stems:
        f.write(s + '\n')

print(f'已生成:')
print(f'  {out_dir / "trainval.txt"}')
print(f'  {out_dir / "test.txt"}')