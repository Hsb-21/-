# Reproduction: Defect Object Detection for Industrial Scenarios 面向工业场景的缺陷目标检测研究


## 三个模块

| 模块 | 说明 | 开关变量 |
|---|---|---|
| CR Loss | 交叉熵 + R-IoU 回归 | `use_cr_loss` |
| APS-RoI-Align | C2+C4 Inception 融合 + GAP+FC+Sigmoid 注意力 + 上采样细化 | `use_aps_roi_align`/`use_aps_fusion`/`use_aps_fine_sampling` |
| DRoI | 在 MMRotate 内置 RRoI 上做 FC 残差精调 | `use_droi` |

## 环境
- Python 3.7
- PyTorch 1.7.0 + CUDA 10.1
- MMCV-full 1.4.5+ (1.x版本)
- MMDetection 2.22.0+

## 安装
```bash
# 创建并激活虚拟环境
conda create -n openmmlab python=3.7 -y
conda activate openmmlab

# 安装 PyTorch 1.7.0 和 CUDA 10.1
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.1 -c pytorch -y

# 使用 MIM 安装 OpenMMLab 依赖
pip install openmim
mim install mmcv-full
mim install mmdet

# 安装 MMRotate
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .

## 版本断言绕过（必须）

MMCV 2.2.0 高于 mmdet/mmrotate 官方允许上限，需注释掉版本断言：

- `site-packages/mmdet/__init__.py` 中的 `assert (mmcv_version >= ...)` 整块
- `<mmrotate>/mmrotate/__init__.py` 中同样的 assert 块

`scripts/apply_patches.py` 会自动处理 mmrotate 那一侧；mmdet 需要手动改。

## 数据准备

从 [百度 AI Studio](https://aistudio.baidu.com/aistudio/datasetdetail/54106) 下载 HRSC2016，放到：

```
data/hrsc/
├── FullDataSet/
│   ├── AllImages/
│   └── Annotations/
└── ImageSets/
    ├── trainval.txt
    └── test.txt
```

`scripts/generate_hrsc_splits.py` 会从 AllImages 自动生成 ImageSets/*.txt。

## 复现步骤

```bash
git clone -b 1.x https://github.com/open-mmlab/mmrotate.git
python scripts/apply_patches.py --mmrotate-dir ./mmrotate
python scripts/generate_hrsc_splits.py
cd mmrotate
python ../scripts/run_ablation.py --epochs 12
```

结果在 `mmrotate/work_dirs/ablation/results.json`。

## 实现差异说明

- **R-IoU** 是连续近似（IoU proxy + SmoothL1），不是论文的像素级精确实现。
- **APS ③ 精细采样** 用 2x 上采样 + 深度可分离卷积 + 自适应池化近似论文采样网格。
- **DRoI** 在 MMRotate 已有 RRoI 之后追加 FC 残差（零初始化），启用初期等价基线。

## License

MIT
