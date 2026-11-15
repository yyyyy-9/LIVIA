# LIVIA：面向高通量微纳光子学仿真的波矢解耦张量架构

**语言 / Language:** 中文 | [English](README_EN.md)

本仓库包含论文 [*"Wavevector-Decoupled Tensor Architecture for High-Throughput Nanophotonic Simulations"*](https://doi.org/10.1364/opticaopen.32024484.v1)（Optica Open 公开预印本，2026，DOI: `10.1364/opticaopen.32024484.v1`）的核心算子、最小验证示例以及论文图表复现脚本与参考数据。

## 目录结构

```text
LIVIA_OpenSource/
├── setup_paths.m                          # MATLAB 路径初始化（支持环境变量 RETICOLO_ROOT）
├── requirements-torch.txt                 # Python 依赖（PyTorch 张量 SMM 与 EUV 算例）
├── requirements-fmmax.txt                 # Python 依赖（FMMAX / JAX 桥接与基准测试）
├── LICENSE                                # MIT 开源协议（源代码）
├── DATA_LICENSE.md                        # 附带仿真数据与折射率表协议说明（CC BY 4.0 / CC0 1.0）
├── THIRD_PARTY.md                         # 上游署名与依赖协议说明（含 res2_reyy.m 改写说明）
├── CITATION.cff                           # 文献引用元数据
├── 01_core_operators/                     # 核心算子与外部求解器桥接
│   ├── matlab_reticolo/                   # MATLAB 张量 SMM、批量级联、1D/2D RETICOLO 桥接与平移复用
│   ├── python_torch_smm/                  # PyTorch (complex128) 与 NumPy 张量 SMM 及级联算子
│   └── python_fmmax/                      # 1D/2D FMMAX 桥接、两级相移复用与级次读取
├── 02_minimal_demos/                      # 最小验证示例
│   ├── demo01_reticolo_bridge_and_shift.m # RETICOLO 1D/2D VPI、原生 retb 显式代换桥接与 S 平移验证
│   └── demo02_fmmax_two_level_reuse.py    # FMMAX 1D/2D 桥接、原生场转换代换与两级平移验证
└── 03_paper_reproduction/                 # 论文图 2–5 与表 1 复现脚本及参考数据
    ├── fig2_euv_multilayer/               # 图 2：EUV Mo/Si 多层膜筛选与自动微分优化
    ├── fig3_mldg_kspace_diag/             # 图 3：MLDG 波矢空间通道诊断
    ├── fig4_mldg_tolerance/               # 图 4 与表 1（MLDG）：厚度容差扫描及 FMMAX 膜系对照基准
    └── fig5_fwbic_3d_sweep/               # 图 5 与表 1（双光栅）：FW-BIC 三维扫描及 FMMAX 位移基准
```

## 快速开始

### 1. MATLAB + RETICOLO 桥接与复用（R2022a+）
请先安装 [RETICOLO V10](https://doi.org/10.5281/zenodo.14631951)，并将环境变量 `RETICOLO_ROOT` 指向包含 `res1.m`、`res2.m` 的目录：
```matlab
setenv('RETICOLO_ROOT', '/path/to/reticolo_v10');
setup_paths();
run('02_minimal_demos/demo01_reticolo_bridge_and_shift.m');
```

### 2. Python + FMMAX 桥接与复用
```bash
pip install -r requirements-fmmax.txt
python 02_minimal_demos/demo02_fmmax_two_level_reuse.py
```

### 3. Python（PyTorch 张量 SMM 与图 2 复现校验）
```bash
pip install -r requirements-torch.txt
python 03_paper_reproduction/fig2_euv_multilayer/check_periodic_baseline.py
python 03_paper_reproduction/fig2_euv_multilayer/reproduce_euv.py --phase verify
```

基准环境的精确版本见 [FMMAX 依赖](requirements-lock-fmmax.txt)和 [PyTorch 依赖](requirements-lock-torch.txt)。

论文图 2–5 及表 1 的直接出图与完整重算命令见 [`03_paper_reproduction/README.md`](03_paper_reproduction/README.md)。

## 开源协议与致谢

- **源代码**：原创代码采用 [MIT License](LICENSE)（`01_core_operators/matlab_reticolo/res2_reyy.m` 除外，见下条）。
- **附带数据**：`03_paper_reproduction/` 下的参考仿真数据采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)；`get_Si_nk.m`（Vuye et al., 1993）与 `get_SiO2_nk.m`（Malitson, 1965）中的折射率表取自 [RefractiveIndex.INFO](https://refractiveindex.info/)（[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)），详见 [DATA_LICENSE.md](DATA_LICENSE.md)。
- **外部求解器与衍生文件**：[RETICOLO](https://doi.org/10.5281/zenodo.14631951) 本体与 [FMMAX](https://github.com/facebookresearch/fmmax)（MIT）不在本仓库重分发，由用户自行安装；`01_core_operators/matlab_reticolo/res2_reyy.m` 为 RETICOLO `res2.m`（J.-P. Hugonin & P. Lalanne）的修改节选版，输出用于后续级联的完整散射矩阵和端口模态，省去外端口选通及衍射效率后处理，保留原作者署名并按上游 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 提供，详见 [THIRD_PARTY.md](THIRD_PARTY.md)。
- **致谢**：平面多层膜张量 SMM 的向量化递推思路受 [PyMoosh](https://github.com/AnMoreau/PyMoosh)（D. Langevin, A. Moreau *et al.*, *J. Opt. Soc. Am. B* **41**, A67–A78, 2024）启发，将其沿光谱/角度维度的向量化散射矩阵递推拓展为“结构批次 × 横向波矢通道”的张量形式。
