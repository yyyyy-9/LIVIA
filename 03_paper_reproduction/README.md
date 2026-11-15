# 论文图 2–5 与表 1 复现说明

请在仓库根目录下运行以下命令。绘图脚本默认读取各子目录 `reference/` 或 `outputs/` 下随附的参考数据，并将生成的矢量图输出至对应 `outputs/` 目录。

## 1. 基于随附参考数据直接出图（MATLAB）

```matlab
setenv('LIVIA_PLOT_GENERATED', '');
setenv('LIVIA_QUICK_DEMO', '');
run('03_paper_reproduction/fig2_euv_multilayer/panels_final.m');
run('03_paper_reproduction/fig3_mldg_kspace_diag/FIGPLOT3.m');
run('03_paper_reproduction/fig4_mldg_tolerance/Compare_B_R.m');
run('03_paper_reproduction/fig5_fwbic_3d_sweep/Fig6re.m');
```

| 论文图号 | 绘图脚本 | 输出 PDF |
| :--- | :--- | :--- |
| 图 2 · EUV 多层膜筛选与优化 | `fig2_euv_multilayer/panels_final.m` | `Figure_Mechanism.pdf` |
| 图 3 · MLDG 波矢空间通道诊断 | `fig3_mldg_kspace_diag/FIGPLOT3.m` | `Figure4_Composite_Raw.pdf` |
| 图 4 · MLDG 双层厚度容差扫描 | `fig4_mldg_tolerance/Compare_B_R.m` | `Figure4_MLDGv1.pdf` |
| 图 5 · 双倾斜光栅 FW-BIC 扫描 | `fig5_fwbic_3d_sweep/Fig6re.m` | `Fig6_REALL.pdf` |

## 2. 正文表 1 各算例对应脚本与原始运行记录

论文正文表 1 与 SI S9 的完整规模绝对耗时在 AMD Ryzen 9 9950X3D（16 核）+ NVIDIA RTX 5080 工作站上测得；各行对应的复现脚本与随附参考记录如下：

| 表 1 算例与工作负载 | 论文报告指标 | 复现入口脚本 | 随附参考数据 / 计时记录 |
| :--- | :--- | :--- | :--- |
| MLDG · 20,000 次级联（$N=100$，对照完整算子 RETICOLO） | 1,623.1 s vs. 104.5 s (CPU) / 41.4 s (GPU)，$15.5\times$ / $39.2\times$ | `fig4_mldg_tolerance/Lis_mldg_parforsweep_allorder.m`、`Lis_mldg_Batched_cacu_TETM.m`、`Lis_mldg_Batched_cacu_TETM_GPUTEST.m` | `fig4_mldg_tolerance/Compare_B_R.m`、`reference/Batched_Cascade2.mat` |
| MLDG · 20,000 次级联（$N=100$，对照外端口选通原生 RETICOLO） | 786.3 s vs. 104.5 s (CPU) / 41.4 s (GPU)，$7.5\times$ / $19.0\times$ | `fig4_mldg_tolerance/Lis_mldg_parforsweep.m`、`Lis_mldg_Batched_cacu_TETM.m`、`Lis_mldg_Batched_cacu_TETM_GPUTEST.m` | `fig4_mldg_tolerance/reference/RETICOLOres.mat`、`reference/Batched_Cascade2.mat` |
| MLDG · 1,000 组完整四分块 $\mathbf{S}$（$N=100$，FMMAX 同 CPU） | 432.316 s vs. 15.755 s，$27.44\times$（100 态为 $27.87\times$） | `fig4_mldg_tolerance/benchmark_mldg_jax.py` | `fig4_mldg_tolerance/inputs/mldg100.npz`、`inputs/mldg1000.npz`、`reference/fmmax_cpu_timings.csv` |
| 双倾斜光栅 · 1,001 个横向位移（$N=50, L=100$，FMMAX 同 CPU） | 903.463 s vs. 4.807 s，$187.95\times$ | `fig5_fwbic_3d_sweep/benchmark_shift_scaling.py` | `fig5_fwbic_3d_sweep/reference/fmmax_shift_scaling.csv` |
| 双倾斜光栅 · $9.81\times 10^6$ 波长–倾角–位移态（RETICOLO CPU） | 2.43 h（TE 零级透射） | `fig5_fwbic_3d_sweep/SWEEP3D.m` | `fig5_fwbic_3d_sweep/reference/figure5_slices.mat` |

## 3. 重新运行数值扫描与基准测试

### 图 2 · EUV 多层膜筛选与优化（Python / PyTorch）
```bash
python 03_paper_reproduction/fig2_euv_multilayer/check_periodic_baseline.py
python 03_paper_reproduction/fig2_euv_multilayer/reproduce_euv.py --phase verify
# 完整前向筛选 + AD 优化 + AD/FD 梯度校验：
# python 03_paper_reproduction/fig2_euv_multilayer/reproduce_euv.py --phase all
```

### 图 3 · MLDG 波长与膜厚缩放诊断（MATLAB + RETICOLO）
```matlab
setup_paths('/path/to/reticolo_v10');
run('03_paper_reproduction/fig3_mldg_kspace_diag/Lis_mldg_wvl.m');
run('03_paper_reproduction/fig3_mldg_kspace_diag/Bloch_test_wvl.m');
run('03_paper_reproduction/fig3_mldg_kspace_diag/Lis_mldg_layertest.m');
run('03_paper_reproduction/fig3_mldg_kspace_diag/Bloch_test.m');
setenv('LIVIA_PLOT_GENERATED', '1');
run('03_paper_reproduction/fig3_mldg_kspace_diag/FIGPLOT3.m');
```

### 图 4 与表 1（MLDG 20,000 次级联）· 厚度容差扫描（MATLAB + RETICOLO）
设置 `LIVIA_QUICK_DEMO=1` 可先做 `5 × 5` 小网格快速验证；清空该变量则运行完整 `100 × 100` 网格：
```matlab
setenv('LIVIA_QUICK_DEMO', '1');
run('03_paper_reproduction/fig4_mldg_tolerance/Lis_mldg_parforsweep.m');
run('03_paper_reproduction/fig4_mldg_tolerance/Lis_mldg_parforsweep_allorder.m');
run('03_paper_reproduction/fig4_mldg_tolerance/Lis_mldg_Batched_cacu_TETM.m');
run('03_paper_reproduction/fig4_mldg_tolerance/Lis_mldg_Batched_cacu_TETM_GPUTEST.m');
setenv('LIVIA_PLOT_GENERATED', '1');
run('03_paper_reproduction/fig4_mldg_tolerance/Compare_B_R.m');
```

### 表 1（MLDG 100 / 1,000 态）与 SI S9 · FMMAX 膜系递推基准（Python / JAX）
```bash
cd 03_paper_reproduction/fig4_mldg_tolerance
python benchmark_mldg_jax.py --tag mldg100 --skip-save-full
python benchmark_mldg_jax.py --tag mldg1000 --input inputs/mldg1000.npz --configs native:4:4 cached_analytic_film:100:1 --skip-save-full
```

### 表 1（双光栅 1,001 位移）与 SI S9 · FMMAX 相对位移规模基准（Python / JAX）
```bash
cd 03_paper_reproduction/fig5_fwbic_3d_sweep
# 小规模快速精度与链路核验（N=4, L=6, M=1,5）：
python benchmark_shift_scaling.py --tag quick_check --orders 4 --layers 6 --grid 384 --counts 1 5 --repeats 1 --skip-save-full
# 完整论文规模（N=50, L=100, M=1..1001，对应表 1 中 187.95x 行）：
python benchmark_shift_scaling.py --tag cpu_fixed60_wl1610_N50_L100 --skip-save-full
```

### 图 5 与表 1（9.81M 态）· 双倾斜光栅三维参数扫描（MATLAB + RETICOLO）
设置 `LIVIA_QUICK_DEMO=1` 可做 `5 × 7 × 51` 快速验证；清空则运行完整 `81 × 121 × 1,001` 扫描：
```matlab
setenv('LIVIA_QUICK_DEMO', '1');
run('03_paper_reproduction/fig5_fwbic_3d_sweep/SWEEP3D.m');
run('03_paper_reproduction/fig5_fwbic_3d_sweep/wavesweep2.m');
setenv('LIVIA_PLOT_GENERATED', '1');
run('03_paper_reproduction/fig5_fwbic_3d_sweep/Fig6re.m');
```
