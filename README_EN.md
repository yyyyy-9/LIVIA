# LIVIA: Wavevector-Decoupled Tensor Architecture for High-Throughput Nanophotonic Simulations

**语言 / Language:** [中文](README.md) | English

Code and reference data accompanying [*"Wavevector-Decoupled Tensor Architecture for High-Throughput Nanophotonic Simulations"*](https://doi.org/10.1364/opticaopen.32024484.v1) (Optica Open preprint, 2026, DOI: `10.1364/opticaopen.32024484.v1`).

## Structure

```text
LIVIA_OpenSource/
├── setup_paths.m                          # MATLAB path setup (reads RETICOLO_ROOT)
├── requirements-torch.txt                 # Python dependencies for PyTorch Tensor SMM & EUV
├── requirements-fmmax.txt                 # Python dependencies for FMMAX / JAX bridges
├── LICENSE                                # MIT License (original source code)
├── DATA_LICENSE.md                        # Bundled dataset & optical-constant licenses (CC BY 4.0 / CC0 1.0)
├── THIRD_PARTY.md                         # Upstream attributions & licenses (including res2_reyy.m)
├── CITATION.cff                           # Citation metadata
├── 01_core_operators/
│   ├── matlab_reticolo/                   # MATLAB Tensor SMM, batched cascades, 1D/2D RETICOLO bridges & phase shifts
│   ├── python_torch_smm/                  # PyTorch (complex128) & NumPy Tensor SMM and cascades
│   └── python_fmmax/                      # 1D/2D FMMAX adapters, two-level phase shifts & port reader
├── 02_minimal_demos/
│   ├── demo01_reticolo_bridge_and_shift.m # RETICOLO 1D/2D VPI, native-retb bridge & S-shift check
│   └── demo02_fmmax_two_level_reuse.py    # FMMAX 1D/2D bridge, native field-basis adapter & two-level shift
└── 03_paper_reproduction/
    ├── fig2_euv_multilayer/               # Fig. 2: EUV Mo/Si multilayer screening & AD refinement
    ├── fig3_mldg_kspace_diag/             # Fig. 3: MLDG k-space channel diagnostics
    ├── fig4_mldg_tolerance/               # Fig. 4 & Table 1 (MLDG): tolerance scan & FMMAX film benchmark
    └── fig5_fwbic_3d_sweep/               # Fig. 5 & Table 1 (dual grating): FW-BIC 3D sweep & FMMAX shift benchmark
```

## Quick Start

### 1. MATLAB + RETICOLO Bridge (R2022a+)
Install [RETICOLO V10](https://doi.org/10.5281/zenodo.14631951) and point `RETICOLO_ROOT` to the directory containing `res1.m` and `res2.m`:
```matlab
setenv('RETICOLO_ROOT', '/path/to/reticolo_v10');
setup_paths();
run('02_minimal_demos/demo01_reticolo_bridge_and_shift.m');
```

### 2. Python + FMMAX Bridge
```bash
pip install -r requirements-fmmax.txt
python 02_minimal_demos/demo02_fmmax_two_level_reuse.py
```

### 3. Python (PyTorch Tensor SMM & Fig. 2 Verification)
```bash
pip install -r requirements-torch.txt
python 03_paper_reproduction/fig2_euv_multilayer/check_periodic_baseline.py
python 03_paper_reproduction/fig2_euv_multilayer/reproduce_euv.py --phase verify
```

For the pinned package versions, see [FMMAX dependencies](requirements-lock-fmmax.txt) and [PyTorch dependencies](requirements-lock-torch.txt).

See [`03_paper_reproduction/README.md`](03_paper_reproduction/README.md) for commands to plot or recompute Figures 2–5 and Table 1.

## License & Acknowledgments

- **Source Code:** Original code is licensed under the [MIT License](LICENSE) (except `01_core_operators/matlab_reticolo/res2_reyy.m`, noted below).
- **Bundled Data:** Simulation reference data under `03_paper_reproduction/` are provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); tabulated optical constants in `get_Si_nk.m` (Vuye et al., 1993) and `get_SiO2_nk.m` (Malitson, 1965) are from [RefractiveIndex.INFO](https://refractiveindex.info/) ([CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)). See [DATA_LICENSE.md](DATA_LICENSE.md).
- **External Solvers & Derivative File:** [RETICOLO](https://doi.org/10.5281/zenodo.14631951) itself and [FMMAX](https://github.com/facebookresearch/fmmax) (MIT) are not redistributed and must be installed separately; `01_core_operators/matlab_reticolo/res2_reyy.m` is a modified excerpt of RETICOLO `res2.m` (J.-P. Hugonin & P. Lalanne) that returns the full scattering matrix and port modes and is provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) with attribution. See [THIRD_PARTY.md](THIRD_PARTY.md).
- **Acknowledgment:** The vectorized planar-stack S-matrix recursion was inspired by [PyMoosh](https://github.com/AnMoreau/PyMoosh) (D. Langevin, A. Moreau *et al.*, *J. Opt. Soc. Am. B* **41**, A67–A78, 2024), extending its spectral/angular vectorization to a structural-batch $\times$ transverse-wavevector tensor layout.
