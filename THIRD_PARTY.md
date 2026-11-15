# Third-Party Software and Upstream Licenses

## 1. Modified Upstream File Included in This Repository

- **`01_core_operators/matlab_reticolo/res2_reyy.m`** — **CC BY 4.0 (upstream), modified**
  - Upstream source: `res2.m` from **RETICOLO V10** by Jean-Paul Hugonin and Philippe Lalanne (Institut d'Optique / CNRS; DOI: [`10.5281/zenodo.14631951`](https://doi.org/10.5281/zenodo.14631951), [`arXiv:2101.00901`](https://arxiv.org/abs/2101.00901)), licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
  - Modification: returns the full four-block scattering matrix and port modes for downstream Redheffer cascades and lateral phase shifts. The adaptation omits exterior-port selection and diffraction-efficiency postprocessing.
  - RETICOLO itself (`res0`, `res1`, `res2`, `retio`, `retb`, `retc`, `retss`, `retsp`, etc.) is **not** redistributed here and must be installed separately by the user.

---

## 2. External Solvers and Runtime Libraries (Not Redistributed)

All external solvers and Python/MATLAB packages below are installed separately by the user and remain governed by their respective upstream licenses.

| Component | Authors / Maintainers | Tested Version | License | Reference / URL |
| :--- | :--- | :--- | :--- | :--- |
| **`res2_reyy.m`** (bundled) | Adapted from RETICOLO `res2.m` (J.-P. Hugonin & P. Lalanne) | RETICOLO V10 | CC BY 4.0 (upstream), modified | DOI: [`10.5281/zenodo.14631951`](https://doi.org/10.5281/zenodo.14631951) |
| **RETICOLO** (external) | J.-P. Hugonin & P. Lalanne (Institut d'Optique / CNRS) | V10 | CC BY 4.0 | DOI: [`10.5281/zenodo.14631951`](https://doi.org/10.5281/zenodo.14631951) · [`arXiv:2101.00901`](https://arxiv.org/abs/2101.00901) |
| **FMMAX** (external) | M. F. Schubert / Meta Platforms, Inc. | `1.7.1` | MIT | [`github.com/facebookresearch/fmmax`](https://github.com/facebookresearch/fmmax) |
| **JAX & jaxlib** | The JAX Authors / Google LLC | `0.10.2` | Apache-2.0 | [`github.com/jax-ml/jax`](https://github.com/jax-ml/jax) |
| **PyTorch** | PyTorch Contributors / Linux Foundation | `2.7.1` (CUDA 12.8) / `2.11.0` | BSD-3-Clause | [`github.com/pytorch/pytorch`](https://github.com/pytorch/pytorch) |
| **NumPy** | NumPy Developers | `1.26.4` / `2.4.6` / `2.5.4` | BSD-3-Clause | [`numpy.org`](https://numpy.org/) |
| **SciPy** | SciPy Developers | `1.13.1` / `1.17.1` / `1.18.1` | BSD-3-Clause | [`scipy.org`](https://scipy.org/) |
| **psutil** | Giampaolo Rodola | `7.2.2` | BSD-3-Clause | [`github.com/giampaolo/psutil`](https://github.com/giampaolo/psutil) |

---

## 3. Commercial Runtimes (User-Supplied)

- **MATLAB** (The MathWorks, Inc., tested with R2026a/R2026b; R2022a+ supported for `pageinv` / `pagemldivide`): required to run `.m` scripts and figure plotting routines. Parallel Computing Toolbox is used only by `parfor` and `gpuArray` benchmark scripts in `03_paper_reproduction/`.
- **COMSOL Multiphysics** (COMSOL AB): not required to run any script in this repository or redraw Figure 5 from the bundled FEM reference exports.
