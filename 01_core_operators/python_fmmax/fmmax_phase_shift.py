from dataclasses import replace
import jax.numpy as jnp
from fmmax import basis, fmm, scattering


def _phase(result: fmm.LayerSolveResult, dx):
    kx = basis.transverse_wavevectors(
        result.in_plane_wavevector,
        result.primitive_lattice_vectors,
        result.expansion,
    )[..., 0]
    p = jnp.exp(-1j * jnp.asarray(dx)[..., None] * kx)
    return p, jnp.concatenate([p, p], axis=-1)


def _similarity(matrix, p):
    return p[..., :, None] * matrix * jnp.conj(p[..., None, :])


def shift_layer_solve_result(result: fmm.LayerSolveResult, dx):
    """Translate a 1D FMMAX layer eigensolution by dx without recomputing EVD."""
    p, p2 = _phase(result, dx)
    updates = {"eigenvectors": p2[..., :, None] * result.eigenvectors}

    for name in (
        "z_permittivity_matrix",
        "inverse_z_permittivity_matrix",
        "z_permeability_matrix",
        "inverse_z_permeability_matrix",
    ):
        updates[name] = _similarity(getattr(result, name), p)

    for name in (
        "omega_script_k_matrix",
        "transverse_permittivity_matrix",
        "transverse_permeability_matrix",
    ):
        updates[name] = _similarity(getattr(result, name), p2)

    if result.tangent_vector_field is not None:
        period = result.primitive_lattice_vectors.u[0]
        nx = result.tangent_vector_field[0].shape[-2]
        spatial_phase = jnp.exp(-2j * jnp.pi * jnp.fft.fftfreq(nx) * dx * nx / period)
        shifted = []
        for field_arr in result.tangent_vector_field:
            field_sft = jnp.fft.ifft(
                jnp.fft.fft(field_arr, axis=-2) * spatial_phase[:, None], axis=-2
            )
            if not jnp.iscomplexobj(field_arr):
                field_sft = field_sft.real
            shifted.append(field_sft)
        updates["tangent_vector_field"] = tuple(shifted)

    return replace(result, **updates)


def shift_scattering_matrix(s: scattering.ScatteringMatrix, dx):
    """Translate a 1D subsystem ScatteringMatrix with uniform exterior ports by dx."""
    _, p_start = _phase(s.start_layer_solve_result, dx)
    _, p_end = _phase(s.end_layer_solve_result, dx)

    def shift_block(block, p_out, p_in):
        return p_out[..., :, None] * block * jnp.conj(p_in[..., None, :])

    return replace(
        s,
        s11=shift_block(s.s11, p_end, p_start),
        s12=shift_block(s.s12, p_end, p_end),
        s21=shift_block(s.s21, p_start, p_start),
        s22=shift_block(s.s22, p_start, p_end),
    )
