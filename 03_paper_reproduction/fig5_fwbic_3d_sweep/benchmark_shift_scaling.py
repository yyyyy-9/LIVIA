"""Dual-slanted-grating cavity: native geometry batches vs. two-level S reuse in FMMAX.

Fixed wavelength and slant angle; sweeps M relative lateral displacements of the
lower grating. Both paths retain all four (4N+2)x(4N+2) dual-polarization blocks
in complex128 on the host before extracting zero-order efficiencies.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter, process_time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false')
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import psutil
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import fmmax
from fmmax import basis, fmm, scattering

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FMMAX_CORE = REPO_ROOT / '01_core_operators' / 'python_fmmax'
if str(FMMAX_CORE) not in sys.path:
    sys.path.insert(0, str(FMMAX_CORE))

from fmmax_phase_shift import shift_layer_solve_result, shift_scattering_matrix

KEYS = ('s11', 's12', 's21', 's22')


@dataclass
class Config:
    orders: int = 50
    layers: int = 100
    grid: int = 6144
    period: float = 0.9
    height: float = 0.2
    cavity: float = 0.3
    fill: float = 1 / 3
    n_external: float = 1.45
    n_liquid: float = 1.331
    n_silicon: float = 3.47
    incidence_deg: float = 0.0


def blocks(s):
    return tuple(np.asarray(getattr(s, key)) for key in KEYS)


class Model:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.orders = np.array([0] + [v for m in range(1, cfg.orders + 1) for v in (m, -m)])
        self.M = len(self.orders)
        self.expansion = basis.Expansion(np.column_stack([self.orders, np.zeros(self.M, int)]))
        self.primitive = fmmax.LatticeVectors(
            u=jnp.asarray([cfg.period, 0.0], dtype=jnp.float64),
            v=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        )
        x = (np.arange(cfg.grid) + 0.5) / cfg.grid * cfg.period - cfg.period / 2
        self.seed_eps = np.where(
            np.abs(x) < cfg.fill * cfg.period / 2,
            cfg.n_silicon ** 2,
            cfg.n_liquid ** 2,
        )[:, None].astype(complex)

        def solve(wl, eps):
            kxy = jnp.asarray([
                2 * jnp.pi / wl * cfg.n_external * np.sin(np.deg2rad(cfg.incidence_deg)),
                0.0,
            ])
            return fmm.eigensolve_isotropic_media(wl, kxy, self.primitive, eps, self.expansion)

        self.solve = jax.jit(solve)
        self.shift_layer = jax.jit(shift_layer_solve_result)
        self._shift_s = jax.jit(shift_scattering_matrix)
        self.stack = jax.jit(lambda layers, ds: scattering.stack_s_matrix(
            layers, ds, force_x64_solve=True,
        ))
        self.star = jax.jit(lambda a, b: scattering.redheffer_star_product(
            a, b, force_x64_solve=True,
        ))

    def shifts(self, angle: float):
        c = self.cfg
        z = (np.arange(c.layers) + 0.5) * c.height / c.layers
        top = z / np.tan(np.deg2rad(angle))
        top -= top[0]
        bottom = -z / np.tan(np.deg2rad(angle))
        bottom -= bottom[-1]
        tp = np.rint(top / c.period * c.grid).astype(int)
        bp = np.rint(bottom / c.period * c.grid).astype(int)
        err = float(max(
            np.max(np.abs(top - tp * c.period / c.grid)),
            np.max(np.abs(bottom - bp * c.period / c.grid)),
        ))
        return tp, bp, err

    def native_modes(self, wl, pixels):
        result = [
            self.solve(jnp.asarray(wl), jnp.asarray(np.roll(self.seed_eps, int(p), axis=0)))
            for p in pixels
        ]
        jax.block_until_ready(result)
        return tuple(result)

    def phase_modes(self, seed, pixels):
        result = [
            self.shift_layer(seed, jnp.asarray(p * self.cfg.period / self.cfg.grid))
            for p in pixels
        ]
        jax.block_until_ready(result)
        return tuple(result)

    def shift_s(self, s, dx):
        return self._shift_s(s, dx)

    def uniform(self, wl, n):
        r = self.solve(jnp.asarray(wl), jnp.asarray([[complex(n) ** 2]]))
        return jax.block_until_ready(r)

    def grating_s(self, modes, liquid):
        c = self.cfg
        layers = (liquid, *modes, liquid)
        ds = jnp.asarray([0.0] + [c.height / c.layers] * c.layers + [0.0])
        return self.stack(layers, ds)

    def full_s(self, top, bottom, liquid, outside):
        c = self.cfg
        layers = (outside, liquid, *top, liquid, *bottom, liquid, outside)
        ds = jnp.asarray(
            [0.0, 0.0]
            + [c.height / c.layers] * c.layers
            + [c.cavity]
            + [c.height / c.layers] * c.layers
            + [0.0, 0.0]
        )
        return self.stack(layers, ds)

    def fixed_s(self, liquid, outside):
        sin = self.stack((outside, liquid), jnp.asarray([0.0, 0.0]))
        sc = self.stack((liquid, liquid, liquid), jnp.asarray([0.0, self.cfg.cavity, 0.0]))
        sout = self.stack((liquid, outside), jnp.asarray([0.0, 0.0]))
        return sin, sc, sout


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def dump(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def compare(actual, expected, modes: int):
    by_block = []
    for a, b in zip(actual, expected):
        numerator = denominator = 0.0
        maximum = maximum_relative_per_design = 0.0
        for x, y in zip(a, b):
            delta = x - y
            maximum = max(maximum, float(np.max(np.abs(delta))))
            maximum_relative_per_design = max(
                maximum_relative_per_design,
                float(np.linalg.norm(delta) / max(1.0, np.linalg.norm(y))),
            )
            numerator += float(np.vdot(delta, delta).real)
            denominator += float(np.vdot(y, y).real)
        by_block.append({
            'max_abs': maximum,
            'relative_frobenius': float(np.sqrt(numerator) / max(1.0, np.sqrt(denominator))),
            'maximum_relative_per_design': maximum_relative_per_design,
        })
    powers = {}
    for pol, index in [('TE', 0), ('TM', modes)]:
        for name, block in [('T00', 0), ('R00', 2)]:
            powers[f'{pol}_{name}'] = float(np.max(np.abs(
                np.abs(actual[block][:, index, index]) ** 2
                - np.abs(expected[block][:, index, index]) ** 2
            )))
    return {
        'max_abs': max(x['max_abs'] for x in by_block),
        'max_relative_frobenius_by_block': max(x['relative_frobenius'] for x in by_block),
        'max_relative_per_design_and_block': max(x['maximum_relative_per_design'] for x in by_block),
        'by_block': by_block,
        'power_differences': powers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--orders', type=int, default=50)
    parser.add_argument('--layers', type=int, default=100)
    parser.add_argument('--grid', type=int, default=6144)
    parser.add_argument('--wavelength', type=float, default=1.61)
    parser.add_argument('--angle', type=float, default=60.0)
    parser.add_argument('--counts', type=int, nargs='+', default=[1, 5, 21, 101, 501, 1001])
    parser.add_argument('--native-chunk', type=int, default=4)
    parser.add_argument('--native-workers', type=int, default=4)
    parser.add_argument('--reuse-chunk', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--skip-save-full', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'runs')
    args = parser.parse_args()
    if not args.tag or Path(args.tag).name != args.tag or args.tag in ('.', '..'):
        parser.error('--tag must be a single directory name')

    dest = args.output_dir.resolve() / args.tag
    dest.mkdir(parents=True, exist_ok=False)
    snapshot = dest / 'source_snapshot'
    snapshot.mkdir()
    sources = [Path(__file__), FMMAX_CORE / 'fmmax_phase_shift.py', HERE / 'SWEEP3D.m']
    package = Path(fmmax.__file__).parent
    sources += [
        package / name
        for name in ('scattering.py', '_scattering.py', 'fmm.py', 'utils.py', '_fmm_result.py')
        if (package / name).exists()
    ]
    manifest = []
    for index, source in enumerate(sources):
        target = snapshot / f'{index:02d}_{source.name}'
        shutil.copyfile(source, target)
        manifest.append({
            'source': str(source),
            'snapshot': str(target.relative_to(dest)),
            'sha256': sha(source),
        })
    dump(dest / 'source_manifest.json', manifest)

    cfg = Config(orders=args.orders, layers=args.layers, grid=args.grid)
    model = Model(cfg)
    dim = 2 * model.M
    wl = np.asarray(args.wavelength, dtype=np.float64)
    tp, bp, slant_error = model.shifts(args.angle)

    samples = {}
    for count in args.counts:
        targets = np.zeros(1) if count == 1 else np.linspace(-0.30, 0.30, count)
        pixels = np.rint(targets / cfg.period * cfg.grid).astype(np.int64)
        assert np.unique(pixels).size == count
        samples[count] = {
            'target_dx': targets,
            'pixels': pixels,
            'dx': pixels * cfg.period / cfg.grid,
        }
    np.savez(
        dest / 'inputs.npz',
        top_pixels=tp,
        bottom_pixels=bp,
        wavelength=wl,
        angle=args.angle,
        orders=model.orders,
        **{f'{key}_{n}': value for n, values in samples.items() for key, value in values.items()},
    )

    report = {
        'date': datetime.now().astimezone().isoformat(),
        'tag': args.tag,
        'config': vars(cfg),
        'wavelength_um': args.wavelength,
        'slant_angle_deg': args.angle,
        'counts': args.counts,
        'repeats': args.repeats,
        'python': sys.executable,
        'fmmax': fmmax.__version__,
        'jax': jax.__version__,
        'jaxlib': importlib.metadata.version('jaxlib'),
        'numpy': np.__version__,
        'devices': list(map(str, jax.devices())),
        'physical_cpus': psutil.cpu_count(logical=False),
        'logical_cpus': psutil.cpu_count(),
        'affinity': getattr(psutil.Process(), 'cpu_affinity', lambda: None)(),
        'thread_environment': {
            key: os.environ.get(key)
            for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'XLA_FLAGS')
        },
        'precision': 'complex128',
        'block_shape': [dim, dim],
        'output_policy': 'all four full dual-polarization native S blocks, all incoming columns, on CPU before extracting efficiency',
        'timing_policy': 'allocation, preparation, dispatch, synchronization, full CPU writes and thread pool creation included; JIT warmup and disk compression separate',
        'native_policy': 'unmodified native EVD on independent shifted material grids and native full-stack batch; fixed upper modes prepared once per trial',
        'reuse_policy': 'one seed EVD, 2*layers slice shifts, two subsystem S builds per trial; full lower S shifted per displacement; native batched final star',
        'slant_pixel_rounding_error_um': slant_error,
        'native_chunk': args.native_chunk,
        'native_workers': args.native_workers,
        'reuse_chunk': args.reuse_chunk,
        'compile_and_warmup': [],
        'runs': [],
        'checks': [],
        'serialization': [],
        'completed': False,
    }

    def event(name, **values):
        entry = {'time': datetime.now().astimezone().isoformat(), 'event': name, **values}
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        with (dest / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        dump(dest / 'result.json', report)

    def independent_bottom(pixels):
        result = tuple(
            model.solve(
                jnp.asarray(wl),
                jnp.asarray(np.stack([np.roll(model.seed_eps, int(base + delta), axis=0) for delta in pixels])),
            )
            for base in bp
        )
        return jax.block_until_ready(result)

    def geometry_job(pixels, top, liquid, outside):
        t = perf_counter()
        bottom = independent_bottom(pixels)
        eigen_seconds = perf_counter() - t
        t = perf_counter()
        value = blocks(model.full_s(top, bottom, liquid, outside))
        return value, {'bottom_eigen': eigen_seconds, 'full_stack': perf_counter() - t}

    def warm(native_sizes, reuse_sizes):
        begin = perf_counter()
        liquid = model.uniform(wl, cfg.n_liquid)
        outside = model.uniform(wl, cfg.n_external)
        seed = model.native_modes(wl, [0])[0]
        top = model.phase_modes(seed, tp)
        bottom = model.phase_modes(seed, bp)
        sin, sc, sout = model.fixed_s(liquid, outside)
        upper = model.star(model.star(sin, model.grating_s(top, liquid)), sc)
        lower = model.star(model.grating_s(bottom, liquid), sout)
        jax.block_until_ready((upper, lower))
        for size in sorted(native_sizes):
            t = perf_counter()
            value, _ = geometry_job(np.arange(size), top, liquid, outside)
            report['compile_and_warmup'].append({
                'method': 'native_batch',
                'chunk': size,
                'compile_and_first_seconds': perf_counter() - t,
            })
            del value
            event('warm_native_shape', chunk=size)
        for size in sorted(reuse_sizes):
            t = perf_counter()
            warm_dx = jnp.asarray(np.arange(size, dtype=np.float64) * cfg.period / cfg.grid)
            value = blocks(model.star(upper, model.shift_s(lower, warm_dx)))
            report['compile_and_warmup'].append({
                'method': 'S_reuse',
                'chunk': size,
                'compile_and_first_seconds': perf_counter() - t,
            })
            del value
            event('warm_reuse_shape', chunk=size)
        report['warmup_total_seconds'] = perf_counter() - begin

    def run(count: int, method: str, chunk: int, workers: int, repeat: int):
        pieces = [(start, min(start + chunk, count)) for start in range(0, count, chunk)]
        cpu_begin = process_time()
        begin = perf_counter()
        output = tuple(np.empty((count, dim, dim), dtype=np.complex128) for _ in KEYS)
        parts = {}
        t = perf_counter()
        liquid = model.uniform(wl, cfg.n_liquid)
        outside = model.uniform(wl, cfg.n_external)
        parts['uniform_ports'] = perf_counter() - t
        if method == 'native_batch':
            t = perf_counter()
            top = model.native_modes(wl, tp)
            parts['fixed_top_eigen'] = perf_counter() - t
        else:
            t = perf_counter()
            seed = model.native_modes(wl, [0])[0]
            parts['seed_eigen'] = perf_counter() - t
            t = perf_counter()
            top = model.phase_modes(seed, tp)
            bottom = model.phase_modes(seed, bp)
            parts['slice_phase'] = perf_counter() - t
            t = perf_counter()
            sin, sc, sout = model.fixed_s(liquid, outside)
            upper = model.star(model.star(sin, model.grating_s(top, liquid)), sc)
            lower = model.star(model.grating_s(bottom, liquid), sout)
            jax.block_until_ready((upper, lower))
            parts['subsystem_S_build'] = perf_counter() - t
        preparation = perf_counter() - begin
        inner_begin = perf_counter()
        aggregate_jobs = {'bottom_eigen': 0.0, 'full_stack': 0.0}
        if method == 'native_batch':
            actual_workers = min(workers, len(pieces))
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                pending = {
                    executor.submit(
                        geometry_job, samples[count]['pixels'][start:end], top, liquid, outside
                    ): (start, end)
                    for start, end in pieces
                }
                for index, future in enumerate(as_completed(pending), 1):
                    start, end = pending.pop(future)
                    values, timings = future.result()
                    for target, value in zip(output, values):
                        target[start:end] = value
                    for key, seconds in timings.items():
                        aggregate_jobs[key] += seconds
                    del values
                    if index % 10 == 0 or index == len(pieces):
                        event(
                            'native_progress',
                            count=count,
                            repeat=repeat,
                            completed_chunks=index,
                            chunks=len(pieces),
                            elapsed=perf_counter() - begin,
                            process_rss_gb=psutil.Process().memory_info().rss / 1e9,
                        )
        else:
            actual_workers = 1
            for start, end in pieces:
                values = blocks(model.star(
                    upper, model.shift_s(lower, jnp.asarray(samples[count]['dx'][start:end]))
                ))
                for target, value in zip(output, values):
                    target[start:end] = value
                del values
        inner = perf_counter() - inner_begin
        seconds = perf_counter() - begin
        cpu_seconds = process_time() - cpu_begin
        finite = all(np.isfinite(value).all() for value in output)
        entry = {
            'count': count,
            'method': method,
            'repeat': repeat,
            'chunk': chunk,
            'workers': actual_workers,
            'wall_seconds': seconds,
            'cpu_seconds': cpu_seconds,
            'preparation_seconds': preparation,
            'inner_seconds': inner,
            'parts': parts,
            'worker_job_seconds_sum': aggregate_jobs if method == 'native_batch' else None,
            'output_bytes': count * 4 * dim * dim * 16,
            'patterned_EVD_matrices': cfg.layers * (1 + count) if method == 'native_batch' else 1,
            'slice_modal_shifts': 0 if method == 'native_batch' else 2 * cfg.layers,
            'full_S_shifts': count if method == 'S_reuse' else 0,
            'nonzero_full_S_shifts': int(np.count_nonzero(samples[count]['pixels'])) if method == 'S_reuse' else 0,
            'inner_batches': len(pieces),
            'all_finite': finite,
        }
        report['runs'].append(entry)
        event('timed_run', **entry)
        if not finite:
            raise RuntimeError('Nonfinite output')
        return output

    if args.calibrate:
        warm({1, 4, 8, 16}, {16})
        count = args.counts[0]
        assert count == 16
        native_outputs = []
        for chunk, workers in [(4, 4), (8, 2), (16, 1)]:
            native_outputs.append(run(count, 'native_batch', chunk, workers, 1))
        reused = run(count, 'S_reuse', 16, 1, 1)
        for output in native_outputs:
            report['checks'].append(compare(reused, output, model.M))
    else:
        native_sizes = {min(args.native_chunk, count) for count in args.counts}
        native_sizes |= {count % args.native_chunk for count in args.counts if count % args.native_chunk}
        reuse_sizes = {min(args.reuse_chunk, count) for count in args.counts}
        reuse_sizes |= {count % args.reuse_chunk for count in args.counts if count % args.reuse_chunk}
        warm(native_sizes, reuse_sizes)
        for count in args.counts:
            last = {}
            for repeat in range(1, args.repeats + 1):
                methods = ['native_batch', 'S_reuse'] if repeat % 2 else ['S_reuse', 'native_batch']
                for method in methods:
                    chunk = min(args.native_chunk if method == 'native_batch' else args.reuse_chunk, count)
                    last[method] = run(
                        count, method, chunk,
                        args.native_workers if method == 'native_batch' else 1, repeat,
                    )
                check = compare(last['S_reuse'], last['native_batch'], model.M)
                report['checks'].append({'count': count, 'repeat': repeat, **check})
                event('pair_checked', count=count, repeat=repeat, **check)
            if not args.skip_save_full:
                for method, output in last.items():
                    path = dest / f'M{count:04d}_{method}_full_s.npz'
                    t = perf_counter()
                    payload = dict(zip(KEYS, output))
                    for pol, index in [('TE', 0), ('TM', model.M)]:
                        payload[f'{pol}_T00'] = np.abs(output[0][:, index, index]) ** 2
                        payload[f'{pol}_R00'] = np.abs(output[2][:, index, index]) ** 2
                    np.savez_compressed(
                        path, **payload, dx=samples[count]['dx'],
                        pixels=samples[count]['pixels'], orders=model.orders,
                    )
                    serialize_seconds = perf_counter() - t
                    report['serialization'].append({
                        'count': count,
                        'method': method,
                        'file': path.name,
                        'seconds': serialize_seconds,
                        'bytes': path.stat().st_size,
                        'sha256': sha(path),
                    })
                    event('saved', count=count, method=method, seconds=serialize_seconds)
            del last
    report['completed'] = True
    report['finished'] = datetime.now().astimezone().isoformat()
    report['input_sha256'] = sha(dest / 'inputs.npz')
    dump(dest / 'result.json', report)
    event('complete')


if __name__ == '__main__':
    main()
