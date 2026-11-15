"""MLDG native-batch FMMAX versus reusable subdomains on the same JAX backend.

Native calls are unmodified fmmax APIs. All four 402x402 native blocks are
returned and retained on the host; selected efficiency is read afterwards.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false')
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import psutil
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import fmmax
from fmmax import basis, fmm, scattering

HERE=Path(__file__).resolve().parent
INPUT=HERE/'inputs/mldg100.npz'
KEYS=('s11','s12','s21','s22')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def arrays(s):return tuple(getattr(s,k) for k in KEYS)


def host(s):return tuple(np.asarray(x) for x in s)


def compare(actual,reference):
    differences=[a-b for a,b in zip(actual,reference)]
    return {
        'max_abs':max(float(np.max(np.abs(x))) for x in differences),
        'relative_frobenius':float(np.sqrt(sum(np.vdot(x,x).real for x in differences)) /
            max(1.,np.sqrt(sum(np.vdot(x,x).real for x in reference)))),
        'by_block_max_abs':[float(np.max(np.abs(x))) for x in differences],
    }


def film_h_diagonals(d,kx,n_list,wavelength,exit_thickness):
    """Verified production SMM recurrence expressed in JAX, full channel vectors.

    Both ports are substrate n=1.51; output order is native FMMAX.
    TE production E amplitudes are converted to FMMAX H amplitudes.
    """
    k0=2*jnp.pi/wavelength
    kz=jnp.sqrt((n_list[:,None]*k0)**2-(kx[None,:]+1e-16j)**2)
    kz=jnp.where(jnp.imag(kz)<-1e-14,-kz,kz)
    kz=jnp.where((kz.real<0)&(jnp.abs(kz.imag)<1e-14),-kz,kz)
    L=d.shape[-1]
    shape=d.shape[:-1]+(kx.shape[-1],)
    z=jnp.zeros(shape,dtype=jnp.complex128);o=jnp.ones_like(z)
    def solve(pol):
        def body(index,state):
            j=L-index
            s11,s12,s21,s22=state
            ku,kl=kz[j],kz[j+1]
            if pol=='TE':
                den=ku+kl;r=(ku-kl)/den;td=2*ku/den;tu=2*kl/den
            else:
                eu,el=n_list[j]**2,n_list[j+1]**2
                den=el*ku+eu*kl
                r=(el*ku-eu*kl)/den;td=2*el*ku/den;tu=2*eu*kl/den
            dj=d[...,jnp.minimum(j,L-1)]
            phase=jnp.where(j<L,jnp.exp(1j*dj[...,None]*kl),1.+0j)
            s11,s12,s21=s11*phase**2,s12*phase,s21*phase
            inv=1/(1+r*s11)
            return (r+tu*s11*inv*td,tu*inv*s12,s21*inv*td,s22-s21*inv*r*s12)
        raw=jax.lax.fori_loop(0,L+1,body,(z,o,o,z))
        rt,tu,td,rb=raw
        if pol=='TE':rt,tu,td,rb=-rt,td,tu,-rb
        p=jnp.exp(1j*kz[-1]*exit_thickness)
        # Apply end-layer phase in FMMAX stack_s_matrix convention.
        return td,rb*p,rt,tu*p
    te,tm=solve('TE'),solve('TM')
    return tuple(jnp.concatenate([a,b],axis=-1) for a,b in zip(te,tm))


def build_functions():
    def native(d,modes,g,kx,ns):
        air,pattern,low,high,sub=modes
        layers=(air,pattern)+tuple(s for _ in range(19) for s in (low,high))+(sub,)
        # FMMAX expects thickness to broadcast directly against modal q (...,D).
        ds=(jnp.asarray(1.),jnp.asarray(1.))+tuple(d[...,j,None] for j in range(38))+(jnp.asarray(1.),)
        return arrays(scattering.stack_s_matrix(layers,ds,force_x64_solve=True))

    def native_film(d,modes,g,kx,ns):
        air,pattern,low,high,sub=modes
        layers=(sub,)+tuple(s for _ in range(19) for s in (low,high))+(sub,)
        ds=(jnp.asarray(0.),)+tuple(d[...,j,None] for j in range(38))+(jnp.asarray(1.),)
        return scattering.stack_s_matrix(layers,ds,force_x64_solve=True)

    def cached_native_film(d,modes,g,kx,ns):
        film=native_film(d,modes,g,kx,ns)
        return arrays(scattering.redheffer_star_product(g,film,force_x64_solve=True))

    def analytic_film(d,modes,g,kx,ns):
        diagonals=film_h_diagonals(d,kx,ns,jnp.asarray(1.065),jnp.asarray(1.))
        size=diagonals[0].shape[-1]
        blocks=tuple(v[...,None,:]*jnp.eye(size,dtype=jnp.complex128) for v in diagonals)
        sub=modes[-1]
        return scattering.ScatteringMatrix(**dict(zip(KEYS,blocks)),
            start_layer_solve_result=sub,start_layer_thickness=jnp.asarray(0.),
            end_layer_solve_result=sub,end_layer_thickness=jnp.asarray(1.))

    def cached_analytic_film(d,modes,g,kx,ns):
        film=analytic_film(d,modes,g,kx,ns)
        return arrays(scattering.redheffer_star_product(g,film,force_x64_solve=True))

    return {name:jax.jit(fn) for name,fn in {
        'native':native,'cached_native_film':cached_native_film,
        'cached_analytic_film':cached_analytic_film,
        'film_native':lambda *a:arrays(native_film(*a)),
        'film_analytic':lambda *a:arrays(analytic_film(*a)),
    }.items()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    parser.add_argument('--input',type=Path,default=INPUT)
    parser.add_argument('--orders',type=int,default=100)
    parser.add_argument('--groups',type=int,help='Number of designs to read (default: all supplied designs)')
    parser.add_argument('--configs',nargs='+',default=['native:4:4','cached_native_film:4:4','cached_analytic_film:4:4','cached_analytic_film:100:1'])
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--progress-seconds',type=float,default=25.)
    parser.add_argument('--skip-save-full',action='store_true')
    parser.add_argument('--device',choices=['cpu','gpu'],default='cpu')
    args=parser.parse_args()
    if not args.tag or Path(args.tag).name != args.tag or args.tag in ('.', '..'):
        parser.error('--tag must be a single directory name')
    dest=HERE/'runs'/args.tag
    input_path=args.input.resolve()
    input_hash=sha(input_path)
    with np.load(input_path) as archive:
        data = dict(archive)
    groups = args.groups if args.groups is not None else len(data['designs'])
    if not 1 <= groups <= len(data['designs']):
        parser.error('--groups must select designs from the supplied input')
    designs = data['designs'][:groups]
    dest.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(input_path, dest/'inputs.npz')
    device=jax.devices(args.device)[0]
    jax.config.update('jax_default_device',device)
    N=args.orders;M=2*N+1;D=2*M
    orders=np.array([0]+[x for k in range(1,N+1) for x in (k,-k)])
    expansion=basis.Expansion(np.column_stack([orders,np.zeros(M,int)]))
    lattice=basis.LatticeVectors(u=jnp.asarray([.9,0.],dtype=jnp.float64),v=jnp.asarray([0.,1.],dtype=jnp.float64))
    kxy=jnp.asarray([np.pi/.9,0.],dtype=jnp.float64)
    kx=kxy[0]+jnp.asarray(orders)*2*jnp.pi/.9
    ns=jnp.asarray([1.51]+[1.46,2.14]*19+[1.51],dtype=jnp.complex128)
    report={'date':datetime.now().astimezone().isoformat(),'tag':args.tag,
        'python':sys.executable,'fmmax':fmmax.__version__,'jax':jax.__version__,'jaxlib':importlib.metadata.version('jaxlib'),
        'numpy':np.__version__,'device':str(device),'N':N,'modes_per_pol':M,'block_shape':[D,D],
        'groups':len(designs),'polarization_states':2*len(designs),'precision':'complex128',
        'input_source':str(input_path),'input_sha256':input_hash,'input_paper_indices':data['paper_indices'][:len(designs)].tolist(),
        'geometry':{'wavelength_um':1.065,'period_um':.9,'grating_height_um':1.,'fill':.54,
            'grid':5000,'outer_port_thickness_um':[1.,1.],'connection_medium_n':1.51,
            'lattice_dtype':'float64','order_table':orders.tolist()},
        'output_policy':'all four native dual-polarization blocks; entire batch retained on CPU before efficiency extraction',
        'output_bytes':len(designs)*4*D*D*16,
        'thread_environment':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','XLA_FLAGS']},
        'affinity':getattr(psutil.Process(), 'cpu_affinity', lambda: None)(),'logical_cpus':psutil.cpu_count(),'physical_cpus':psutil.cpu_count(logical=False),
        'setup':{},'compile':[],'validation':{},'runs':[]}
    def emit(event,**kw):
        value={'event':event,**kw};print(json.dumps(value,ensure_ascii=False),flush=True)
        with (dest/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
        dump(dest/'result.json',report)
    snapshot=dest/'source_snapshot';snapshot.mkdir()
    sources = [Path(__file__), input_path]
    metadata = input_path.with_suffix('.json')
    if metadata.is_file():
        sources.append(metadata)
    package=Path(fmmax.__file__).parent
    sources += [package/name for name in ['scattering.py','_scattering.py','fmm.py','utils.py','_fmm_result.py'] if (package/name).exists()]
    records=[]
    for i,p in enumerate(sources):
        target=snapshot/f'{i:02d}_{p.name}';shutil.copyfile(p,target)
        records.append({'source':str(p),'snapshot':str(target.relative_to(dest)),'sha256':sha(p)})
    dump(dest/'source_manifest.json',records)
    dump(dest/'protocol.json',{**vars(args),'groups':groups,'input':str(input_path)})
    emit('start',groups=len(designs),orders=N,device=str(device))
    t=perf_counter()
    solve=jax.jit(lambda eps:fmm.eigensolve_isotropic_media(jnp.asarray(1.065),kxy,lattice,eps,expansion))
    uniform={n:solve(jnp.asarray([[n*n]],dtype=jnp.complex128)) for n in [1.,1.46,2.14,1.51]}
    x=(np.arange(5000)+.5)/5000*.9-.45
    eps=np.where(abs(x)<.9*.54/2,1.46**2,1.)[:,None]
    patterned=solve(jnp.asarray(eps,dtype=jnp.complex128))
    modes=(uniform[1.],patterned,uniform[1.46],uniform[2.14],uniform[1.51])
    jax.block_until_ready(modes)
    report['setup']['eigensolutions_seconds']=perf_counter()-t
    report['setup']['uniform_eigenvectors_identity_error']=max(float(np.max(abs(np.asarray(s.eigenvectors)-np.eye(D)))) for s in uniform.values())
    t=perf_counter()
    g=jax.jit(lambda a,p,b:scattering.stack_s_matrix((a,p,b),(jnp.asarray(1.),jnp.asarray(1.),jnp.asarray(0.)),force_x64_solve=True))(modes[0],modes[1],modes[-1])
    jax.block_until_ready(g)
    report['setup']['grating_S_compile_and_first_seconds']=perf_counter()-t
    np.savez_compressed(dest/'grating_s.npz',**{k:np.asarray(v) for k,v in zip(KEYS,arrays(g))})
    emit('modes_and_grating_ready',**report['setup'])
    fns=build_functions();common_args=(modes,g,kx,ns)
    compiled=set()
    def warm(method,chunk):
        signature=(method,chunk)
        if signature in compiled:return
        dd=np.repeat(designs[:1],chunk,axis=0)
        t=perf_counter();out=fns[method](jnp.asarray(dd),*common_args);jax.block_until_ready(out)
        elapsed=perf_counter()-t
        report['compile'].append({'method':method,'chunk':chunk,'compile_and_first_run_seconds':elapsed})
        compiled.add(signature);emit('compiled',method=method,chunk=chunk,seconds=elapsed)
    # Independent stack and film checks at two actual design points.
    dd=jnp.asarray(designs[:min(2,len(designs))])
    checks={}
    native=host(fns['native'](dd,*common_args))
    for method in ['cached_native_film','cached_analytic_film']:
        candidate=host(fns[method](dd,*common_args))
        checks[method]=compare(candidate,native)
    checks['analytic_film_vs_native']=compare(host(fns['film_analytic'](dd,*common_args)),host(fns['film_native'](dd,*common_args)))
    report['validation']['initial_two_designs']=checks
    report['validation']['all_finite']=all(np.isfinite(x).all() for x in native)
    emit('initial_validation',checks=checks)
    if max(v['relative_frobenius'] for v in checks.values())>1e-7:
        raise RuntimeError('Initial matched-model comparison needs diagnosis; diagnostics saved.')
    del native
    configs=[]
    for value in args.configs:
        method,chunk,workers=value.split(':');chunk=int(chunk);workers=int(workers)
        if method not in fns or chunk<1 or workers<1:raise ValueError(value)
        warm(method,chunk);configs.append((method,chunk,workers))
    outputs={}
    process=psutil.Process()
    for repeat in range(args.repeats):
        sequence=configs if repeat%2==0 else list(reversed(configs))
        for method,chunk,workers in sequence:
            count=len(designs);jobs=list(range(0,count,chunk))
            before=process.cpu_times();start=perf_counter()
            buffers=[np.empty((count,D,D),dtype=np.complex128) for _ in KEYS]
            timing=[]
            def work(begin):
                end=min(begin+chunk,count);dd=designs[begin:end]
                if len(dd)<chunk:dd=np.concatenate([dd,np.repeat(dd[-1:],chunk-len(dd),axis=0)],axis=0)
                t=perf_counter();result=fns[method](jnp.asarray(dd),*common_args)
                jax.block_until_ready(result);compute=perf_counter()-t
                t=perf_counter()
                for destbuf,block in zip(buffers,result):np.copyto(destbuf[begin:end],np.asarray(block)[:end-begin])
                copy=perf_counter()-t
                return {'start':begin,'end':end,'synchronized_call_seconds':compute,'host_copy_seconds':copy}
            completed_groups=0;last_progress=perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for finished in pool.map(work,jobs):
                    timing.append(finished)
                    completed_groups+=finished['end']-finished['start']
                    if perf_counter()-last_progress>=args.progress_seconds:
                        emit('trial_progress',method=method,repeat=repeat+1,
                            groups_completed=completed_groups,total_groups=count,
                            elapsed_seconds=perf_counter()-start)
                        last_progress=perf_counter()
            wall=perf_counter()-start;after=process.cpu_times()
            eta={pol:np.abs(buffers[2][:,i*M+2,i*M])**2 for i,pol in enumerate(['TE','TM'])}
            cross=max(float(np.max(np.abs(b[:,:M,M:]))) for b in buffers)
            entry={'repeat':repeat+1,'method':method,'chunk':chunk,'workers':workers,'wall_seconds':wall,
                'cpu_seconds':after.user+after.system-before.user-before.system,
                'groups':count,'all_outputs_finite':all(np.isfinite(b).all() for b in buffers),
                'cross_polarization_max':cross,'jobs':timing,'available_memory_at_endpoint':psutil.virtual_memory().available}
            if N==100:
                entry['eta_max_difference_vs_saved_reference']={p:float(np.max(np.abs(v-data[p+'_reference'][:count]))) for p,v in eta.items()}
            report['runs'].append(entry)
            emit('timed_run',**{k:v for k,v in entry.items() if k!='jobs'})
            # Keep the most recent output of each physical path for full-batch checks.
            if repeat==args.repeats-1:outputs[method]=buffers
            else:del buffers
    if 'native' in outputs:
        for method,buffer in outputs.items():
            report['validation']['full_batch_'+method]=compare(buffer,outputs['native'])
    for method,buffers in outputs.items():
        if not args.skip_save_full:
            t=perf_counter()
            np.savez_compressed(dest/(method+'_full_s.npz'),**dict(zip(KEYS,buffers)),designs=designs,
                orders=orders,TE_eta=np.abs(buffers[2][:,2,0])**2,TM_eta=np.abs(buffers[2][:,M+2,M])**2)
            report.setdefault('serialization',{})[method]=perf_counter()-t
    report['finished']=datetime.now().astimezone().isoformat()
    report['input_unchanged']=sha(input_path)==input_hash
    report['completed']=True
    dump(dest/'result.json',report)
    emit('complete',full_batch_checks={k:v for k,v in report['validation'].items() if k.startswith('full_batch')})


if __name__=='__main__':main()
