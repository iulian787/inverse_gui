#!/usr/bin/env python3
"""Capture what `conda activate <env>` sets, so the Runner does not need `conda run`.

Why this exists
---------------
The Runner spawns the optimizer via the env's interpreter directly
(<prefix>/bin/python) rather than `conda run -n cenv python`. That is deliberate:
`conda run` builds a 4-deep process tree (conda -> conda -> bash -> python) and
Popen.terminate() kills only the outermost, leaving a multi-GB torch+IPOPT process
running while cancel() reports success.

The cost of bypassing `conda run` is that activation hooks in
<prefix>/etc/conda/activate.d/ never execute. For cenv there are three, and one is
load-bearing: libblas_mkl_activate.sh sets MKL_INTERFACE_LAYER=LP64,GNU, which
selects the MKL BLAS interface used by numpy/scipy/torch. (The other two set
GSETTINGS_SCHEMA_DIR and XML_CATALOG_FILES -- GTK and libxml2, irrelevant here.)

This script runs the activation once and records the resulting environment delta to
JSON. The Runner merges that delta into the child's env dict, so the child is fully
activated while staying a direct, single-level, killable child process.

Re-run it whenever the solver env is rebuilt or updated.

    python scripts/probe_solver_env.py --env cenv --out env/cenv.activation.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

# Set by conda's own bookkeeping rather than by a package's activation hook. Injecting
# these into a child that was not launched through conda is misleading at best.
SKIP_PREFIXES = ('CONDA_', '_CE_', 'PS1', 'SHLVL', '_')
SKIP_EXACT = {'PATH', 'PWD', 'OLDPWD'}


def find_conda(explicit=None):
    for cand in (explicit, os.environ.get('CONDA_EXE'), shutil.which('conda')):
        if cand and os.path.exists(cand):
            return cand
    for prefix in ('~/miniconda3', '~/anaconda3', '~/miniforge3', '/opt/conda'):
        cand = os.path.expanduser(f'{prefix}/bin/conda')
        if os.path.exists(cand):
            return cand
    return None


def read_env(argv):
    """Run `env -0` under argv and parse into a dict."""
    out = subprocess.run(argv, capture_output=True, check=True).stdout
    result = {}
    for entry in out.split(b'\0'):
        if b'=' in entry:
            k, v = entry.split(b'=', 1)
            result[k.decode()] = v.decode()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='cenv', help='conda env name')
    ap.add_argument('--conda-exe', default=None)
    ap.add_argument('--out', default='env/cenv.activation.json')
    args = ap.parse_args()

    conda = find_conda(args.conda_exe)
    if not conda:
        sys.exit("conda not found; pass --conda-exe")

    activated = read_env([conda, 'run', '-n', args.env, '--no-capture-output', 'env', '-0'])
    prefix = activated.get('CONDA_PREFIX')
    if not prefix:
        sys.exit(f"could not resolve CONDA_PREFIX for env {args.env!r}")

    # Baseline: this shell's environment, which is what the GUI would otherwise pass on.
    baseline = dict(os.environ)

    delta = {}
    for k, v in activated.items():
        if k in SKIP_EXACT or k.startswith(SKIP_PREFIXES):
            continue
        if baseline.get(k) != v:
            delta[k] = v

    payload = {
        'env_name': args.env,
        'prefix': prefix,
        'python': os.path.join(prefix, 'bin', 'python'),
        'activation_env': delta,
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write('\n')

    print(f"env      : {args.env}")
    print(f"python   : {payload['python']}")
    print(f"delta    : {len(delta)} var(s) -> {args.out}")
    for k in sorted(delta):
        print(f"  {k}={delta[k]}")


if __name__ == '__main__':
    main()
