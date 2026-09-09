#!/bin/bash
# One-time setup on the santis login node (as run on 2026-09-09).
set -e
ROOT=$SCRATCH/tmp/gt4py_ad_2026_09_09
mkdir -p $ROOT && cd $ROOT
[ -d gt4py ] || git clone --branch ad_halo https://github.com/havogt/gt4py.git gt4py
uenv run icon/26.7:v1 --view=default -- bash -c "
  uv venv --python \$(which python3) $ROOT/venv
  uv pip install --python $ROOT/venv/bin/python 'jax[cuda13]==0.11.1' numpy
  uv pip install --python $ROOT/venv/bin/python -e $ROOT/gt4py
"
