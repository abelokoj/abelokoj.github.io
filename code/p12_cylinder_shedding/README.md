# Vortex shedding behind a circular cylinder at Re = 100 (blog post p12)

A D2Q9 lattice Boltzmann solver, compiled with Numba, checked against the DFG 2D-2 benchmark of Schäfer and Turek (1996) and against published results for an unconfined cylinder.

## Files

| File | Purpose |
|---|---|
| `lbm_core.py` | TRT collision, streaming, Bouzidi interpolated bounce-back with momentum-exchange forces, Zou–He inlet and outlet |
| `cyl_run.py` | Runs one case (`--case st` for the channel benchmark, `--case free` for the unconfined cylinder) at `--N` nodes per diameter |
| `cyl_analyze.py` | Cycle-by-cycle drag, lift and Strouhal number; writes `data/cyl_summary.json` |
| `plot_cyl.py` | All figures and the animation, from the saved data |
| `bench.py` | Times each kernel (lattice updates per second) |
| `pub_style.py` | Shared figure style and the `save_data`/`load_data` helpers |

## Reproducing the post

```bash
mamba create -n blog-cfd -c conda-forge python=3.12 numpy scipy numba matplotlib
mamba activate blog-cfd
export NUMBA_NUM_THREADS=8
python cyl_run.py --case st --N 20 --T 30
python cyl_run.py --case st --N 40 --T 30
python cyl_run.py --case st --N 40 --T 30 --u 0.02 --tag cyl_st_N40_u002
python cyl_run.py --case st --N 60 --T 30 --u 0.02 --tag cyl_st_N60_u002
python cyl_run.py --case free --N 20 --T 200 --u 0.02 --tag cyl_free_N20_u002
python cyl_run.py --case st --N 20 --T 30 --u 0.02 --tag cyl_st_N20_u002
python cyl_run.py --case free --N 20 --T 200
python cyl_analyze.py data/cyl_*.npz
python plot_cyl.py
```

On an 8-core laptop shared with other work the solver ran at 20 to 70 million lattice updates per second: the N = 60 channel run at u0 = 0.02 takes about 75 minutes, the unconfined runs about 40 minutes each, the others minutes.

## Running on a cluster

The kernels are parallelized with Numba threads, so the code uses the cores of one node. The finer runs that were not made on the laptop are the natural jobs to send to a cluster: `--case st --N 80 --T 30` (about 580,000 nodes and 600,000 steps) and `--case free --N 40 --T 300` (about 2.9 million nodes and 300,000 steps). Request one node, set `NUMBA_NUM_THREADS` to its core count, and expect a few hours for the first and most of a day for the second at 100 million updates per second.

## Saved data

Every run writes `data/cyl_<case>_N<N>.npz` as it finishes (and, every 20 convective units while it runs, a `_partial` checkpoint of the force histories, so a run cut short by a cluster time limit still leaves usable data): the drag, lift and pressure-difference histories, the vorticity over the last shedding cycles, and the run parameters (read them with `load_data` from `pub_style.py`). `cyl_analyze.py` adds the benchmark quantities to `data/cyl_summary.json`, and `plot_cyl.py` draws every figure from these files only, so no figure requires a rerun. `plot_cyl.py` also writes the exact arrays behind each figure to `figures/<figure>_plotdata.npz`.

`data/cyl_st_N60_u002.npz` (77 MB, mostly vorticity snapshots) is kept locally but not in git, because it is over GitHub's 50 MB warning size; its benchmark numbers are in `data/cyl_summary.json`, the arrays of every figure that uses it are in `figures/*_plotdata.npz`, and the README command above recreates it.
