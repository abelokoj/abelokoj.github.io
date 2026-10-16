# Kelvin–Helmholtz roll-up and vortex pairing (blog post p10)

Code for the post "Kelvin–Helmholtz Roll-Up and Vortex Pairing". Everything is plain Python (NumPy, SciPy, Matplotlib); no compiled extensions are needed.

## Files

| File | Purpose |
|---|---|
| `kh_core.py` | Chebyshev solvers for the Rayleigh and Orr–Sommerfeld equations, and the pseudo-spectral Navier–Stokes solver (`SpectralNS2D`) |
| `kh_linear.py` | Dispersion relation from theory and from the simulation, plus a resolution check (about 6 minutes) |
| `kh_rollup.py` | Nonlinear run: roll-up, saturation and pairing (about 30 minutes at 512 x 1024 on an 8-core laptop) |
| `plot_kh.py` | All figures and the animation, drawn from the saved data in `data/` |
| `pub_style.py` | Shared figure style and the `save_data`/`load_data` helpers |

## Reproducing the post

```bash
mamba create -n blog-cfd -c conda-forge python=3.12 numpy scipy numba matplotlib
mamba activate blog-cfd
python kh_linear.py
python kh_rollup.py --nx 512 --ny 1024 --tag main --frames
python kh_rollup.py --nx 256 --ny 512 --tag coarse
python plot_kh.py
```

The runs were made on WSL2 Ubuntu (AMD Ryzen 7 6800H, 8 cores) with Python 3.12, NumPy 2.5.3, SciPy 1.18.1 and Matplotlib 3.11.2. The logs of those runs are in `logs/`.

## Running on a cluster

`kh_rollup.py` uses SciPy's multithreaded FFTs (`workers=-1`), so it scales with the cores of a single node; request one node and set `--cpus-per-task` to its core count. A finer grid (for example `--nx 1024 --ny 2048 --dt 0.01`) or a higher Reynolds number (edit `RE`) is a one-line change.

## Saved data

Every run script saves its results to `data/` as it finishes (the long run also writes a `_partial` checkpoint of its time series every 1,000 steps, removed when the run completes), and `plot_kh.py` draws only from those files, so any figure can be redrawn or restyled without rerunning a simulation (`load_data("kh_rollup_main")` in `pub_style.py` returns the arrays and the run parameters as a dictionary). In addition, `plot_kh.py` writes the exact arrays behind each figure to `figures/<figure>_plotdata.npz` (for example the growth-rate curves and the measured points of Figure 1), for reuse in other plots or tools.

| File | Contents |
|---|---|
| `kh_dispersion.npz` | Rayleigh and Orr–Sommerfeld growth-rate curves, measured growth rates, mode-amplitude histories, resolution check |
| `kh_rollup_main.npz` | Time series (energy, enstrophy, modal amplitudes, momentum thickness, energy-budget residual) and vorticity and scalar snapshots of the 512 x 1024 run |
| `kh_rollup_coarse.npz` | The same for the 256 x 512 resolution check |
| `kh_frames_main.npz` | Animation frames (172 MB; kept locally but not in git, because it exceeds GitHub's file-size limit; recreate it with `kh_rollup.py --frames`) |
