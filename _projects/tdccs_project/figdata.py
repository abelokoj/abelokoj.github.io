"""
figdata.py
==========
Persist the numbers behind each figure so that plotting can be redone
without recomputing.

Why this exists
---------------
Several runs in this project take minutes to hours, because the stable
time step scales like dx^3. Re-running a solver only to move a legend or
change a colormap is wasteful. These two functions write every array a
figure needs into a single compressed .npz file, and read it back.

Typical use, inside a figure function:

    from figdata import save_data, load_data

    results = ...            # the expensive part
    save_data("figure7", x=x, u_tdcncs=u1, u_tdccs=u2, times=times, N=80)

and later, in a separate plotting script:

    d = load_data("figure7")
    x = d["x"];  u1 = d["u_tdcncs"];  N = int(d["N"])

Anything that is not an array (an int, a float, a string) is stored as a
zero-dimensional array, so read it back with int(...), float(...) or
str(...) as appropriate.
"""
import json
import os

import numpy as np

#: Where the .npz files go, resolved relative to THIS file rather than
#: the working directory, matching how pub_style.FIGDIR behaves.
DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def save_data(basename, outdir=None, meta=None, **arrays):
    """Write every keyword argument into <DATADIR>/<basename>.npz.

    Parameters
    ----------
    basename : str
        Filename stem, without extension. Use the figure name, so the
        data file sits alongside the figure it produced.
    outdir : str, optional
        Destination directory. Defaults to DATADIR.
    meta : dict, optional
        Small scalars worth recording, for example {"N": 80, "eps": 5e-4,
        "scheme": "TDCCS"}. Stored as a JSON string under the key
        "__meta__" so it survives the round trip unchanged.
    **arrays
        The arrays to store. Dictionaries keyed by time are flattened,
        since .npz keys must be strings: pass snapshots as, for example,
        u_TDCNCS={0.0: arr, 0.5: arr} and they are stored as
        "u_TDCNCS@0.0" and "u_TDCNCS@0.5".

    Returns
    -------
    str : the path written.
    """
    outdir = DATADIR if outdir is None else outdir
    os.makedirs(outdir, exist_ok=True)

    flat = {}
    for key, val in arrays.items():
        if isinstance(val, dict):
            # A {time: array} snapshot dictionary. Record the keys too,
            # so load_data can rebuild the dictionary in order.
            # Normalize every stamp to float BEFORE building the key.
            # Otherwise a time given as the integer 0 is written as
            # "u@0" but looked up on load as "u@0.0", and the read fails.
            stamps = sorted(float(t) for t in val)
            flat[f"{key}__keys"] = np.asarray(stamps, dtype=float)
            for t in stamps:
                # val may be keyed by int or float; look up either form.
                arr = val[t] if t in val else val[int(t)]
                flat[f"{key}@{t}"] = np.asarray(arr)
        else:
            flat[key] = np.asarray(val)

    flat["__meta__"] = np.asarray(json.dumps(meta or {}))

    path = os.path.join(outdir, f"{basename}.npz")
    np.savez_compressed(path, **flat)
    print(f"    data saved -> {path}", flush=True)
    return path


def load_data(basename, outdir=None):
    """Read back what save_data wrote.

    Returns a plain dict. Snapshot dictionaries are reassembled, so a
    field saved as u_TDCNCS={0.0: ..., 0.5: ...} comes back in that same
    form, keyed by float. Metadata is returned under "meta".
    """
    outdir = DATADIR if outdir is None else outdir
    path = os.path.join(outdir, f"{basename}.npz")
    with np.load(path, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}

    out = {"meta": json.loads(str(raw.pop("__meta__")))}

    # Rebuild the {time: array} dictionaries first.
    for key in [k for k in raw if k.endswith("__keys")]:
        field = key[: -len("__keys")]
        stamps = raw.pop(key)
        out[field] = {float(t): raw.pop(f"{field}@{t}") for t in stamps}

    out.update(raw)
    return out


def has_data(basename, outdir=None):
    """True if a saved file already exists, for skip-if-present logic."""
    outdir = DATADIR if outdir is None else outdir
    return os.path.exists(os.path.join(outdir, f"{basename}.npz"))
