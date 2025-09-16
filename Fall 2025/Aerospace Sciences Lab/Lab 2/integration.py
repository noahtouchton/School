#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

def _deg_or_rad(beta, degrees=True):
    return np.deg2rad(beta) if degrees else beta

def compute_ds_from_xy(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)
    # pad to same length as x by repeating last segment (assumes small spacing)
    return np.concatenate([ds, ds[-1:]])

def integrate_trapz(y_vals, x_vals):
    return np.trapz(y_vals, x_vals)

def integrate_pressure_forces(
    beta, dp, s=None, x=None, y=None, beta_in_degrees=True,
    q_inf=None, chord=None
):
    """
    Returns:
        results: dict with Dp (drag per unit span), Lp (lift per unit span),
                 and optionally CDp, CLp if q_inf and chord are provided.
    Notes:
        - Uses D' = ∮ Δp * sin(beta) ds
        - Uses L' = -∮ Δp * cos(beta) ds
        - If x,y provided, ds and cos/sin(beta) can be formed two ways:
            (A) directly from beta (preferred if given by the tunnel)
            (B) geometrically via dy/ds and dx/ds. Here we stick to (A),
                but still use x,y to compute ds if s is not provided.
    """
    beta = np.asarray(beta).astype(float)
    dp   = np.asarray(dp).astype(float)

    if s is not None:
        s = np.asarray(s).astype(float)
        if s.ndim != 1 or len(s) != len(beta):
            raise ValueError("s must be 1D and same length as beta/dp.")
        ds = np.diff(s)
        ds = np.concatenate([ds, ds[-1:]])  # pad
    elif x is not None and y is not None:
        x = np.asarray(x).astype(float)
        y = np.asarray(y).astype(float)
        if len(x) != len(beta) or len(y) != len(beta):
            raise ValueError("x,y must match the length of beta/dp.")
        ds = compute_ds_from_xy(x, y)
    else:
        # If neither s nor (x,y) are given, assume unit spacing
        ds = np.ones_like(beta)

    # Convert beta if needed
    beta_rad = _deg_or_rad(beta, degrees=beta_in_degrees)

    sinb = np.sin(beta_rad)
    cosb = np.cos(beta_rad)

    # Element-wise integrands (per surface element)
    # D' = ∮ dp * sin(beta) ds
    # L' = -∮ dp * cos(beta) ds
    dD = dp * sinb
    dL = -dp * cosb

    # If s exists, integrate over s; otherwise integrate over a running arc-length index
    if s is not None:
        x_axis = s
    else:
        # Cumulative surrogate abscissa from ds
        x_axis = np.zeros_like(ds)
        x_axis[1:] = np.cumsum(ds)[:-1]

    Dp = integrate_trapz(dD, x_axis)
    Lp = integrate_trapz(dL, x_axis)

    results = {"Dp": Dp, "Lp": Lp}

    if (q_inf is not None) and (chord is not None):
        CDp = Dp / (q_inf * chord)
        CLp = Lp / (q_inf * chord)
        results.update({"CDp": CDp, "CLp": CLp})

    return results

def main():
    ap = argparse.ArgumentParser(description="Integrate pressure drag & lift from beta and (p - p_wall).")
    ap.add_argument("csv", nargs="?", default="data.csv",
                    help="CSV file with columns. Default = data.csv in the same folder.")
    ap.add_argument("--beta-col", default="beta", help="Column name for beta (default: beta).")
    ap.add_argument("--dp-col",   default="dp",   help="Column name for Δp = p - p_wall (default: dp).")
    ap.add_argument("--s-col",    default=None,   help="Column name for arc length s (optional).")
    ap.add_argument("--x-col",    default=None,   help="Column name for x coordinate (optional).")
    ap.add_argument("--y-col",    default=None,   help="Column name for y coordinate (optional).")
    ap.add_argument("--beta-deg", action="store_true", help="Set if beta is in degrees (default).")
    ap.add_argument("--beta-rad", action="store_true", help="Set if beta is in radians.")
    ap.add_argument("--q", type=float, default=None, help="Freestream dynamic pressure q_inf (optional).")
    ap.add_argument("--chord", type=float, default=None, help="Reference chord length c (optional).")
    args = ap.parse_args()

    # Default to degrees unless explicitly told radians
    beta_in_degrees = True
    if args.beta_rad:
        beta_in_degrees = False
    if args.beta_deg:
        beta_in_degrees = True

    df = pd.read_csv(args.csv)

    if args.beta_col not in df or args.dp_col not in df:
        raise ValueError(f"CSV must contain columns '{args.beta_col}' and '{args.dp_col}'.")

    beta = df[args.beta_col].to_numpy()
    dp   = df[args.dp_col].to_numpy()

    s = df[args.s_col].to_numpy() if args.s_col and args.s_col in df else None
    x = df[args.x-col].to_numpy() if args.x_col and args.x_col in df else None
    y = df[args.y-col].to_numpy() if args.y_col and args.y_col in df else None

    results = integrate_pressure_forces(
        beta, dp, s=s, x=x, y=y, beta_in_degrees=beta_in_degrees,
        q_inf=args.q, chord=args.chord
    )

    print("=== Pressure Force Integration Results ===")
    print(f"D' (pressure drag per unit span): {results['Dp']:.6g}")
    print(f"L' (pressure lift  per unit span): {results['Lp']:.6g}")
    if "CDp" in results and "CLp" in results:
        print(f"C_D,pressure: {results['CDp']:.6g}")
        print(f"C_L,pressure: {results['CLp']:.6g}")

if __name__ == "__main__":
    main()