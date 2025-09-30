#!/usr/bin/env python3
"""
Plot a satellite ground track over a world map using a TLE and time window.

Requires:
  pip install skyfield cartopy matplotlib numpy

Example:
  python groundtrack.py \
    --tle1 "1 99999U 25001A   25273.00000000  .00000000  00000-0  00000-0 0 00009" \
    --tle2 "2 99999  97.0000 285.4040 0440000 349.6006 193.5206 15.55555555    09" \
    --start "2027-09-14T16:00:00Z" --hours 12 --step_s 20 --out track.png --show
"""

import argparse
from datetime import datetime, timedelta, timezone
import numpy as np
import matplotlib.pyplot as plt

from skyfield.api import EarthSatellite, load, wgs84

# cartopy can be a little heavy at first install; it's worth it for nice maps
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def parse_args():
    p = argparse.ArgumentParser(description="Plot satellite ground track.")
    p.add_argument("--tle1", required=True, help="TLE line 1")
    p.add_argument("--tle2", required=True, help="TLE line 2")
    p.add_argument("--name", default="MySat", help="Satellite name label")
    p.add_argument("--start", required=True,
                   help="Start time UTC ISO8601 (e.g. 2027-09-14T16:00:00Z)")
    p.add_argument("--hours", type=float, default=6.0, help="Duration in hours")
    p.add_argument("--step_s", type=float, default=30.0,
                   help="Step size in seconds (e.g., 10–60)")
    p.add_argument("--elmask", type=float, default=None,
                   help="Optional elevation mask (deg) to shade station visibility")
    p.add_argument("--station", type=str, default=None,
                   help="Optional station as 'lat,lon,alt_m' (e.g., '67.857,20.226,0')")
    p.add_argument("--out", default="groundtrack.png", help="Output PNG filename")
    p.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    p.add_argument("--show", action="store_true", help="Show the plot window")
    return p.parse_args()

def build_times(ts, t0_utc, hours, step_s):
    n = int(np.ceil(hours*3600/step_s)) + 1
    # Skyfield timescale array
    times = ts.utc([ (t0_utc + timedelta(seconds=i*step_s)).replace(tzinfo=timezone.utc)
                     for i in range(n) ])
    return times

def safe_wrap_longitudes(lons_deg):
    """Wrap longitudes to [-180, 180] for nice plotting."""
    lons = ((lons_deg + 180) % 360) - 180
    return lons

def compute_ground_track(sat, ts, t0_utc, hours, step_s):
    times = build_times(ts, t0_utc, hours, step_s)
    geocentric = sat.at(times)
    subpoints = wgs84.subpoint(geocentric)
    lats = subpoints.latitude.degrees
    lons = subpoints.longitude.degrees
    # wrap to [-180, 180]
    lons = safe_wrap_longitudes(lons)
    return times, lats, lons

def plot_world(ax):
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.2)
    ax.gridlines(draw_labels=True, linestyle=":", linewidth=0.5)

def split_by_dateline(lons, lats):
    """
    Split a path so it doesn't draw long lines across the map when crossing the dateline.
    Returns list of (lons_segment, lats_segment).
    """
    segments = []
    start = 0
    for i in range(1, len(lons)):
        if abs(lons[i] - lons[i-1]) > 180:  # big jump -> split
            segments.append((lons[start:i], lats[start:i]))
            start = i
    segments.append((lons[start:], lats[start:]))
    return segments

def station_from_str(station_str):
    lat, lon, alt = station_str.split(",")
    return float(lat), float(lon), float(alt)

def visibility_mask(sat, ts, t0_utc, hours, step_s, station_llh, elmask_deg):
    """Compute elevation vs time for a single station, return intervals where elev >= mask."""
    times = build_times(ts, t0_utc, hours, step_s)
    lat, lon, alt = station_llh
    # Skyfield station
    stn = wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon, elevation_m=alt)
    diff = sat - stn
    topocentric = diff.at(times)
    alt_deg = topocentric.altaz()[0].degrees  # altitude == elevation
    visible = alt_deg >= elmask_deg
    return times, visible

def main():
    args = parse_args()
    # Time handling
    if args.start.endswith("Z"):
        start_dt = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    else:
        start_dt = datetime.fromisoformat(args.start)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)

    ts = load.timescale()
    sat = EarthSatellite(args.tle1.strip(), args.tle2.strip(), args.name, ts)

    # Compute ground track
    times, lats, lons = compute_ground_track(sat, ts, start_dt, args.hours, args.step_s)

    # Figure + map
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(11, 5.5), dpi=args.dpi)
    ax = plt.axes(projection=proj)
    plot_world(ax)

    # Plot ground track, split at dateline to avoid long wrap lines
    segments = split_by_dateline(lons, lats)
    for seg_lons, seg_lats in segments:
        ax.plot(seg_lons, seg_lats, transform=proj)

    # Mark periodic points (every N minutes) to sense motion
    every_min = max(1, int(round(args.step_s/60)))
    idx = np.arange(0, len(lats), every_min*60//int(max(args.step_s,1)))
    ax.plot(np.array(lons)[idx], np.array(lats)[idx], marker=".", linestyle="None", transform=proj)

    # Optional: station + visibility shading
    if args.station and args.elmask is not None:
        lat_s, lon_s, alt_s = station_from_str(args.station)
        ax.plot([lon_s], [lat_s], marker="^", markersize=6, transform=proj)
        # quick visibility timeline & overlay
        times_vis, vis = visibility_mask(sat, ts, start_dt, args.hours, args.step_s,
                                         (lat_s, lon_s, alt_s), args.elmask)
        # draw small pale circles along track where visible
        vis_lats = np.array(lats)[vis]
        vis_lons = np.array(lons)[vis]
        ax.plot(vis_lons, vis_lats, marker="o", linestyle="None", transform=proj)

    title = f"{args.name} ground track  |  start {start_dt.astimezone(timezone.utc).isoformat()}  |  {args.hours:.1f} h"
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
