#!/usr/bin/env python3
"""
Compute satellite passes over one or more ground stations from a TLE.

Features
- Reads TLE (two lines) from CLI.
- Stations supplied either as repeated --station "Name,lat,lon,alt_m"
  OR from a JSON file with a list of objects:
    [{"name":"Svalbard","lat":78.2288,"lon":15.3996,"alt_m":0}, ...]
- Filters by minimum elevation (--el-min, degrees) and minimum duration (--min-dur-s, seconds).
- Outputs one CSV per station and a combined passes_all.csv.
- Prints a neat console summary.

Requires:
  pip install skyfield numpy pandas

Optional (but nice): set your console to UTF-8 for clean output on Windows.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from skyfield.api import EarthSatellite, load, wgs84


@dataclass
class Station:
    name: str
    lat: float
    lon: float
    alt_m: float


def parse_station_arg(s: str) -> Station:
    # "Name,lat,lon,alt_m"
    parts = [p.strip() for p in s.split(",")]
    if len(parts) < 4:
        raise ValueError(f"--station needs 'Name,lat,lon,alt_m' but got: {s}")
    name = parts[0]
    lat, lon, alt = float(parts[1]), float(parts[2]), float(parts[3])
    return Station(name=name, lat=lat, lon=lon, alt_m=alt)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Plan satellite passes for given ground stations."
    )
    ap.add_argument("--tle1", required=True, help="TLE line 1")
    ap.add_argument("--tle2", required=True, help="TLE line 2")
    ap.add_argument("--name", default="MySat", help="Satellite name label")
    ap.add_argument("--start", required=True,
                    help="Start time UTC ISO8601 (e.g. 2027-09-14T16:00:00Z)")
    ap.add_argument("--hours", type=float, default=24.0, help="Lookahead window, hours")
    ap.add_argument("--el-min", type=float, default=10.0,
                    help="Minimum elevation (deg) to consider a pass")
    ap.add_argument("--min-dur-s", type=int, default=120,
                    help="Minimum duration (seconds) for a pass")
    ap.add_argument("--stations-json", type=str, default=None,
                    help="Path to stations JSON list")
    ap.add_argument("--station", action="append", default=[],
                    help="Station as 'Name,lat,lon,alt_m' (repeatable)")
    ap.add_argument("--out-prefix", default="passes",
                    help="Output filename prefix (default: passes)")
    return ap.parse_args()


def parse_start(s: str) -> datetime:
    # Accept ...Z or with offset; ensure tz-aware UTC
    if s.endswith("Z"):
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    else:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_stations(args) -> List[Station]:
    stations: List[Station] = []
    if args.stations_json:
        with open(args.stations_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            stations.append(
                Station(
                    name=item["name"],
                    lat=float(item["lat"]),
                    lon=float(item["lon"]),
                    alt_m=float(item.get("alt_m", 0.0)),
                )
            )
    for s in args.station:
        stations.append(parse_station_arg(s))
    if not stations:
        raise SystemExit("No stations provided. Use --station or --stations-json.")
    # Deduplicate by (name, lat, lon, alt)
    uniq = []
    seen = set()
    for st in stations:
        key = (st.name, round(st.lat, 6), round(st.lon, 6), round(st.alt_m, 1))
        if key not in seen:
            seen.add(key)
            uniq.append(st)
    return uniq


def group_passes(times, events) -> List[Tuple[int, int, int]]:
    """
    Group Skyfield find_events() outputs into (rise_idx, culm_idx, set_idx).
    events codes: 0=rise, 1=culminate, 2=set
    Robust to missing leading/trailing rise/set in the window.
    """
    triplets = []
    last_rise = None
    last_culm = None
    for i, ev in enumerate(events):
        if ev == 0:  # rise
            last_rise = i
            last_culm = None
        elif ev == 1:  # culminate
            last_culm = i
        elif ev == 2:  # set
            if last_rise is not None and last_culm is not None:
                triplets.append((last_rise, last_culm, i))
            # reset
            last_rise, last_culm = None, None
    return triplets


def compute_passes_for_station(
    sat: EarthSatellite,
    ts,
    st: Station,
    t0: datetime,
    hours: float,
    el_min_deg: float,
    min_dur_s: int
) -> pd.DataFrame:
    """Return a dataframe of passes for a single station with filters applied."""
    t1 = t0 + timedelta(hours=hours)
    t0_sf = ts.from_datetime(t0)
    t1_sf = ts.from_datetime(t1)

    # Build Skyfield station
    gs = wgs84.latlon(latitude_degrees=st.lat,
                      longitude_degrees=st.lon,
                      elevation_m=st.alt_m)

    # Find AOS/CULM/LOS with desired elevation mask
    t_events, events = sat.find_events(gs, t0_sf, t1_sf, altitude_degrees=el_min_deg)

    # Group into passes
    triplets = group_passes(t_events, events)
    rows = []

    # For per-sample max elevation check, sample each pass more finely
    for (i_rise, i_culm, i_set) in triplets:
        trise = t_events[i_rise].utc_datetime().replace(tzinfo=timezone.utc)
        tculm = t_events[i_culm].utc_datetime().replace(tzinfo=timezone.utc)
        tset  = t_events[i_set].utc_datetime().replace(tzinfo=timezone.utc)
        duration_s = int((tset - trise).total_seconds())

        # Query the culminated elevation directly
        top = (sat - gs).at(t_events[i_culm])
        max_el_deg = float(top.altaz()[0].degrees)

        # Filter by duration
        if duration_s < min_dur_s:
            continue

        # Build a few extra stats (optional): min range, range at max elev
        rng_km = float(top.distance().km)

        rows.append({
            "station": st.name,
            "AOS_utc": trise.isoformat().replace("+00:00", "Z"),
            "LOS_utc": tset.isoformat().replace("+00:00", "Z"),
            "duration_s": duration_s,
            "max_elev_deg": round(max_el_deg, 1),
            "range_at_max_km": round(rng_km, 1),
            "station_lat": st.lat,
            "station_lon": st.lon,
            "station_alt_m": st.alt_m,
        })

    df = pd.DataFrame(rows).sort_values(["AOS_utc"]).reset_index(drop=True)
    return df


def main():
    args = parse_args()
    t0 = parse_start(args.start)

    ts = load.timescale()
    sat = EarthSatellite(args.tle1.strip(), args.tle2.strip(), args.name, ts)

    stations = load_stations(args)
    all_passes = []

    print(f"\nSatellite: {args.name}")
    print(f"Window: {t0.isoformat()}  for {args.hours} h")
    print(f"Filters: el_min={args.el_min:.1f}°, min_dur={args.min_dur_s}s\n")

    for st in stations:
        df = compute_passes_for_station(
            sat, ts, st, t0, args.hours, args.el_min, args.min_dur_s
        )
        all_passes.append(df)

        out_file = f"{args.out_prefix}_{st.name.replace(' ','_')}.csv"
        if not df.empty:
            df.to_csv(out_file, index=False)
            print(f"[{st.name}] {len(df)} passes  →  {out_file}")
            # Show a concise preview
            preview = df[["AOS_utc", "duration_s", "max_elev_deg"]].head(5)
            print(preview.to_string(index=False))
        else:
            print(f"[{st.name}] No passes meeting criteria.")

        print("-" * 60)

    if all_passes:
        df_all = pd.concat(all_passes, ignore_index=True) if len(all_passes) > 1 else all_passes[0]
        if not df_all.empty:
            df_all = df_all.sort_values(["AOS_utc", "station"])
            all_file = f"{args.out_prefix}_all.csv"
            df_all.to_csv(all_file, index=False)
            print(f"Combined CSV → {all_file}\n")
            # Quick high-level stats
            by_station = df_all.groupby("station")["duration_s"].agg(["count","mean","max"]).round(1)
            print("Summary (per station):")
            print(by_station.to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
