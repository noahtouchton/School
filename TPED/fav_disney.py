#!/usr/bin/env python3
"""
wdw_ranker.py — Walt Disney World ride ranker (CLI)

What it does
------------
• Fetches live attraction lists from Queue-Times (https://queue-times.com/en-US/pages/api).
• Includes currently operating + temporary refurb closures; excludes permanently closed,
  shows, walkthroughs/exhibits, character meets, and transportation.
• Asks pairwise “Which do you prefer?” questions (no ties) using a human-in-the-loop merge sort.
• Produces:
    - overall ranking (all 4 parks combined)
    - per-park rankings (MK, EPCOT, DHS, DAK)
    - exports: Markdown and CSV in ./output/
• Displays a 1-line description (park + guessed ride type) for each item while comparing.

Compliance: Per Queue-Times API terms, please attribute “Powered by Queue-Times.com”.
"""

from __future__ import annotations
import csv
import datetime as dt
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple
import math
from dataclasses import dataclass

# Items you remove mid-comparison get recorded here and skipped everywhere
REMOVED_BY_USER: set[str] = set()

try:
    import requests  # lightweight; pip install requests
except ImportError:
    print("This script requires 'requests'. Install with: pip install requests")
    sys.exit(1)

# ---------------------------
# Config / constants
# ---------------------------

QUEUE_TIMES_PARKS = "https://queue-times.com/parks.json"
QUEUE_TIMES_PARK_QUEUE_FMT = "https://queue-times.com/parks/{id}/queue_times.json"

# Park name keys we'll match against Queue-Times "Walt Disney World Resort" group
PARK_NAME_MAP = {
    "Magic Kingdom Park": "MK",
    "Epcot": "EPCOT",
    "EPCOT": "EPCOT",
    "Disney's Hollywood Studios": "DHS",
    "Disney’s Hollywood Studios": "DHS",
    "Disney's Animal Kingdom Theme Park": "DAK",
    "Disney’s Animal Kingdom Theme Park": "DAK",
}

# Hard blocklist of permanently closed attractions (to be safe in 2025+)
PERMA_CLOSED = {
    # MK
    "Stitch's Great Escape!", "Stitch’s Great Escape!",
    "The Pirates League", "Sorcerers of the Magic Kingdom",
    # EPCOT
    "Ellen's Energy Adventure", "Ellen’s Energy Adventure",
    "Wonders of Life", "Body Wars", "Cranium Command",
    # DHS
    "The Great Movie Ride", "Voyage of the Little Mermaid",  # show (closed long-term)
    "Star Wars: A Galactic Spectacular",  # show
    # DAK
    "Primeval Whirl", "Rivers of Light", "UP! A Great Bird Adventure",  # shows
}

# Exclusion keyword rules (case-insensitive)
# - shows
SHOW_KEYWORDS = r"(show|spectacular|parade|fireworks|festival|beauty and the beast|frozen sing-along|lion king|phantasmic|indiana jones epic stunt)"
# - character meets
MEET_KEYWORDS = r"(meet|greet|greeting|princess fairy|royal sommerhus|storybook circus greetings|town square theater)"
# - walkthroughs/exhibits/play areas
WALK_KEYWORDS = r"(trail|trek|treehouse|exhibit|gallery|play area|boneyard|affection section|imageworks|discovery island trails|conservation station|rafiki's planet watch)"
# - transportation (external transit). Internal “boat rides” stay in.
TRANSPORT_KEYWORDS = r"(monorail|skyliner|friendship boat|ferry|water transportation|parking tram)"

EXCLUDE_PAT = re.compile("|".join([
    SHOW_KEYWORDS, MEET_KEYWORDS, WALK_KEYWORDS, TRANSPORT_KEYWORDS
]), re.IGNORECASE)

# Simple ride “type” guesser by name
TYPE_RULES: List[Tuple[str, str]] = [
    (r"(coaster|mine train|slinky dog|rock 'n' roller|tron|barnstormer|expedition everest)", "Coaster"),
    (r"(shoot|blaster|buzz|midway mania|smugglers run)", "Interactive/Shooter"),
    (r"(boat|pirates|small world|na'vi river|living with the land|frozen ever after|tiana)", "Boat Ride"),
    (r"(dark ride|haunted mansion|pooh|peter pan|under the sea|mermaid|dinosaur)", "Dark Ride"),
    (r"(spinner|astro orbiter|dumbo|triceratop spin|magic carpets)", "Spinner"),
    (r"(simulator|star tours|mission: space)", "Simulator"),
    (r"(raft|kali river rapids)", "Rapids"),
    (r"(safari|kilimanjaro)", "Safari"),
    (r"(soarin')", "Soarin’ Glider"),
    (r"(test track)", "High-Speed Dark Ride"),
    (r"(tower of terror)", "Drop Tower"),
    (r"(spaceship earth)", "Omnimover"),
    (r"(mickey & minnie|runaway railway)", "Trackless Dark Ride"),
]

# Extra patterns
SINGLE_RIDER_PAT = re.compile(r"\bSingle Rider\b", re.IGNORECASE)
PARTY_ONLY_PAT = re.compile(r"(Not[- ]?So[- ]?Scary|Very Merry|After Hours|Trick[- ]?or[- ]?Treat)", re.IGNORECASE)

# Expand SHOW/WALK/TRANSPORT filters
SHOW_KEYWORDS = r"(" + "|".join([
    r"show", r"spectacular", r"parade", r"fireworks", r"festival",
    r"beauty and the beast", r"frozen sing[- ]?along", r"lion king",
    r"phantasmic", r"indiana jones", r"musical", r"short film", r"vacation fun",
    r"philharMagic", r"carousel of progress", r"country bear", r"tiki room",
    r"hall of presidents", r"turtle talk", r"walt disney presents"
]) + r")"

WALK_KEYWORDS = r"(" + "|".join([
    r"trail", r"trails", r"trek", r"treehouse", r"exhibit", r"gallery",
    r"play ?area", r"splash ?n'? ?soak", r"boneyard", r"affection section",
    r"imageworks", r"conservation station", r"rafiki['’]s planet watch",
    r"aquarium", r"advanced training lab", r"project tomorrow", r"kidcot",
    r"house of the whispering willows", r"palais du cinéma", r"launch bay",
    r"presents", r"wilderness explorers", r"a pirate'?s adventure",
    r"bruce['’]s shark world", r"seabase aquarium", r"dino[- ]?sue",
    r"cinderella castle", r"tree of life"
]) + r")"

TRANSPORT_KEYWORDS = r"(" + "|".join([
    r"monorail", r"skyliner", r"friendship boat", r"ferry", r"water transportation",
    r"parking tram", r"railroad -", r"main street vehicles"
]) + r")"

EXCLUDE_PAT = re.compile("|".join([
    SHOW_KEYWORDS, MEET_KEYWORDS, WALK_KEYWORDS, TRANSPORT_KEYWORDS
]), re.IGNORECASE)

# Optional: explicit names to block if APIs rename oddly
EXCLUDE_NAME_SET = {
    # MK
    "A Pirate's Adventure ~ Treasures of the Seven Seas",
    "Casey Jr. Splash 'N' Soak Station",
    "Cinderella Castle",
    "Country Bear Musical Jamboree",
    "Enchanted Tales with Belle",
    "Main Street Vehicles",
    "Mickey's PhilharMagic",
    "Walt Disney's Carousel of Progress",
    "Walt Disney's Enchanted Tiki Room",
    "Walt Disney World Railroad - Fantasyland",
    "Walt Disney World Railroad - Main Street, U.S.A.",
    "The Hall of Presidents",

    # EPCOT
    "Advanced Training Lab",
    "Bruce's Shark World",
    "House of the Whispering Willows",
    "Kidcot Fun Stops",
    "Palais du Cinéma",
    "Project Tomorrow: Inventing the Wonders of the Future",
    "SeaBase Aquarium",
    "Turtle Talk With Crush",

    # DHS
    "Star Wars Launch Bay",
    "Star Wars Launch Bay: BB-8 Astromech on Duty",
    "Star Wars Launch Bay: Encounter Darth Vader",
    "Vacation Fun - An Original Animated Short with Mickey & Minnie",
    "Walt Disney Presents",
    "The Little Mermaid – A Musical Adventure – New!",

    # DAK
    "Dino-Sue",
    "Wilderness Explorers",
    "Tree of Life",
    "Feathered Friends in Flight!",
}
# Known park IDs as a last-resort fallback (stable on Queue-Times)
FALLBACK_PARK_IDS = {
    "EPCOT": 5,   # https://queue-times.com/en-US/parks/5/queue_times
    "MK": 6,      # https://queue-times.com/en-US/parks/6/queue_times
    "DHS": 7,     # https://queue-times.com/en-US/parks/7/queue_times
    "DAK": 8,     # https://queue-times.com/en-US/parks/8/queue_times
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TODAY = dt.date.today().isoformat()


# ---------------------------
# Helpers
# ---------------------------

def fetch_wdw_parks() -> Dict[str, int]:
    """
    Return {short_code: park_id} for the four WDW parks.
    Robust to API group naming changes by scanning all groups.
    Falls back to known park IDs if any are missing.
    """
    headers = {"User-Agent": "wdw-ranker/1.0 (+queue-times client)"}
    r = requests.get(QUEUE_TIMES_PARKS, timeout=20, headers=headers)
    r.raise_for_status()
    groups = r.json()

    found: Dict[str, int] = {}

    # Scan every group; do not assume a specific "Walt Disney World" label.
    for g in groups:
        for p in g.get("parks", []):
            name = (p.get("name") or "").strip()
            if name in PARK_NAME_MAP:
                found[PARK_NAME_MAP[name]] = p["id"]

    # Fill any missing via stable fallback IDs
    for code, pid in FALLBACK_PARK_IDS.items():
        found.setdefault(code, pid)

    required = {"MK", "EPCOT", "DHS", "DAK"}
    missing = required - set(found.keys())
    if missing:
        raise RuntimeError(f"Still missing parks: {missing}. Got: {found}")
    return found


def fetch_park_rides(park_id: int) -> List[Dict]:
    """Fetch a park's rides list from Queue-Times (defensive parsing)."""
    url = QUEUE_TIMES_PARK_QUEUE_FMT.format(id=park_id)
    headers = {"User-Agent": "wdw-ranker/1.0 (+queue-times client)"}
    r = requests.get(url, timeout=20, headers=headers)
    r.raise_for_status()
    data = r.json()

    rides: List[Dict] = []

    # top-level rides (if present)
    for ride in data.get("rides", []) or []:
        rides.append({
            "name": (ride.get("name") or "").strip(),
            "is_open": ride.get("is_open"),
            "wait_time": ride.get("wait_time"),
        })

    # rides nested by land
    for land in data.get("lands", []) or []:
        for ride in land.get("rides", []) or []:
            rides.append({
                "name": (ride.get("name") or "").strip(),
                "is_open": ride.get("is_open"),
                "wait_time": ride.get("wait_time"),
            })

    # de-dupe by name while preserving first-seen fields
    seen = {}
    for r1 in rides:
        nm = r1["name"]
        if not nm:
            continue
        if nm not in seen:
            seen[nm] = r1
    return list(seen.values())


def should_exclude(name: str) -> Tuple[bool, str]:
    """Return (exclude?, reason)."""
    if name in PERMA_CLOSED:
        return True, "permanently_closed"
    if name in EXCLUDE_NAME_SET:
        return True, "explicit_name"
    if SINGLE_RIDER_PAT.search(name):
        return True, "single_rider"
    if PARTY_ONLY_PAT.search(name):
        return True, "party_only"
    if EXCLUDE_PAT.search(name):
        # falls under shows/walkthrough/transport after keyword expansion
        # finer reason isn't strictly necessary, but helpful:
        if re.search(SHOW_KEYWORDS, name, re.IGNORECASE):
            return True, "shows"
        if re.search(WALK_KEYWORDS, name, re.IGNORECASE):
            return True, "walkthrough_exhibit"
        if re.search(TRANSPORT_KEYWORDS, name, re.IGNORECASE):
            return True, "transportation"
        return True, "filtered"
    # Also drop obvious duplicates like 'Single Rider' variants (handled above),
    # or weird placeholders with no real queue.
    return False, ""


def guess_type(name: str) -> str:
    for pat, label in TYPE_RULES:
        if re.search(pat, name, re.IGNORECASE):
            return label
    return "Attraction"


def build_catalog() -> Tuple[Dict[str, Dict], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Build the attraction catalog using current Queue-Times data.

    Returns:
      catalog: {ride_name: {"park": "MK/EPCOT/DHS/DAK", "desc": "PARK — Type"}}
      excluded_by_rule: {rule: [ "Name (PARK)", ... ]}
      refurb_candidates: {park: [names...]}  # is_open == False but INCLUDED
    """
    parks = fetch_wdw_parks()

    catalog: Dict[str, Dict] = {}
    excluded_by_rule: Dict[str, List[str]] = defaultdict(list)
    refurb_candidates: Dict[str, List[str]] = defaultdict(list)

    for park_code, pid in parks.items():
        # Gather raw list for this park (defensive against dupes/empty names)
        raw = fetch_park_rides(pid)
        seen_names: set[str] = set()

        for entry in raw:
            name = (entry.get("name") or "").strip()
            if not name:
                continue
            if name in seen_names:
                continue
            seen_names.add(name)

            # Apply exclusion logic (single rider, shows, walkthroughs, transport, party-only, etc.)
            exclude, reason = should_exclude(name)
            if exclude:
                excluded_by_rule[reason].append(f"{name} ({park_code})")
                continue

            # Keep refurb (is_open == False) but flag it for your info
            if entry.get("is_open") is False:
                refurb_candidates[park_code].append(name)

            # Add to main catalog
            catalog[name] = {
                "park": park_code,
                "desc": f"{park_code} — {guess_type(name)}"
            }

    return catalog, excluded_by_rule, refurb_candidates


# ---------------------------
# Ranking (merge sort with human comparator)
# ---------------------------

def ask_preference(a: str, b: str, meta: Dict[str, Dict], tracker: ComparisonTracker) -> Tuple[int | None, str | None]:
    """
    Returns:
        (1, None)   -> prefer a
        (-1, None)  -> prefer b
        (None, 'a') -> remove a from lists (user chose to delete)
        (None, 'b') -> remove b from lists (user chose to delete)

    No ties allowed. Shows a progress bar.
    """
    desc_a = meta[a]["desc"]
    desc_b = meta[b]["desc"]

    while True:
        print("\n" + render_progress(tracker))
        print("Which do you prefer?  (press 3 to remove one from the list)")
        print(f" [1] {a}  ({desc_a})")
        print(f" [2] {b}  ({desc_b})")
        choice = input("Choose 1, 2, or 3 (remove): ").strip()

        if choice == "1":
            tracker.tick(1)
            return 1, None
        if choice == "2":
            tracker.tick(1)
            return -1, None
        if choice == "3":
            # Ask which to remove
            while True:
                which = input("Remove which? [1] for the first, [2] for the second: ").strip()
                if which == "1":
                    REMOVED_BY_USER.add(a)
                    tracker.tick(1)  # still counts as a handled comparison step
                    print(f"→ Removed: {a}")
                    return None, "a"
                if which == "2":
                    REMOVED_BY_USER.add(b)
                    tracker.tick(1)
                    print(f"→ Removed: {b}")
                    return None, "b"
                print("Please press 1 or 2 to choose which to remove.")
        print("Please press 1, 2, or 3.")


def merge_sort_interactive(items: List[str], meta: Dict[str, Dict], tracker: ComparisonTracker | None = None) -> List[str]:
    # Filter out anything already removed before we start
    items = [x for x in items if x not in REMOVED_BY_USER]
    if len(items) <= 1:
        return items[:]
    if tracker is None:
        tracker = ComparisonTracker(estimate_total_comparisons(len(items)))

    mid = len(items) // 2
    left = merge_sort_interactive(items[:mid], meta, tracker)
    right = merge_sort_interactive(items[mid:], meta, tracker)
    return merge(left, right, meta, tracker)


def merge(left: List[str], right: List[str], meta: Dict[str, Dict], tracker: ComparisonTracker) -> List[str]:
    out: List[str] = []
    i = j = 0

    while i < len(left) and j < len(right):
        # Skip anything that was removed after we built the halves
        if left[i] in REMOVED_BY_USER:
            i += 1
            continue
        if right[j] in REMOVED_BY_USER:
            j += 1
            continue

        decision, removed = ask_preference(left[i], right[j], meta, tracker)
        if removed == "a":
            # drop left[i]
            i += 1
            continue
        if removed == "b":
            # drop right[j]
            j += 1
            continue

        # Normal decision path
        if decision == 1:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1

    # Append the rest (filter out any removed items)
    while i < len(left):
        if left[i] not in REMOVED_BY_USER:
            out.append(left[i])
        i += 1
    while j < len(right):
        if right[j] not in REMOVED_BY_USER:
            out.append(right[j])
        j += 1
    return out


# ---------------------------
# Export
# ---------------------------

def export_markdown(overall: List[str], per_park: Dict[str, List[str]], meta: Dict[str, Dict],
                    excluded: Dict[str, List[str]], refurb: Dict[str, List[str]]) -> str:
    fname = os.path.join(OUTPUT_DIR, f"wdw_rankings_{TODAY}.md")
    with open(fname, "w", encoding="utf-8") as f:
        f.write("# Walt Disney World Ride Rankings\n")
        f.write(f"_Generated {TODAY}. Data source: Powered by Queue-Times.com_\n\n")
        f.write("## Overall Ranking\n\n")
        for idx, name in enumerate(overall, 1):
            f.write(f"{idx}. **{name}** — {meta[name]['desc']}\n")
        f.write("\n---\n")
        for park in ["MK", "EPCOT", "DHS", "DAK"]:
            f.write(f"## {park} Ranking\n\n")
            for idx, name in enumerate(per_park.get(park, []), 1):
                f.write(f"{idx}. **{name}** — {meta[name]['desc']}\n")
            f.write("\n")
        f.write("---\n")
        f.write("## Included but currently closed for refurbishment (FYI)\n")
        if any(refurb.values()):
            for park, names in refurb.items():
                if not names: continue
                f.write(f"- {park}: " + "; ".join(sorted(names)) + "\n")
        else:
            f.write("- None detected from API snapshot.\n")
        f.write("\n## Excluded (for transparency)\n")
        for rule, names in excluded.items():
            if names:
                f.write(f"- **{rule}**: " + "; ".join(sorted(names)) + "\n")
    return fname


def export_csv(overall: List[str], per_park: Dict[str, List[str]], meta: Dict[str, Dict]) -> str:
    fname = os.path.join(OUTPUT_DIR, f"wdw_rankings_{TODAY}.csv")
    with open(fname, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "attraction", "park", "type_guess", "scope"])
        # overall
        for idx, name in enumerate(overall, 1):
            w.writerow([idx, name, meta[name]["park"], meta[name]["desc"].split("—", 1)[-1].strip(), "overall"])
        # per-park
        for park, listing in per_park.items():
            for idx, name in enumerate(listing, 1):
                w.writerow([idx, name, park, meta[name]["desc"].split("—", 1)[-1].strip(), park])
    return fname
@dataclass
class ComparisonTracker:
    total_estimate: int
    done: int = 0

    def tick(self, n: int = 1):
        self.done += n

def estimate_total_comparisons(n: int) -> int:
    # Heuristic: ~ n * log2(n) comparisons for merge-sort with a human comparator
    return max(1, int(math.ceil(n * max(1.0, math.log2(max(2, n))))))

def render_progress(tracker: ComparisonTracker) -> str:
    # Simple 30-char bar
    width = 30
    pct = min(1.0, tracker.done / max(1, tracker.total_estimate))
    fill = int(round(width * pct))
    bar = "#" * fill + "-" * (width - fill)
    percent = int(round(100 * pct))
    return f"[{bar}] {tracker.done}/{tracker.total_estimate} ({percent}%)"

# ---------------------------
# Orchestration
# ---------------------------

def main():
    print("Fetching current attraction lists for the four Walt Disney World parks...")
    catalog, excluded, refurb = build_catalog()
    if not catalog:
        print("No attractions found — aborting.")
        sys.exit(2)

    # ---- Pretty-print the generated lists (grouped by park) ----
    print("\n✅ Attraction lists generated successfully!\n")
    parks_order = ["MK", "EPCOT", "DHS", "DAK"]
    park_names = {
        "MK": "Magic Kingdom",
        "EPCOT": "EPCOT",
        "DHS": "Disney's Hollywood Studios",
        "DAK": "Disney's Animal Kingdom",
    }

    grouped = defaultdict(list)
    for ride, info in catalog.items():
        grouped[info["park"]].append(ride)

    for code in parks_order:
        print(f"\n=== {park_names[code]} ({code}) ===")
        if grouped[code]:
            for name in sorted(grouped[code], key=lambda s: s.lower()):
                desc = catalog[name]["desc"].split("—", 1)[-1].strip()
                print(f" - {name}  ({desc})")
        else:
            print(" (No attractions found)")

    # Refurb (included) list
    if any(refurb.values()):
        print("\n---\n⚙️  Currently closed for refurbishment (still included):")
        for park, names in refurb.items():
            if names:
                print(f" {park}: " + ", ".join(sorted(names)))
    else:
        print("\nNo refurb attractions detected.")

    # Excluded transparency
    if any(excluded.values()):
        print("\n---\n🚫 Excluded items (for your info):")
        for rule, names in excluded.items():
            if names:
                print(f" {rule}: " + ", ".join(sorted(names)))
    else:
        print("\nNo excluded items.")

    # ---- Prep per-park arrays for ranking ----
    by_park: Dict[str, List[str]] = defaultdict(list)
    for name, info in catalog.items():
        by_park[info["park"]].append(name)
    for k in by_park:
        by_park[k].sort(key=lambda s: s.lower())

    # ---- Overall ranking (with progress bar + [3]=remove) ----
    overall_pool = sorted(catalog.keys(), key=lambda s: (catalog[s]["park"], s.lower()))
    overall_pool = [x for x in overall_pool if x not in REMOVED_BY_USER]

    print("\n=== Overall Ranking ===")
    overall_tracker = ComparisonTracker(estimate_total_comparisons(len(overall_pool)))
    overall_ranked = merge_sort_interactive(overall_pool, catalog, overall_tracker)

    # ---- Per-park rankings (respect removals, each with progress) ----
    per_park_ranked: Dict[str, List[str]] = {}
    for park in parks_order:
        print(f"\n=== {park} Ranking ===")
        park_items = [x for x in by_park[park] if x not in REMOVED_BY_USER]
        if park_items:
            tracker = ComparisonTracker(estimate_total_comparisons(len(park_items)))
            per_park_ranked[park] = merge_sort_interactive(park_items, catalog, tracker)
        else:
            per_park_ranked[park] = []

    # ---- Removed-by-user summary ----
    if REMOVED_BY_USER:
        print("\nRemoved by you during ranking:")
        for nm in sorted(REMOVED_BY_USER):
            # If it’s still in meta, show park/type; otherwise just the name
            if nm in catalog:
                print(f" - {nm} — {catalog[nm]['desc']}")
            else:
                print(f" - {nm}")

    # ---- Exports ----
    md_path = export_markdown(overall_ranked, per_park_ranked, catalog, excluded, refurb)
    csv_path = export_csv(overall_ranked, per_park_ranked, catalog)

    print("\nDone!")
    print(f"Markdown: {md_path}")
    print(f"CSV:      {csv_path}")
    print("\nPowered by Queue-Times.com")
    print("\nNote: Refurb closures were INCLUDED in your comparisons but flagged in the report.")
    print("      Permanently-closed, shows, walkthroughs, meets, party-only, transportation,")
    print("      and single-rider duplicates were excluded and listed for transparency.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")