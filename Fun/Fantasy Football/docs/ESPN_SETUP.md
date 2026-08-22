# Linking your real ESPN Fantasy team

ESPN is GridironIQ's primary integration. Unlike Yahoo, there's no developer app and no
approval queue — you authenticate with the cookies your own browser already has.

## 1. Find your League ID

Open your league on espn.com. The URL contains it:

```
https://fantasy.espn.com/football/team?leagueId=123456&teamId=3
```

`123456` is your League ID.

## 2. Get your cookies (private leagues only)

If your league is public you can skip this. Most leagues are private.

While logged into ESPN in Chrome or Edge:

1. Press **F12** to open DevTools
2. Go to **Application** → **Storage** → **Cookies** → `https://fantasy.espn.com`
3. Copy the **Value** of these two cookies:
   - `espn_s2` — a long URL-encoded string
   - `SWID` — a GUID in curly braces, e.g. `{1A2B3C4D-...}`. Include the braces.

In Firefox it's **Storage** → **Cookies** instead of Application.

## 3. Connect

Start the app, open **My ESPN Team**, paste the League ID, season year, and both cookies,
then hit **Connect ESPN league**.

The app confirms it found your team by name. If it says it couldn't match your SWID to a
team, the SWID came from a different ESPN account than the one that owns the team.

Credentials are stored in your local SQLite database (`data/fantasy.db`) and are only ever
sent to ESPN.

### Cookies expire

`espn_s2` is good for roughly a year, but logging out of ESPN invalidates it early. If the
app starts returning authorization errors, repeat step 2 and reconnect.

## What you get

| Feature | Where | Notes |
| --- | --- | --- |
| Live draft assistant | **Draft Room** → ESPN live | Polls ESPN every 5s, tracks every pick, ranks the best available for *your* roster and league settings. |
| Manual draft mode | **Draft Room** → Manual | Same recommendations with no league linked at all — you click each player as he's taken. Progress survives a browser refresh. |
| Cheat sheet CSV | **Draft Room** | Export the full board; paste into ESPN's custom draft rankings so autodraft follows it. |
| Optimal lineup | **My ESPN Team** | One click sets your real lineup for the week, then re-reads the roster to verify it stuck. |
| Waiver upgrades | **My ESPN Team** | Finds free agents projected to beat your weakest player at each position; executes add/drops. |

## Limitations, honestly

- **The draft pick itself is manual.** ESPN's draft room is a websocket app; this tool tells
  you exactly who to take, you click it in ESPN. Everything after the draft is automated.
- **Write operations use ESPN's private endpoints.** Lineup changes and add/drops call the
  same URLs espn.com's own UI calls. They aren't a published API and could change without
  notice. Lineup changes are verified by re-reading your roster afterward, so you'll get a
  clear error rather than a silent no-op — but check ESPN after your first write to confirm.
- **K and D/ST have no model projections.** They're valued from ADP at draft time and left
  alone by the lineup optimizer.

## Before draft night

1. Make sure data is current (from `backend/`, using the venv Python):
   - `scripts/run_ingestion.py`
   - `scripts/train_models.py`
   - `scripts/generate_projections.py 2026 1`
2. Connect your league on **My ESPN Team**.
3. Open **Draft Room**. If ESPN live mode works, use it; if anything is off, switch to
   Manual — it needs nothing but the local database.
