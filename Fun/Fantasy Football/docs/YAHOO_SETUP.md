# Linking your real Yahoo Fantasy team

> **Yahoo is no longer the recommended integration — use [ESPN_SETUP.md](ESPN_SETUP.md).**
>
> As of August 2026 Yahoo gates Fantasy Sports API access behind a manual approval process
> at <https://sports.yahoo.com/developer/access/>. Creating an app in the developer console
> is no longer enough: the console no longer offers a "Fantasy Sports" permission checkbox,
> and tokens from an unapproved app fail every fantasy call with
> `401 oauth_problem="additional_authorization_required"`. Access is **read-only by
> default**; the lineup and add/drop features here need read/write, which requires
> justifying the use case in the application.
>
> The Yahoo code in this repo is complete and works the moment an approved client id is in
> `.env` — it's kept for that reason. Everything below applies only after approval.

## 1. Create a Yahoo developer app (one time)

1. Sign in at <https://developer.yahoo.com/apps/create/> with the same Yahoo account that
   owns your fantasy team.
2. Fill in:
   - **Application Name**: anything (e.g. `GridironIQ`)
   - **Redirect URI(s)**: `https://localhost:8000/yahoo/auth/callback`
     (Yahoo requires **https** here — that's fine, see "Connect your account" below)
   - **OAuth Client Type**: **Confidential Client**
   - **API Permissions**: leave the checkboxes (OpenID Connect, TW Auction) unchecked —
     Yahoo no longer lists Fantasy Sports separately; Fantasy API access is included for
     any OAuth app.
3. Create the app and copy the **Client ID (Consumer Key)** and **Client Secret**.

## 2. Configure GridironIQ

Copy `.env.example` to `.env` (if you haven't) and fill in:

```
YAHOO_CLIENT_ID=your-client-id
YAHOO_CLIENT_SECRET=your-client-secret
YAHOO_REDIRECT_URI=https://localhost:8000/yahoo/auth/callback
```

Restart the backend after editing `.env`.

## 3. Connect your account

Open the app → **My Yahoo Team** → **Connect Yahoo**. Approve access on Yahoo's page.

After approving, your browser will land on a **"This site can't be reached"** page — that's
expected: Yahoo forces an https redirect but the local backend speaks plain http. The
authorization code is in the address bar. Copy the **entire URL**
(`https://localhost:8000/yahoo/auth/callback?code=...`) and paste it into the
"Paste the redirect URL" box on the My Yahoo Team page. The app extracts the code and
completes the login.

Tokens are stored locally in `data/yahoo_tokens.json` (gitignored) and refresh
automatically. "Disconnect Yahoo" deletes them.

## What you can do once linked

| Feature | Where | How it works |
| --- | --- | --- |
| Live draft assistant | **Draft Room** | Polls Yahoo every 5s during your draft, tracks every pick, and ranks the best available players for *your* roster and *your* league's settings (VORP from your own ML projections, blended with market ADP for rookies). |
| Cheat sheet export | **Draft Room** | CSV of the full AI board. Mirror it into Yahoo's Pre-Draft Rankings and even autodraft follows the AI. |
| Optimal lineup | **My Yahoo Team** | One click applies the model's best lineup to your real team for the week. |
| Waiver upgrades | **My Yahoo Team** | Scans free agents/waivers for players projected to beat your weakest player at each position; executes add/drops (with FAAB bids) through the API. |

## Known limitation: the draft pick itself

Yahoo's API does **not** allow third-party apps to submit live draft picks — that's a
Yahoo restriction, not ours. On draft night, keep the Yahoo draft room open next to the
GridironIQ Draft Room: this app tells you exactly who to take the moment you're on the
clock; you click the pick in Yahoo. Everything after the draft (lineups, waivers,
add/drops) is fully automated through the API.

## Before draft night checklist

1. Ingest current data and generate projections:
   - `backend/scripts/run_ingestion.py`
   - `backend/scripts/generate_projections.py` (after models are trained via
     `backend/scripts/train_models.py`)
2. Connect Yahoo and select your league on **My Yahoo Team**.
3. Open **Draft Room**, hit **Go live** when the draft starts.
