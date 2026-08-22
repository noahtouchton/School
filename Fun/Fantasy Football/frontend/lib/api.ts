const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Player = {
  id: string;
  name: string;
  position: string;
  nfl_team: string;
  status: string;
  injury_status: string | null;
  age: number | null;
  experience: number | null;
};

export type PlayerListResponse = {
  count: number;
  players: Player[];
};

export async function fetchPlayers(params: {
  position?: string;
  team?: string;
  search?: string;
  limit?: number;
}): Promise<PlayerListResponse> {
  const query = new URLSearchParams();
  if (params.position) query.set("position", params.position);
  if (params.team) query.set("team", params.team);
  if (params.search) query.set("search", params.search);
  query.set("limit", String(params.limit ?? 200));

  const res = await fetch(`${API_URL}/players?${query.toString()}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch players: ${res.status}`);
  }
  return res.json();
}

export type Projection = {
  player_id: string;
  name: string;
  position: string;
  nfl_team: string;
  status: string;
  injury_status: string | null;
  projected_points: number;
  adp: number | null;
  boosted_due_to: string | null;
};

export type ProjectionListResponse = {
  season: number;
  week: number;
  count: number;
  players: Projection[];
};

export async function fetchProjections(params: {
  position?: string;
  search?: string;
  limit?: number;
}): Promise<ProjectionListResponse> {
  const query = new URLSearchParams();
  if (params.position) query.set("position", params.position);
  if (params.search) query.set("search", params.search);
  query.set("limit", String(params.limit ?? 300));

  const res = await fetch(`${API_URL}/projections?${query.toString()}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch projections: ${res.status}`);
  }
  return res.json();
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = await res.json();
      if (body.detail) detail = String(body.detail);
    } catch {
      // non-JSON error body; keep the status code
    }
    throw new Error(detail);
  }
  return res.json();
}

function jsonPost(body: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

// ---------------------------------------------------------------------------
// ESPN integration (primary)
// ---------------------------------------------------------------------------

export type EspnStatus = {
  connected: boolean;
  league_id?: number;
  year?: number;
  nickname?: string | null;
  has_cookies?: boolean;
};

export type RosterSlot = { position: string; count: number };

export type EspnLeague = {
  league_id: number;
  year: number;
  name: string | null;
  num_teams: number;
  current_week: number;
  roster_positions: RosterSlot[];
  my_team_id: number | null;
  my_team_name: string | null;
  teams: { team_id: number; name: string; owners: number }[];
};

export type BoardEntry = {
  name: string;
  position: string;
  nfl_team: string | null;
  season_points: number;
  per_game?: number | null;
  vorp: number;
  tier: number;
  overall_rank: number;
  position_rank: number;
  adp: number | null;
  source: string;
  injury_status: string | null;
};

export type DraftRecommendation = BoardEntry & {
  score: number;
  reasons: string[];
};

export type EspnDraftState = {
  drafted: number;
  current_pick: number;
  total_rounds: number;
  num_teams: number;
  my_team_id: number | null;
  my_team_name: string | null;
  my_draft_slot: number | null;
  my_next_pick: number | null;
  picks_until_my_turn: number;
  drafted_by_me: number;
  picks: {
    round: number;
    round_pick: number;
    name: string;
    team_name: string | null;
    is_mine: boolean;
  }[];
  my_roster: { name: string }[];
  my_roster_positions: string[];
  recommendations: DraftRecommendation[];
  best_available: BoardEntry[];
};

export type LineupEntry = {
  player_key: string;
  name: string;
  position: string;
  nfl_team: string | null;
  slot: string;
  current_slot: string | null;
  projected_points: number | null;
  injury_status: string | null;
  bye_week: string | null;
};

export type LineupResult = {
  starters: LineupEntry[];
  bench: LineupEntry[];
  changes: { player_key: string; name: string; from: string | null; to: string }[];
  projected_total: number;
  current_total: number;
  improvement: number;
};

export type EspnTeamResponse = {
  team: {
    team_id: number;
    name: string;
    wins: number;
    losses: number;
    points_for: number;
    acquisition_budget_spent: number;
  };
  league: { name: string | null; year: number; current_week: number; num_teams: number };
  projection_week: { season: number | null; week: number | null };
  lineup: LineupResult;
};

export type WaiverRec = {
  add: {
    player_key: string;
    name: string;
    position: string;
    nfl_team: string | null;
    projected_points: number;
    percent_owned: number | string | null;
    injury_status: string | null;
  };
  drop: {
    player_key: string;
    name: string;
    position: string;
    projected_points: number;
    is_undroppable: boolean;
  };
  projected_gain: number;
  suggested_faab: number;
};

export const fetchEspnStatus = () => apiFetch<EspnStatus>("/espn/status");
export const connectEspn = (payload: {
  league_id: number;
  year: number;
  espn_s2?: string;
  swid?: string;
  nickname?: string;
}) =>
  apiFetch<{
    connected: boolean;
    league_name: string | null;
    num_teams: number;
    current_week: number;
    my_team: string | null;
    identified_my_team: boolean;
    teams: { team_id: number; name: string }[];
  }>("/espn/connect", jsonPost(payload));
export const disconnectEspn = () =>
  apiFetch<{ connected: boolean }>("/espn/disconnect", { method: "POST" });
export const fetchEspnLeague = () => apiFetch<EspnLeague>("/espn/league");
export const fetchEspnDraft = () => apiFetch<EspnDraftState>("/espn/draft");
export const fetchEspnTeam = () => apiFetch<EspnTeamResponse>("/espn/team");
export const applyEspnLineup = (week?: number) =>
  apiFetch<{
    applied: boolean;
    changes: { name: string; from: string | null; to: string }[];
    detail?: string;
    improvement?: number;
    verified?: boolean;
    remaining_changes?: unknown[];
  }>("/espn/lineup", jsonPost({ week }));
export const fetchEspnWaivers = () =>
  apiFetch<{ budget_spent: number; recommendations: WaiverRec[] }>("/espn/waivers");
export const espnTransaction = (payload: {
  add_player_id?: number;
  drop_player_id?: number;
  bid?: number;
}) => apiFetch<{ ok: boolean }>("/espn/transactions", jsonPost(payload));
export const espnCheatsheetCsvUrl = () => `${API_URL}/espn/cheatsheet.csv`;

// ---------------------------------------------------------------------------
// Manual draft board (works with no league connected at all)
// ---------------------------------------------------------------------------

export type ManualBoardRequest = {
  num_teams: number;
  roster_positions?: RosterSlot[];
  drafted: string[];
  my_players: string[];
  picks_until_next: number;
  top_n?: number;
};

export type ManualBoardResponse = {
  recommendations: DraftRecommendation[];
  best_available: BoardEntry[];
  my_roster_positions: string[];
  needs: {
    starters_needed: Record<string, number>;
    flex_needed: number;
    bench_open: number;
    total_open: number;
  };
  board_size: number;
  drafted_count: number;
};

export const fetchManualBoard = (payload: ManualBoardRequest) =>
  apiFetch<ManualBoardResponse>("/draft/recommendations", jsonPost(payload));

export const searchDraftPlayers = (search: string, limit = 8) =>
  apiFetch<{ players: BoardEntry[] }>(
    `/draft/players?search=${encodeURIComponent(search)}&limit=${limit}`
  );

// ---------------------------------------------------------------------------
// Yahoo integration (secondary -- Yahoo now gates API access behind approval)
// ---------------------------------------------------------------------------

export type YahooStatus = { has_credentials: boolean; connected: boolean };

export type YahooLeague = {
  league_key: string;
  name: string;
  season: number;
  num_teams: number;
  scoring_type: string | null;
  draft_status: string | null;
  current_week: string | null;
  url: string | null;
  is_finished: boolean;
};

export type YahooTeam = {
  team_key: string;
  name: string;
  is_mine: boolean;
  draft_position: string | null;
  faab_balance: string | null;
  logo: string | null;
};

export type DraftPick = {
  pick: number;
  round: number;
  team_key: string;
  team_name: string | null;
  player: { name: string | null; position: string | null; nfl_team: string | null };
};

export type DraftState = {
  draft_status: string | null;
  draft_type: string | null;
  num_teams: number;
  total_rounds: number;
  my_team_key: string | null;
  my_draft_position: number | null;
  current_pick: number;
  my_next_pick: number | null;
  picks_until_my_turn: number;
  on_the_clock: string | null;
  i_am_on_the_clock: boolean;
  picks: DraftPick[];
  my_roster: { name: string | null; position: string | null; nfl_team: string | null }[];
  recommendations: DraftRecommendation[];
  best_available: BoardEntry[];
};

export type MyTeamResponse = {
  team: YahooTeam & { waiver_priority: string | null };
  league: {
    name: string;
    season: number;
    current_week: string | null;
    num_teams: number;
    uses_faab: boolean;
  };
  projection_week: { season: number | null; week: number | null };
  lineup: LineupResult;
};

export const fetchYahooStatus = () => apiFetch<YahooStatus>("/yahoo/status");
export const fetchYahooLoginUrl = () =>
  apiFetch<{ authorize_url: string }>("/yahoo/auth/login");
export const submitYahooCode = (code: string) =>
  apiFetch<{ connected: boolean }>("/yahoo/auth/code", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code }),
  });
export const disconnectYahoo = () =>
  apiFetch<{ connected: boolean }>("/yahoo/auth/disconnect", { method: "POST" });

export const fetchYahooLeagues = () =>
  apiFetch<{ leagues: YahooLeague[] }>("/yahoo/leagues");
export const fetchYahooLeague = (leagueKey: string) =>
  apiFetch<{ league: YahooLeague & { roster_positions: { position: string; count: number }[] }; teams: YahooTeam[]; my_team_key: string | null }>(
    `/yahoo/leagues/${leagueKey}`
  );

export const fetchDraftState = (leagueKey: string) =>
  apiFetch<DraftState>(`/yahoo/leagues/${leagueKey}/draft`);
export const fetchCheatsheet = (leagueKey: string, limit = 250) =>
  apiFetch<{ players: BoardEntry[] }>(
    `/yahoo/leagues/${leagueKey}/cheatsheet?limit=${limit}`
  );
export const cheatsheetCsvUrl = (leagueKey: string) =>
  `${API_URL}/yahoo/leagues/${leagueKey}/cheatsheet.csv`;

export const fetchMyTeam = (leagueKey: string, week?: number) =>
  apiFetch<MyTeamResponse>(
    `/yahoo/leagues/${leagueKey}/team${week ? `?week=${week}` : ""}`
  );
export const applyOptimalLineup = (leagueKey: string, week: number) =>
  apiFetch<{ applied: boolean; changes: unknown[]; detail?: string; improvement?: number }>(
    `/yahoo/leagues/${leagueKey}/lineup`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ week }),
    }
  );

export const fetchWaivers = (leagueKey: string) =>
  apiFetch<{ faab_balance: string | null; recommendations: WaiverRec[] }>(
    `/yahoo/leagues/${leagueKey}/waivers`
  );
export const executeYahooTransaction = (
  leagueKey: string,
  payload: { add_player_key?: string; drop_player_key?: string; faab_bid?: number }
) =>
  apiFetch<{ ok: boolean }>(`/yahoo/leagues/${leagueKey}/transactions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
