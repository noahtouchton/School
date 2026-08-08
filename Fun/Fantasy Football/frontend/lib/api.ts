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
