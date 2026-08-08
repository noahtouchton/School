from typing import Dict, Any
from .base_agent import AgentParameters

PRETRAINED_MODELS: Dict[str, Dict[str, Any]] = {
    "Optimized Super Expert": AgentParameters(
        vorp_decay_qb=0.65, vorp_decay_rb=0.45, vorp_decay_wr=0.45, vorp_decay_te=0.35,
        early_rb_limit=99, early_qb_limit=99, early_rb_minimum=0,
        rookie_boost=1.12, age_penalty_threshold=29, age_penalty_factor=0.03,
        waiver_min_improvement=1.5, waiver_max_faab_pct=0.15, faab_urgency_factor=1.25,
        trade_min_gain=0.8, young_player_trade_boost=0.15, trade_future_discount=0.88,
        matchup_adjustment=0.75, qb_wr_stack_boost=1.20
    ).__dict__,

    "Optimized Zero RB": AgentParameters(
        vorp_decay_qb=0.60, vorp_decay_rb=0.30, vorp_decay_wr=0.55, vorp_decay_te=0.40,
        early_rb_limit=0, early_qb_limit=99, early_rb_minimum=0,
        rookie_boost=1.20, age_penalty_threshold=28, age_penalty_factor=0.04,
        waiver_min_improvement=1.0, waiver_max_faab_pct=0.25, faab_urgency_factor=1.40,
        trade_min_gain=0.5, young_player_trade_boost=0.25, trade_future_discount=0.85,
        matchup_adjustment=0.85, qb_wr_stack_boost=1.25
    ).__dict__,

    "Optimized Hero RB": AgentParameters(
        vorp_decay_qb=0.55, vorp_decay_rb=0.50, vorp_decay_wr=0.50, vorp_decay_te=0.35,
        early_rb_limit=1, early_qb_limit=99, early_rb_minimum=0,
        rookie_boost=1.15, age_penalty_threshold=29, age_penalty_factor=0.03,
        waiver_min_improvement=1.5, waiver_max_faab_pct=0.18, faab_urgency_factor=1.20,
        trade_min_gain=0.8, young_player_trade_boost=0.18, trade_future_discount=0.88,
        matchup_adjustment=0.70, qb_wr_stack_boost=1.18
    ).__dict__,

    "Optimized Robust RB": AgentParameters(
        vorp_decay_qb=0.45, vorp_decay_rb=0.70, vorp_decay_wr=0.35, vorp_decay_te=0.30,
        early_rb_limit=99, early_qb_limit=99, early_rb_minimum=3,
        rookie_boost=1.05, age_penalty_threshold=30, age_penalty_factor=0.02,
        waiver_min_improvement=2.0, waiver_max_faab_pct=0.12, faab_urgency_factor=1.10,
        trade_min_gain=1.0, young_player_trade_boost=0.10, trade_future_discount=0.92,
        matchup_adjustment=0.50, qb_wr_stack_boost=1.10
    ).__dict__,

    "Optimized Late Round QB": AgentParameters(
        vorp_decay_qb=0.25, vorp_decay_rb=0.48, vorp_decay_wr=0.48, vorp_decay_te=0.38,
        early_rb_limit=99, early_qb_limit=0, early_rb_minimum=0,
        rookie_boost=1.10, age_penalty_threshold=29, age_penalty_factor=0.03,
        waiver_min_improvement=1.2, waiver_max_faab_pct=0.15, faab_urgency_factor=1.20,
        trade_min_gain=0.7, young_player_trade_boost=0.15, trade_future_discount=0.89,
        matchup_adjustment=0.80, qb_wr_stack_boost=1.15
    ).__dict__,

    "Optimized Trade Demon": AgentParameters(
        vorp_decay_qb=0.55, vorp_decay_rb=0.45, vorp_decay_wr=0.45, vorp_decay_te=0.35,
        early_rb_limit=99, early_qb_limit=99, early_rb_minimum=0,
        rookie_boost=1.15, age_penalty_threshold=28, age_penalty_factor=0.04,
        waiver_min_improvement=1.5, waiver_max_faab_pct=0.15, faab_urgency_factor=1.20,
        trade_min_gain=0.1, young_player_trade_boost=0.30, trade_future_discount=0.82,
        matchup_adjustment=0.65, qb_wr_stack_boost=1.15
    ).__dict__,

    "Optimized Waiver Wolf": AgentParameters(
        vorp_decay_qb=0.50, vorp_decay_rb=0.42, vorp_decay_wr=0.42, vorp_decay_te=0.35,
        early_rb_limit=99, early_qb_limit=99, early_rb_minimum=0,
        rookie_boost=1.18, age_penalty_threshold=29, age_penalty_factor=0.03,
        waiver_min_improvement=0.5, waiver_max_faab_pct=0.35, faab_urgency_factor=1.60,
        trade_min_gain=0.9, young_player_trade_boost=0.15, trade_future_discount=0.88,
        matchup_adjustment=0.75, qb_wr_stack_boost=1.15
    ).__dict__,

    "Optimized Matchup All Star": AgentParameters(
        vorp_decay_qb=0.60, vorp_decay_rb=0.45, vorp_decay_wr=0.45, vorp_decay_te=0.35,
        early_rb_limit=99, early_qb_limit=99, early_rb_minimum=0,
        rookie_boost=1.08, age_penalty_threshold=29, age_penalty_factor=0.03,
        waiver_min_improvement=1.0, waiver_max_faab_pct=0.15, faab_urgency_factor=1.20,
        trade_min_gain=0.8, young_player_trade_boost=0.12, trade_future_discount=0.90,
        matchup_adjustment=1.00, qb_wr_stack_boost=1.20
    ).__dict__,

    "Optimized High Risk": AgentParameters(
        vorp_decay_qb=0.70, vorp_decay_rb=0.40, vorp_decay_wr=0.40, vorp_decay_te=0.40,
        early_rb_limit=99, early_qb_limit=99, early_rb_minimum=0,
        rookie_boost=1.35, age_penalty_threshold=27, age_penalty_factor=0.05,
        waiver_min_improvement=0.8, waiver_max_faab_pct=0.22, faab_urgency_factor=1.35,
        trade_min_gain=0.4, young_player_trade_boost=0.35, trade_future_discount=0.80,
        matchup_adjustment=0.90, qb_wr_stack_boost=1.30
    ).__dict__,

    "Optimized Conservative": AgentParameters(
        vorp_decay_qb=0.40, vorp_decay_rb=0.50, vorp_decay_wr=0.50, vorp_decay_te=0.30,
        early_rb_limit=99, early_qb_limit=99, early_rb_minimum=0,
        rookie_boost=0.85, age_penalty_threshold=31, age_penalty_factor=0.01,
        waiver_min_improvement=3.5, waiver_max_faab_pct=0.05, faab_urgency_factor=1.00,
        trade_min_gain=2.0, young_player_trade_boost=0.05, trade_future_discount=0.95,
        matchup_adjustment=0.40, qb_wr_stack_boost=1.05
    ).__dict__
}
