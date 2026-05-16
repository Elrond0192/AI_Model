"""Reusable Plotly chart components for the Basketball AI GUI."""
from __future__ import annotations
from typing import Dict, List, Optional

try:
    import plotly.graph_objects as go
    import plotly.express as px
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


def trajectory_chart(
    seasons: List[str],
    ratings: List[float],
    ci_low: Optional[List[float]] = None,
    ci_high: Optional[List[float]] = None,
    player_name: str = "Player",
) -> Optional[object]:
    """Return a Plotly figure for a player rating trajectory with optional CI band."""
    if not _PLOTLY_AVAILABLE:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=seasons, y=ratings, mode="lines+markers",
        name=player_name, line=dict(color="#1f77b4", width=2),
    ))
    if ci_low and ci_high:
        fig.add_trace(go.Scatter(
            x=seasons + seasons[::-1],
            y=ci_high + ci_low[::-1],
            fill="toself", fillcolor="rgba(31,119,180,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% CI",
        ))
    fig.update_layout(
        title=f"Rating Trajectory – {player_name}",
        xaxis_title="Season", yaxis_title="Rating",
        yaxis=dict(range=[0, 10]),
        template="plotly_white",
    )
    return fig


def style_radar_chart(
    style_vector: Dict[str, float],
    player_name: str = "Player",
    team_name: str = "Team",
) -> Optional[object]:
    """Return a Plotly radar chart showing player style vs team style."""
    if not _PLOTLY_AVAILABLE or not style_vector:
        return None
    labels = list(style_vector.keys())
    values = [style_vector[k] for k in labels]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself", name=player_name,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Style Fit: {player_name} @ {team_name}",
        template="plotly_white",
    )
    return fig


def compatibility_heatmap(
    player_names: List[str],
    team_names: List[str],
    scores: List[List[float]],
) -> Optional[object]:
    """Return a Plotly heatmap of player-team compatibility scores."""
    if not _PLOTLY_AVAILABLE:
        return None
    fig = px.imshow(
        scores, x=team_names, y=player_names,
        color_continuous_scale="RdYlGn", zmin=0, zmax=1,
        title="Player–Team Compatibility Heatmap",
        labels=dict(color="Compatibility"),
    )
    return fig
