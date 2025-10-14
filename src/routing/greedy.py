"""
Greedy sequencing fallback for route optimization.

This module provides a simple greedy algorithm that sorts stops by score
and fits them into the available time window. Used as a fallback when:
- OR-Tools VRPTW solver fails or times out
- User specifies --fast flag for quick results
- Distance matrix is unavailable

The greedy approach is fast (~1ms) but suboptimal, as it doesn't consider
actual travel times between consecutive stops.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd


def _to_datetime(value):
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def _format_window(open_time: Optional[datetime], close_time: Optional[datetime]) -> Optional[str]:
    if not open_time or not close_time:
        return None
    return f"{open_time.strftime('%H:%M')}–{close_time.strftime('%H:%M')}"


@dataclass
class Stop:
    """A single stop in the itinerary."""

    place_id: str
    place_name: str
    lat: float
    lng: float
    score: float
    eta_from_anchor_sec: int
    arrival_time: datetime
    departure_time: datetime
    service_time_min: int
    reason: Optional[str] = None
    open_now: Optional[bool] = None
    open_time: Optional[datetime] = None
    close_time: Optional[datetime] = None
    maps_url: Optional[str] = None


@dataclass
class GreedySequenceResult:
    """Result from greedy sequencing."""
    
    stops: List[Stop]
    total_travel_time_sec: int
    total_service_time_sec: int
    total_duration_sec: int
    num_stops_skipped: int
    sequence_method: str = "greedy"


def greedy_sequence(
    candidates: pd.DataFrame,
    anchor_lat: float,
    anchor_lng: float,
    start_time: datetime,
    end_time: datetime,
    service_time_min: int = 35,
    min_travel_time_sec: int = 180
) -> GreedySequenceResult:
    """
    Greedy sequencing: sort by score and fit into time window.
    
    This is a fast, simple heuristic that:
    1. Sorts candidates by score (descending)
    2. Iterates through sorted list
    3. Adds each stop if it fits in remaining time
    4. Uses ETA from anchor as travel time estimate
    
    Args:
        candidates: DataFrame with columns: id, name, lat, lng, score, eta (from anchor)
        anchor_lat: Starting latitude
        anchor_lng: Starting longitude
        start_time: Window start time
        end_time: Window end time
        service_time_min: Minutes to spend at each stop
        min_travel_time_sec: Minimum travel time between stops (default: 3 min)
        
    Returns:
        GreedySequenceResult with selected stops and timing
    """
    # Sort by score (descending), then by eta (ascending) as tiebreaker
    sorted_df = candidates.sort_values(['score', 'eta'], ascending=[False, True])
    
    stops: List[Stop] = []
    current_time = start_time
    total_travel_time = 0
    num_skipped = 0
    
    for _, row in sorted_df.iterrows():
        # Estimate travel time from anchor (simplified - doesn't account for actual route)
        travel_time_sec = max(min_travel_time_sec, int(row['eta']))
        arrival = current_time + timedelta(seconds=travel_time_sec)

        # Service time at this stop
        service_time_sec = service_time_min * 60
        departure = arrival + timedelta(seconds=service_time_sec)

        open_time_raw = row.get('open_time', None)
        close_time_raw = row.get('close_time', None)
        open_time = _to_datetime(open_time_raw) if pd.notna(open_time_raw) else None
        close_time = _to_datetime(close_time_raw) if pd.notna(close_time_raw) else None

        open_now_value = row.get('open_now', None)
        if isinstance(open_now_value, (int, float)):
            open_now_value = bool(open_now_value)
        elif pd.isna(open_now_value):
            open_now_value = None

        window_str = _format_window(open_time, close_time)
        reason_parts = [f"Score={row['score']:.2f}"]
        if window_str:
            reason_parts.append(f"window={window_str}")
        reason_parts.append("greedy selection")

        # Check if we have time
        if departure > end_time:
            num_skipped += 1
            continue

        # Add stop
        stops.append(Stop(
            place_id=str(row['id']),
            place_name=str(row['name']),
            lat=float(row['lat']),
            lng=float(row['lng']),
            score=float(row['score']),
            eta_from_anchor_sec=int(row['eta']),
            arrival_time=arrival,
            departure_time=departure,
            service_time_min=service_time_min,
            reason="; ".join(reason_parts),
            open_now=open_now_value,
            open_time=open_time,
            close_time=close_time,
            maps_url=row.get('maps_url')
        ))
        
        total_travel_time += travel_time_sec
        current_time = departure
    
    total_service_time = len(stops) * service_time_min * 60
    total_duration = int((current_time - start_time).total_seconds())
    
    return GreedySequenceResult(
        stops=stops,
        total_travel_time_sec=total_travel_time,
        total_service_time_sec=total_service_time,
        total_duration_sec=total_duration,
        num_stops_skipped=num_skipped,
        sequence_method="greedy"
    )


def format_reason(stop: Stop, additional_info: dict = None) -> str:
    """
    Format a human-readable reason for including this stop.
    
    Args:
        stop: The stop to format
        additional_info: Optional dict with extra info (e.g., diversity_gain, open_now)
        
    Returns:
        Formatted reason string
    """
    parts = [f"Score={stop.score:.2f}"]
    
    if additional_info:
        if 'diversity_gain' in additional_info:
            parts.append(f"diversity={additional_info['diversity_gain']:.2f}")
        if 'open_now' in additional_info:
            parts.append(f"open_now={additional_info['open_now']}")
        if 'cluster_label' in additional_info:
            parts.append(f"cluster={additional_info['cluster_label']}")
    
    parts.append("greedy")
    
    return "; ".join(parts)
