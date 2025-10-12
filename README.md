# geo-maps-agents

This repository hosts a production-flavored OpenAI Agents SDK reference implementation for orchestrating multi-stage geospatial trip planning. The core entry point is [`geotrip_agent.py`](geotrip_agent.py), which wires together:

- A triage/orchestrator agent coordinating dedicated data, spatial, and UX agents.
- Google Places Text Search and Details (New) calls with FieldMasks for cost-aware discovery.
- Traffic-aware travel time matrices, H3 binning, HDBSCAN clustering, and heuristic TSP-TW sequencing.
- A deck.gl overlay rendered on Google Maps JS to remain Terms-of-Service compliant.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install "openai-agents>=0.4" httpx "h3<4" numpy pandas hdbscan "ortools>=9.10" cachetools pydantic "uvicorn[standard]"

export OPENAI_API_KEY=sk-...
export GOOGLE_MAPS_API_KEY=AIza...

python geotrip_agent.py
```

The script runs an end-to-end itinerary optimization example centered on Tokyo Station and writes a deck.gl-on-Google-Maps HTML overlay when Places and Routes credentials are supplied.