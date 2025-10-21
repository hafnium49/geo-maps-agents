import { H3HexagonLayer } from '@deck.gl/geo-layers';
import { useMemo } from 'react';
import DeckGoogleMap from './common/DeckGoogleMap';

type HeatmapDatum = {
  id?: string;
  hexId?: string;
  score?: number;
  value?: number;
  localness?: number;
  ring?: string;
};

export type H3HeatmapProps = {
  center: { lat: number; lng: number };
  hexes: HeatmapDatum[];
  legend?: string;
  googleMapsApiKey?: string;
};

function resolveHexId(hex: HeatmapDatum): string {
  return hex.hexId ?? hex.id ?? '';
}

function resolveValue(hex: HeatmapDatum): number {
  if (typeof hex.score === 'number') return hex.score;
  if (typeof hex.value === 'number') return hex.value;
  if (typeof hex.localness === 'number') return hex.localness;
  return 0;
}

export default function H3Heatmap({ center, hexes, legend, googleMapsApiKey }: H3HeatmapProps) {
  const layer = useMemo(() => {
    return new H3HexagonLayer({
      id: 'h3-heatmap',
      data: hexes,
      pickable: true,
      extruded: false,
      filled: true,
      opacity: 0.85,
      getHexagon: (d: HeatmapDatum) => resolveHexId(d),
      getFillColor: (d: HeatmapDatum) => {
        const raw = Math.max(0, Math.min(1, resolveValue(d)));
        const base = Math.floor(raw * 255);
        const ring = d.ring ?? 'core';
        const alpha = ring === 'belt' ? 140 : 190;
        return [255 - base * 0.4, 90 + base * 0.2, 60 + base * 0.1, alpha];
      },
      getLineColor: (d: HeatmapDatum) => {
        const ring = d.ring ?? 'core';
        return ring === 'belt' ? [255, 255, 255, 40] : [255, 255, 255, 100];
      },
      lineWidthMinPixels: 1,
    });
  }, [hexes]);

  return (
    <div className="geo-widget">
      <DeckGoogleMap center={center} layers={[layer]} googleMapsApiKey={googleMapsApiKey} />
      {legend && <div className="geo-widget-legend">{legend}</div>}
    </div>
  );
}
