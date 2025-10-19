import { H3HexagonLayer } from '@deck.gl/layers';
import { useMemo } from 'react';
import DeckGoogleMap from './common/DeckGoogleMap';

export type HexBin = {
  id: string;
  value: number;
};

export type H3HeatmapProps = {
  center: { lat: number; lng: number };
  hexes: HexBin[];
  legend?: string;
  googleMapsApiKey?: string;
};

export default function H3Heatmap({ center, hexes, legend, googleMapsApiKey }: H3HeatmapProps) {
  const layer = useMemo(() => {
    return new H3HexagonLayer({
      id: 'h3-heatmap',
      data: hexes,
      pickable: true,
      extruded: false,
      filled: true,
      getHexagon: (d: HexBin) => d.id,
      getFillColor: (d: HexBin) => {
        const normalized = Math.max(0, Math.min(1, d.value));
        const intensity = Math.floor(normalized * 255);
        return [255, 120 - intensity * 0.3, 40, 180];
      },
      getLineColor: [255, 255, 255, 80],
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
