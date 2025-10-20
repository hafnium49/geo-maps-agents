import { ScatterplotLayer } from '@deck.gl/layers';
import { useMemo } from 'react';
import DeckGoogleMap from './common/DeckGoogleMap';

type RefineDot = {
  hexId: string;
  center: { lat: number; lng: number };
  score?: number;
};

export type RefineDotsProps = {
  center: { lat: number; lng: number };
  hexes: RefineDot[];
  googleMapsApiKey?: string;
};

export default function RefineDots({ center, hexes, googleMapsApiKey }: RefineDotsProps) {
  const layer = useMemo(() => {
    return new ScatterplotLayer({
      id: 'refine-dots',
      data: hexes,
      getPosition: (d: RefineDot) => [d.center.lng, d.center.lat],
      getRadius: (d: RefineDot) => {
        const score = typeof d.score === 'number' ? d.score : 0.5;
        return 40 + score * 80;
      },
      radiusUnits: 'meters',
      pickable: false,
      stroked: false,
      filled: true,
      opacity: 0.9,
      getFillColor: (d: RefineDot) => {
        const score = typeof d.score === 'number' ? d.score : 0.6;
        const intensity = Math.floor(score * 255);
        return [255, 200 - intensity * 0.4, 100 + intensity * 0.2, 220];
      },
    });
  }, [hexes]);

  return <DeckGoogleMap center={center} layers={[layer]} googleMapsApiKey={googleMapsApiKey} />;
}
