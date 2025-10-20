import { H3HexagonLayer } from '@deck.gl/layers';
import { useMemo } from 'react';
import DeckGoogleMap from './common/DeckGoogleMap';

export type RefinedHex = {
  hexId: string;
  center: { lat: number; lng: number };
  score?: number;
};

export type RefineOutlinesProps = {
  center: { lat: number; lng: number };
  hexes: RefinedHex[];
  googleMapsApiKey?: string;
};

export default function RefineOutlines({ center, hexes, googleMapsApiKey }: RefineOutlinesProps) {
  const layer = useMemo(() => {
    return new H3HexagonLayer({
      id: 'refine-outlines',
      data: hexes,
      filled: false,
      stroked: true,
      lineWidthMinPixels: 1.5,
      getHexagon: (d: RefinedHex) => d.hexId,
      getLineColor: [255, 255, 255, 220],
    });
  }, [hexes]);

  return <DeckGoogleMap center={center} layers={[layer]} googleMapsApiKey={googleMapsApiKey} />;
}
