import { PolygonLayer } from '@deck.gl/layers';
import { useMemo } from 'react';
import DeckGoogleMap from './common/DeckGoogleMap';

export type IsochroneRingProps = {
  minutes: number;
  polygon: [number, number][];
};

export type IsochroneRingsWidgetProps = {
  center: { lat: number; lng: number };
  rings: IsochroneRingProps[];
  googleMapsApiKey?: string;
};

export default function IsochroneRings({ center, rings, googleMapsApiKey }: IsochroneRingsWidgetProps) {
  const layer = useMemo(() => {
    return new PolygonLayer<IsochroneRingProps>({
      id: 'isochrone-rings',
      data: rings,
      pickable: false,
      getPolygon: (d) => d.polygon,
      getFillColor: (d) => {
        const opacity = Math.max(0.15, 0.5 - d.minutes * 0.01);
        return [70, 180, 255, Math.floor(opacity * 255)];
      },
      getLineColor: [0, 120, 255, 180],
      lineWidthMinPixels: 1,
    });
  }, [rings]);

  return <DeckGoogleMap center={center} layers={[layer]} googleMapsApiKey={googleMapsApiKey} />;
}
