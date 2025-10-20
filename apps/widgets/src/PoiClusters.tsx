import { ScatterplotLayer, TextLayer } from '@deck.gl/layers';
import { useMemo } from 'react';
import DeckGoogleMap from './common/DeckGoogleMap';

export type Poi = {
  id: string;
  name: string;
  lat: number;
  lng: number;
  score?: number | null;
  etaSec?: number | null;
  googleMapsUri?: string | null;
  clusterLabel?: string | null;
};

export type PoiClustersProps = {
  center: { lat: number; lng: number };
  pois: Poi[];
  googleMapsApiKey?: string;
};

export default function PoiClusters({ center, pois, googleMapsApiKey }: PoiClustersProps) {
  const layers = useMemo(() => {
    const scatter = new ScatterplotLayer<Poi>({
      id: 'poi-scatter',
      data: pois,
      pickable: true,
      getPosition: (d) => [d.lng, d.lat],
      getRadius: (d) => 60 + (d.score ?? 0) * 120,
      radiusUnits: 'meters',
      getFillColor: (d) => {
        const score = d.score ?? 0.5;
        return [40, 120 + score * 120, 220, 200];
      },
      getLineColor: [255, 255, 255],
      lineWidthMinPixels: 1.5,
      onClick: (info) => {
        const target = info.object as Poi | null;
        if (target?.googleMapsUri) {
          window.open(target.googleMapsUri, '_blank');
        }
      },
    });

    const labels = new TextLayer<Poi>({
      id: 'poi-labels',
      data: pois.filter((poi) => poi.clusterLabel),
      getPosition: (d) => [d.lng, d.lat],
      getText: (d) => d.clusterLabel ?? '',
      getColor: [32, 32, 32, 230],
      sizeUnits: 'meters',
      sizeMaxPixels: 36,
      sizeMinPixels: 14,
      getSize: 60,
      getTextAnchor: 'middle',
      getAlignmentBaseline: 'bottom',
      background: true,
      getBackgroundColor: [255, 255, 255, 220],
      getBorderColor: [200, 200, 200, 255],
      getBorderWidth: 1,
      padding: [6, 8],
    });

    return [scatter, labels];
  }, [pois]);

  return <DeckGoogleMap center={center} layers={layers} googleMapsApiKey={googleMapsApiKey} />;
}
