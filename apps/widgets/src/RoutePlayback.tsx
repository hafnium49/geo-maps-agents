import { PathLayer, ScatterplotLayer } from '@deck.gl/layers';
import { useEffect, useMemo, useState } from 'react';
import DeckGoogleMap from './common/DeckGoogleMap';

export type RoutePoint = {
  lat: number;
  lng: number;
  t?: number | null;
};

export type RouteStop = {
  lat: number;
  lng: number;
  name: string;
  place_id?: string | null;
};

export type RoutePlaybackProps = {
  center: { lat: number; lng: number };
  path: RoutePoint[];
  stops: RouteStop[];
  googleMapsApiKey?: string;
};

export default function RoutePlayback({ center, path, stops, googleMapsApiKey }: RoutePlaybackProps) {
  const [cursor, setCursor] = useState(0);

  useEffect(() => {
    if (path.length === 0) return;
    const maxT = Math.max(...path.map((p) => p.t ?? 0));
    if (maxT <= 0) return;

    let frame = 0;
    const interval = setInterval(() => {
      frame = (frame + 1) % (maxT + 60);
      setCursor(frame);
    }, 500);
    return () => clearInterval(interval);
  }, [path]);

  const layers = useMemo(() => {
    const pathLayer = new PathLayer<RoutePoint>({
      id: 'route-path',
      data: [path],
      getPath: (d) => d.map((p) => [p.lng, p.lat]),
      getColor: [255, 140, 0],
      widthUnits: 'meters',
      getWidth: 8,
      rounded: true,
    });

    const stopsLayer = new ScatterplotLayer<RouteStop>({
      id: 'route-stops',
      data: stops,
      getPosition: (d) => [d.lng, d.lat],
      getFillColor: [20, 120, 255],
      getRadius: 40,
      radiusUnits: 'meters',
      pickable: true,
      onClick: (info) => {
        const stop = info.object as RouteStop | null;
        if (stop?.place_id) {
          window.dispatchEvent(new CustomEvent('geo:focus-place', { detail: stop.place_id }));
        }
      },
    });

    const markerPosition = (() => {
      if (path.length === 0) return null;
      const ordered = [...path].sort((a, b) => (a.t ?? 0) - (b.t ?? 0));
      let current = ordered[0];
      for (const point of ordered) {
        if ((point.t ?? 0) <= cursor) {
          current = point;
        } else {
          break;
        }
      }
      return current;
    })();

    const markerLayer = markerPosition
      ? new ScatterplotLayer<RoutePoint>({
          id: 'route-marker',
          data: [markerPosition],
          getPosition: (d) => [d.lng, d.lat],
          getFillColor: [255, 80, 0],
          getRadius: 70,
          radiusUnits: 'meters',
        })
      : null;

    return markerLayer ? [pathLayer, stopsLayer, markerLayer] : [pathLayer, stopsLayer];
  }, [path, stops, cursor]);

  return <DeckGoogleMap center={center} layers={layers} googleMapsApiKey={googleMapsApiKey} />;
}
