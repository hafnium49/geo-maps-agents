import { GoogleMapsOverlay } from '@deck.gl/google-maps';
import { Layer } from '@deck.gl/core';
import { useEffect, useRef, useState } from 'react';
import { Loader } from '@googlemaps/js-api-loader';

export type DeckGoogleMapProps = {
  center: { lat: number; lng: number };
  zoom?: number;
  mapId?: string;
  layers: Layer<any>[];
  googleMapsApiKey?: string;
};

export function DeckGoogleMap({
  center,
  zoom = 13,
  mapId,
  layers,
  googleMapsApiKey,
}: DeckGoogleMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<GoogleMapsOverlay | null>(null);
  const mapRef = useRef<google.maps.Map | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    async function ensureGoogleMaps(): Promise<void> {
      if (typeof window === 'undefined') return;
      if ((window as any).google?.maps) return;
      if (!googleMapsApiKey) {
        setError('Google Maps API not loaded. Provide googleMapsApiKey or load the JS API script globally.');
        return;
      }
      const loader = new Loader({
        apiKey: googleMapsApiKey,
        version: 'weekly',
        libraries: ['maps'] as any,
      });
      await loader.load();
    }

    async function init(): Promise<void> {
      await ensureGoogleMaps();
      if (!isMounted) return;
      if (!containerRef.current) return;
      if (!(window as any).google?.maps) {
        setError('Google Maps library unavailable.');
        return;
      }

      mapRef.current = new google.maps.Map(containerRef.current, {
        center,
        zoom,
        mapId,
        disableDefaultUI: true,
        clickableIcons: false,
      });
      overlayRef.current = new GoogleMapsOverlay({ layers });
      overlayRef.current.setMap(mapRef.current);
      setError(null);
    }

    init();

    return () => {
      isMounted = false;
      if (overlayRef.current) {
        overlayRef.current.setMap(null);
        overlayRef.current = null;
      }
      mapRef.current = null;
    };
  }, [center.lat, center.lng, zoom, mapId, layers, googleMapsApiKey]);

  useEffect(() => {
    if (overlayRef.current) {
      overlayRef.current.setProps({ layers });
    }
  }, [layers]);

  if (error) {
    return <div className="geo-widget-error">{error}</div>;
  }

  return <div ref={containerRef} style={{ width: '100%', height: '100%', borderRadius: 8, overflow: 'hidden' }} />;
}

export default DeckGoogleMap;
