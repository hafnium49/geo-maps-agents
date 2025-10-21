import { ComponentType } from 'react';
import { createRoot } from 'react-dom/client';
import H3Heatmap from './H3Heatmap';
import PoiClusters from './PoiClusters';
import IsochroneRings from './IsochroneRings';
import RoutePlayback from './RoutePlayback';
import DayTimeline from './DayTimeline';
import RefineOutlines from './RefineOutlines';
import RefineDots from './RefineDots';
import './styles.css';

type WidgetFactory = (element: HTMLElement, props: any) => () => void;

declare global {
  interface Window {
    geoWidgets?: Record<string, WidgetFactory>;
  }
}

function render(Component: ComponentType<any>): WidgetFactory {
  return (element, props) => {
    const root = createRoot(element);
    root.render(<Component {...props} />);
    return () => root.unmount();
  };
}

const registry: Record<string, WidgetFactory> = {
  'geo.h3Heatmap': render((props) => <H3Heatmap {...props} />),
  'geo.poiClusters': render((props) => <PoiClusters {...props} />),
  'geo.isochroneRings': render((props) => <IsochroneRings {...props} />),
  'geo.routePlayback': render((props) => <RoutePlayback {...props} />),
  'geo.dayTimeline': render((props) => <DayTimeline {...props} />),
  'geo.refineOutlines': render((props) => <RefineOutlines {...props} />),
  'geo.refineDots': render((props) => <RefineDots {...props} />),
};

window.geoWidgets = { ...(window.geoWidgets ?? {}), ...registry };

export default registry;
