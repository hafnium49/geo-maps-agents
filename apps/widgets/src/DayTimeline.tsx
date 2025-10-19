import './styles.css';

export type TimelineEntry = {
  name: string;
  arrival_iso: string;
  depart_iso: string;
  eta_sec: number;
  reason: string;
  maps_url?: string | null;
};

export type DayTimelineProps = {
  day: string;
  stops: TimelineEntry[];
};

export default function DayTimeline({ day, stops }: DayTimelineProps) {
  return (
    <div className="geo-timeline">
      <header>
        <h3>{day}</h3>
      </header>
      <ol>
        {stops.map((stop, index) => (
          <li key={`${stop.name}-${index}`}>
            <div className="time-range">
              <span>{new Date(stop.arrival_iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
              <span>→</span>
              <span>{new Date(stop.depart_iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
            </div>
            <div className="details">
              <div className="name">{stop.name}</div>
              <div className="meta">
                <span>{Math.round(stop.eta_sec / 60)} min travel</span>
                {stop.reason && <span className="reason">{stop.reason}</span>}
                {stop.maps_url && (
                  <a href={stop.maps_url} target="_blank" rel="noreferrer">
                    Open in Google Maps
                  </a>
                )}
              </div>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
