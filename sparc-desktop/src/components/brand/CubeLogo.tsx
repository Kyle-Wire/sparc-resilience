import { useMemo } from 'react';

type LogoHue = 'ink' | 'red' | 'purple' | 'amber' | 'crimson';

interface CubeLogoProps {
  size?: number;
  animate?: boolean;
  hue?: LogoHue;
  density?: number;
  intensity?: number;
  className?: string;
}

const HUE_MAP: Record<LogoHue, string> = {
  ink: '#1a1416',
  red: '#e73c25',
  purple: '#602468',
  amber: '#e79024',
  crimson: '#b52a2a',
};

export default function CubeLogo({
  size = 120,
  animate = true,
  hue = 'ink',
  intensity = 1,
  className,
}: CubeLogoProps) {
  const stroke = HUE_MAP[hue] ?? HUE_MAP.ink;
  const uid = useMemo(() => 'cl-' + Math.random().toString(36).slice(2, 8), []);
  const dur = animate ? 6 / Math.max(0.4, intensity) : 0;

  const css = `
    #${uid} .cube-frame { stroke: ${stroke}; stroke-width: 7.56px; stroke-linecap: round; stroke-linejoin: round; fill: none; }
    #${uid} .cube-core  { fill: #ffffff; stroke: ${stroke}; stroke-width: 4.54px; stroke-linejoin: round; }
    #${uid} .cube-poly  { stroke: ${stroke}; stroke-width: 7.56px; stroke-linejoin: round; fill: none; }
    #${uid} .cube-tick  { stroke: #ffffff; stroke-width: 7.56px; stroke-linecap: round; fill: none; }
    #${uid} .matter     { stroke: ${stroke}; stroke-width: 2.27px; stroke-linecap: round; fill: none; }
    #${uid} .matter-a   { stroke-dasharray: 4 6; animation: ${uid}-flow-a ${dur}s linear infinite; }
    #${uid} .matter-b   { stroke-dasharray: 3 5; animation: ${uid}-flow-b ${dur * 1.3}s linear infinite; }
    #${uid} .matter-c   { stroke-dasharray: 6 4; animation: ${uid}-flow-c ${dur * 0.7}s linear infinite reverse; }
    #${uid} .matter-d   { stroke-dasharray: 2 7; animation: ${uid}-flow-a ${dur * 1.6}s linear infinite; }
    #${uid} .core       { transform-origin: 380px 335px; animation: ${uid}-breath ${dur * 1.2}s ease-in-out infinite; }
    @keyframes ${uid}-flow-a { to { stroke-dashoffset: -100; } }
    @keyframes ${uid}-flow-b { to { stroke-dashoffset: -80; } }
    @keyframes ${uid}-flow-c { to { stroke-dashoffset: -120; } }
    @keyframes ${uid}-breath { 0%,100% { transform: scale(1); } 50% { transform: scale(1.03); } }
  `;

  return (
    <div id={uid} className={className} style={{ width: size, height: size, display: 'inline-block' }}>
      <style>{css}</style>
      <svg viewBox="138 48 516 516" xmlns="http://www.w3.org/2000/svg" style={{ width: '100%', height: '100%', display: 'block' }}>
        {/* Core organic blob inside cube */}
        <path className="cube-core core" d="M240.18,251.69v101.37c58.19,21.34,107.47,48.79,142.21,86.21,16.68,14.15,32.91,22.48,48.4,21.17,15.71.18,28.03-8.11,37.81-22.69l33.27-57.47c10.05-17.17,21.32-31.94,36.3-39.32l21.17-4.54c-11.47-19.67-19.51-38.45-22.69-55.96-1.75-16.95-3.39-32.82-4.54-43.86-2.44-10.53-7.75-19.25-16.64-25.71-9.58-9.1-21-14.96-34.79-16.64-15.2-.63-29.73-7.56-43.86-18.15-9.73-7.32-18.62-16.24-27.22-25.71-6.84-5.64-14.52-5.9-22.69-3.02-10.98,3.15-20.75,14.26-30.25,27.22-8.35,11.24-18.79,25.94-30.25,42.35-8.73,13.49-24.09,24.35-40.62,34.75-13.89,6.13-29.23,5.5-45.62,0Z" />

        {/* Hex cube frame */}
        <polygon className="cube-poly" points="396.26 71.88 194.84 189.71 194.84 418.23 396.26 540.12 597.16 419.61 597.16 189.77 396.26 71.88" />
        <polyline className="cube-poly" points="194.84 189.71 396.26 313.66 597.16 189.77" />
        <line className="cube-poly" x1="396.26" y1="540.12" x2="396.26" y2="313.66" />

        {/* White accent ticks */}
        <line className="cube-tick" x1="396.26" y1="71.88" x2="396.26" y2="145.57" />
        <line className="cube-tick" x1="597.16" y1="419.61" x2="509.68" y2="367.94" />
        <line className="cube-tick" x1="194.84" y1="418.23" x2="279.26" y2="368.99" />

        {/* Rear polyline (back face trace) */}
        <polyline className="cube-frame" style={{ opacity: 0.25 }} points="397.85 540.12 592.39 418.14 408.64 312.98 284.67 366.5 199.61 418.59" />

        {/* Matter — flowing dashed interior paths */}
        <g>
          <path className="matter matter-a" d="M246.87,295.28c8.16,6.16,16.55,12.8,25.1,19.97,12.27,10.29,15.52,15.66,27.23,24.2,8.31,6.06,14.86,10.76,24.96,14.37,3.44,1.23,14.89,5.05,30.25,4.54,4.95-.17,16.16-.98,28.74-6.05,9.55-3.85,18.2-9.48,43.86-36.3,12.28-12.84,15.65-19.25,30.25-33.27,4.99-4.8,9.43-8.73,12.82-11.65" />
          <path className="matter matter-b" d="M254.97,260.03c4.77,4.31,11.05,10.4,17.76,18.16,15.72,18.17,16.54,25.53,28.74,34.79,4.07,3.09,16.71,11.93,34.79,13.61,19.08,1.78,33.62-5.4,40.84-9.07,18.79-9.58,19.81-18.3,45.37-42.35,13.55-12.75,26.78-25.19,45.37-34.79,20.02-10.33,39.47-13.87,54.44-14.94" />
          <path className="matter matter-c" d="M256.54,259.89c7.73,6.29,14.8,11.34,20.73,15.27,14.7,9.76,22.25,14.65,31.76,15.12,10.67.54,18.84-3.87,25.71-7.56,7.58-4.08,13.15-8.75,28.74-25.71,21.68-23.59,21.25-25.81,31.76-34.79,6.07-5.19,15.69-13.28,30.25-19.66,14.84-6.51,28.42-8.44,37.84-8.95" />
          <path className="matter matter-d" d="M323.8,388.79c8.21,3.12,20.4,6.43,35.15,5.86,26.59-1.04,45.21-14.09,52.94-19.66,8.38-6.04,8.82-8.16,39.32-42.35,23.49-26.33,27.03-29.32,30.25-31.76,11.14-8.45,27.45-17.07,51.45-20.75" />
          <line className="matter matter-a" x1="274.25" y1="358.41" x2="274.25" y2="282.7" />
          <path className="matter matter-b" d="M299.78,258.29c2.23,7.26,5.48,18.19,7.74,27.46,2.36,9.68,3.87,14.48,4.54,21.17,1.51,15.12,1.37,17.91,1.51,36.3.02,2.03,0,1.23,0,28.74v9.09" />
          <path className="matter matter-c" d="M332.52,217.64c3.4,19.21,6.17,35.62,8.28,48.45,4.4,26.74,6.61,40.34,7.56,51.42,1.17,13.68,1.02,20,3.02,37.81.91,8.11,2.12,15.49,4.54,30.25,3.37,20.6,5.49,30.18,13.61,37.81.52.49,2.7,2.51,6.05,4.54,13.31,8.07,27.71,5.82,33.27,4.54,11.2-2.59,18.27-8.58,24.2-13.61,2.7-2.29,9.66-8.49,19.66-24.2,14.16-22.24,11.99-28.3,24.2-43.86,5.54-7.06,11.14-14.05,21.17-19.66,14.84-8.3,29.34-7.96,34.79-7.56,8.37.61,15.17,2.8,19.93,4.81" />
          <path className="matter matter-a" d="M364.11,173.77c5.51,38.99,10.7,70.44,14.5,92.32,7.4,42.66,7.5,37.5,10.59,58.99,4.17,29.01,5.35,47.98,13.61,75.62,2.91,9.73,7.26,22.45,16.64,36.3,5.08,7.51,10.28,13.34,14.49,17.56" />
          <path className="matter matter-b" d="M400.45,152.63c2.62,17.64,5.01,32.53,6.9,43.89,3.8,22.88,6.29,37.74,10.59,57.47,3.14,14.44,7.56,31.76,7.56,31.76.28,1.13.54,2.17.77,3.11" />
          <path className="matter matter-c" d="M442.64,187.35c1.2,9.86,2.97,20.54,5.54,31.85,3.92,17.28,8.9,33.87,14.03,47.13" />
          <path className="matter matter-d" d="M477.86,201.01c1.73,7.68,4.29,16.42,8.13,25.76,2.86,6.97,5.98,13.16,9.05,18.54" />
          <path className="matter matter-a" d="M504.66,362.44c-5.09-2.7-11.32-6.84-17.15-13.16-4.35-4.7-8.32-8.82-12.1-15.12-5.86-9.79-8.35-16.47-10.59-25.71-1.27-5.25-1.49-17.66-.31-30.29" />
          <path className="matter matter-b" d="M471.93,418.99c-7.08-6.09-12.38-12.01-16.19-16.78-3.56-4.46-8.86-11.16-13.61-21.17-4.12-8.68-5.86-15.69-7.56-22.69-2.65-10.93-3.42-19.21-4.54-31.76-.59-6.64-1.2-15.37-1.45-25.72" />
          <path className="matter matter-c" d="M538,336.39c-10.67-6.46-20.18-14.11-24.79-18.87-4.14-4.28-7.54-7.84-10.59-13.61-4.22-7.98-5.05-15.01-6.05-24.2-.62-5.68-1.04-13.42-.29-22.68" />
        </g>
      </svg>
    </div>
  );
}
