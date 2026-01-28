'use client';

import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

interface ClusterStats {
  cluster: number;
  avg_year: string;
  avg_value_m: number;
  res_pct: number;
  airbnb_total: number;
  schools: number;
  avg_sat: number | null;
  sat_vs_avg: number | null;
  crimes: number;
  crime_density: number;
  crime_vs_avg: number;
  subway_m: number;
  year_vs_avg: number;
  value_vs_avg: number;
  res_vs_avg: number;
  subway_vs_avg: number;
  airbnb_vs_avg: number;
  schools_vs_avg: number;
  res_sqft: number;
  com_sqft: number;
  res_sqft_vs_avg: number;
  com_sqft_vs_avg: number;
  airbnb_per_1k_units: number;
  airbnb_density_vs_avg: number;
}

const CLUSTER_COLORS = [
  '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
  '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
  '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
  '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
  '#ff6666', '#66ff66', '#6666ff', '#ffff66', '#ff66ff',
  '#66ffff', '#ff9966', '#9966ff', '#66ff99', '#ff6699',
  '#99ff66', '#6699ff', '#cc6699', '#99cc66', '#6699cc',
  '#cc9966', '#66cc99', '#9966cc', '#cccc66', '#66cccc',
];

type SortOption = {
  key: string;
  label: string;
  getValue: (stats: ClusterStats) => number;
  ascending: boolean; // true = lower is better (show first), false = higher is better
};

const SORT_OPTIONS: SortOption[] = [
  { key: 'cluster', label: 'Neighborhood #', getValue: (s) => s.cluster, ascending: true },
  { key: 'res_sqft', label: 'Residential $/sqft', getValue: (s) => s.res_sqft, ascending: false },
  { key: 'com_sqft', label: 'Commercial $/sqft', getValue: (s) => s.com_sqft, ascending: false },
  { key: 'year', label: 'Year Built', getValue: (s) => parseInt(s.avg_year) || 0, ascending: false },
  { key: 'res_pct', label: 'Residential %', getValue: (s) => s.res_pct, ascending: false },
  { key: 'subway', label: 'Subway Distance', getValue: (s) => s.subway_m, ascending: true },
  { key: 'airbnb', label: 'Airbnb Listings', getValue: (s) => s.airbnb_total, ascending: false },
  { key: 'airbnb_density', label: 'Airbnb per 1k Units', getValue: (s) => s.airbnb_per_1k_units, ascending: false },
  { key: 'schools', label: 'Schools', getValue: (s) => s.schools, ascending: false },
  { key: 'crime', label: 'Crime Rate', getValue: (s) => s.crime_density, ascending: true },
  { key: 'sat', label: 'SAT Score', getValue: (s) => s.avg_sat || 0, ascending: false },
];

export default function Home() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [hoveredCluster, setHoveredCluster] = useState<ClusterStats | null>(null);
  const [legendHoveredCluster, setLegendHoveredCluster] = useState<number | null>(null);
  const [clusterStatsMap, setClusterStatsMap] = useState<Map<number, ClusterStats>>(new Map());
  const [mapLoaded, setMapLoaded] = useState(false);
  const [sortBy, setSortBy] = useState<string>('cluster');

  useEffect(() => {
    if (!mapContainer.current || map.current) return;

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
      center: [-73.94, 40.78],
      zoom: 10,
      minZoom: 9,
      maxZoom: 16,
      maxBounds: [
        [-74.30, 40.45], // Southwest corner (into New Jersey)
        [-73.65, 40.95], // Northeast corner (into Long Island / Bronx)
      ],
    });

    map.current.on('error', (e) => {
      console.error('Map error:', e);
    });

    map.current.on('load', () => {
      if (!map.current) return;

      map.current.addSource('clusters', {
        type: 'geojson',
        data: '/clusters.geojson',
      });

      map.current.addLayer({
        id: 'clusters-fill',
        type: 'fill',
        source: 'clusters',
        paint: {
          'fill-color': [
            'match',
            ['get', 'cluster'],
            ...CLUSTER_COLORS.flatMap((color, i) => [i, color]),
            '#333333',
          ] as unknown as maplibregl.ExpressionSpecification,
          'fill-opacity': 0.7,
        },
      });

      map.current.addLayer({
        id: 'clusters-outline',
        type: 'line',
        source: 'clusters',
        paint: {
          'line-color': '#000000',
          'line-width': 0.5,
        },
      });

      // Add highlight layer for legend hover
      map.current.addLayer({
        id: 'clusters-highlight',
        type: 'line',
        source: 'clusters',
        paint: {
          'line-color': '#ffffff',
          'line-width': 3,
        },
        filter: ['==', ['get', 'cluster'], -999], // Initially hidden
      });

      // Extract cluster stats from GeoJSON for legend hover
      fetch('/clusters.geojson')
        .then((res) => res.json())
        .then((data) => {
          const statsMap = new Map<number, ClusterStats>();
          for (const feature of data.features) {
            const props = feature.properties as ClusterStats;
            if (props.cluster >= 0 && !statsMap.has(props.cluster)) {
              statsMap.set(props.cluster, props);
            }
          }
          setClusterStatsMap(statsMap);
        });

      map.current.on('mousemove', 'clusters-fill', (e) => {
        if (e.features && e.features.length > 0) {
          const props = e.features[0].properties as ClusterStats;
          setHoveredCluster(props);
        }
      });

      map.current.on('mouseleave', 'clusters-fill', () => {
        setHoveredCluster(null);
      });

      map.current.on('mouseenter', 'clusters-fill', () => {
        if (map.current) map.current.getCanvas().style.cursor = 'pointer';
      });

      map.current.on('mouseleave', 'clusters-fill', () => {
        if (map.current) map.current.getCanvas().style.cursor = '';
      });

      setMapLoaded(true);
    });

    return () => {
      map.current?.remove();
    };
  }, []);

  // Highlight cluster on map when hovering legend
  const highlightCluster = (clusterId: number | null) => {
    if (!map.current || !mapLoaded) return;

    if (clusterId !== null) {
      map.current.setFilter('clusters-highlight', ['==', ['get', 'cluster'], clusterId]);
      setLegendHoveredCluster(clusterId);
      const stats = clusterStatsMap.get(clusterId);
      if (stats) {
        setHoveredCluster(stats);
      }
    } else {
      map.current.setFilter('clusters-highlight', ['==', ['get', 'cluster'], -999]);
      setLegendHoveredCluster(null);
      setHoveredCluster(null);
    }
  };

  const formatSatDiff = (diff: number | null) => {
    if (diff === null || isNaN(diff)) return null;
    if (diff > 0) return { text: `+${diff}`, color: '#4ade80' }; // green
    if (diff < 0) return { text: `${diff}`, color: '#f87171' }; // red
    return { text: '0', color: '#9ca3af' }; // gray
  };

  const formatCrimeDiff = (diff: number | null) => {
    if (diff === null || isNaN(diff)) return null;
    // For crime, lower is better, so flip the colors
    if (diff > 0) return { text: `+${diff}`, color: '#f87171' }; // red (more crime = bad)
    if (diff < 0) return { text: `${diff}`, color: '#4ade80' }; // green (less crime = good)
    return { text: '0', color: '#9ca3af' }; // gray
  };

  // Generic formatter for positive = good metrics
  const formatPositiveDiff = (diff: number | null, decimals = 0) => {
    if (diff === null || isNaN(diff)) return null;
    const formatted = decimals > 0 ? Math.abs(diff).toFixed(decimals) : Math.abs(Math.round(diff));
    if (diff > 0) return { text: formatted, arrow: '▲', color: '#4ade80' }; // green up
    if (diff < 0) return { text: formatted, arrow: '▼', color: '#f87171' }; // red down
    return { text: '0', arrow: '', color: '#9ca3af' }; // gray
  };

  // Generic formatter for negative = good metrics (lower is better)
  const formatNegativeDiff = (diff: number | null, decimals = 0) => {
    if (diff === null || isNaN(diff)) return null;
    const formatted = decimals > 0 ? Math.abs(diff).toFixed(decimals) : Math.abs(Math.round(diff));
    if (diff > 0) return { text: formatted, arrow: '▲', color: '#f87171' }; // red up (more = bad)
    if (diff < 0) return { text: formatted, arrow: '▼', color: '#4ade80' }; // green down (less = good)
    return { text: '0', arrow: '', color: '#9ca3af' }; // gray
  };

  // Neutral formatter - shows direction with colors but no good/bad judgment
  const formatNeutralDiff = (diff: number | null, decimals = 0) => {
    if (diff === null || isNaN(diff)) return null;
    const formatted = decimals > 0 ? Math.abs(diff).toFixed(decimals) : Math.abs(Math.round(diff));
    if (diff > 0) return { text: formatted, arrow: '▲', color: '#4ade80' }; // green up
    if (diff < 0) return { text: formatted, arrow: '▼', color: '#f87171' }; // red down
    return { text: '0', arrow: '', color: '#9ca3af' }; // gray
  };

  // Get sorted cluster IDs based on selected sort option
  const getSortedClusters = (): number[] => {
    const sortOption = SORT_OPTIONS.find((o) => o.key === sortBy) || SORT_OPTIONS[0];
    const clusters = Array.from(clusterStatsMap.entries());

    clusters.sort((a, b) => {
      const valA = sortOption.getValue(a[1]);
      const valB = sortOption.getValue(b[1]);
      return sortOption.ascending ? valA - valB : valB - valA;
    });

    return clusters.map(([id]) => id);
  };

  return (
    <main className="relative" style={{ width: '100vw', height: '100vh' }}>
      <div ref={mapContainer} style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }} />

      {/* Title Card */}
      <div className="absolute top-4 right-4 bg-gradient-to-br from-gray-900 to-gray-800 text-white px-5 py-4 rounded-xl shadow-2xl border border-gray-700">
        <h1 className="text-2xl font-bold tracking-tight">the manhattan project</h1>
        <p className="text-sm text-gray-400 mt-1">a machine learning re-division of manhattan</p>
        <p className="text-xs text-gray-500 mt-1">SKATER clustering on Uber H3 hexagons</p>
        <div className="flex flex-wrap gap-2 mt-3 text-xs text-gray-500">
          <span className="px-2 py-1 bg-gray-800 rounded">NYC PLUTO</span>
          <span className="px-2 py-1 bg-gray-800 rounded">MTA</span>
          <span className="px-2 py-1 bg-gray-800 rounded">NYPD</span>
          <span className="px-2 py-1 bg-gray-800 rounded">Airbnb</span>
          <span className="px-2 py-1 bg-gray-800 rounded">DOE</span>
        </div>
      </div>

      {/* Stats Panel */}
      {hoveredCluster && hoveredCluster.cluster >= 0 && (
        <div className="absolute bottom-4 left-4 bg-gradient-to-br from-gray-900 to-gray-800 text-white px-5 py-4 rounded-xl shadow-2xl border border-gray-700 min-w-[300px]">
          <div className="flex items-center gap-3 mb-4">
            <div
              className="w-4 h-4 rounded-full"
              style={{ backgroundColor: CLUSTER_COLORS[hoveredCluster.cluster % CLUSTER_COLORS.length] }}
            />
            <h2 className="text-xl font-bold">Neighborhood {hoveredCluster.cluster}</h2>
          </div>

          <div className="space-y-3">
            {/* Building Info - with vs avg badges */}
            <div className="space-y-2 text-sm">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-gray-500 text-xs uppercase tracking-wide">Year Built</div>
                  <div className="font-semibold">{hoveredCluster.avg_year}</div>
                </div>
                <div
                  className="text-sm font-bold px-2 py-0.5 rounded-full flex items-center gap-1"
                  style={{
                    color: formatNeutralDiff(hoveredCluster.year_vs_avg)?.color,
                    backgroundColor: 'rgba(255,255,255,0.1)',
                  }}
                >
                  <span>{formatNeutralDiff(hoveredCluster.year_vs_avg)?.arrow}</span>
                  <span>{formatNeutralDiff(hoveredCluster.year_vs_avg)?.text} yrs</span>
                </div>
              </div>

              {hoveredCluster.res_sqft > 0 && (
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-gray-500 text-xs uppercase tracking-wide">Res. Est. $/sqft</div>
                    <div className="font-semibold">${hoveredCluster.res_sqft.toLocaleString()}</div>
                  </div>
                  <div
                    className="text-sm font-bold px-2 py-0.5 rounded-full flex items-center gap-1"
                    style={{
                      color: formatNeutralDiff(hoveredCluster.res_sqft_vs_avg)?.color,
                      backgroundColor: 'rgba(255,255,255,0.1)',
                    }}
                  >
                    <span>{formatNeutralDiff(hoveredCluster.res_sqft_vs_avg)?.arrow}</span>
                    <span>${formatNeutralDiff(hoveredCluster.res_sqft_vs_avg)?.text}</span>
                  </div>
                </div>
              )}

              {hoveredCluster.com_sqft > 0 && (
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-gray-500 text-xs uppercase tracking-wide">Com. Est. $/sqft</div>
                    <div className="font-semibold">${hoveredCluster.com_sqft.toLocaleString()}</div>
                  </div>
                  <div
                    className="text-sm font-bold px-2 py-0.5 rounded-full flex items-center gap-1"
                    style={{
                      color: formatNeutralDiff(hoveredCluster.com_sqft_vs_avg)?.color,
                      backgroundColor: 'rgba(255,255,255,0.1)',
                    }}
                  >
                    <span>{formatNeutralDiff(hoveredCluster.com_sqft_vs_avg)?.arrow}</span>
                    <span>${formatNeutralDiff(hoveredCluster.com_sqft_vs_avg)?.text}</span>
                  </div>
                </div>
              )}

              <div className="flex items-center justify-between">
                <div>
                  <div className="text-gray-500 text-xs uppercase tracking-wide">Residential %</div>
                  <div className="font-semibold">{hoveredCluster.res_pct}%</div>
                </div>
                <div
                  className="text-sm font-bold px-2 py-0.5 rounded-full flex items-center gap-1"
                  style={{
                    color: formatNeutralDiff(hoveredCluster.res_vs_avg)?.color,
                    backgroundColor: 'rgba(255,255,255,0.1)',
                  }}
                >
                  <span>{formatNeutralDiff(hoveredCluster.res_vs_avg)?.arrow}</span>
                  <span>{formatNeutralDiff(hoveredCluster.res_vs_avg)?.text}%</span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <div className="text-gray-500 text-xs uppercase tracking-wide">Subway Distance</div>
                  <div className="font-semibold">{hoveredCluster.subway_m}m avg</div>
                </div>
                <div
                  className="text-sm font-bold px-2 py-0.5 rounded-full flex items-center gap-1"
                  style={{
                    color: formatNegativeDiff(hoveredCluster.subway_vs_avg)?.color,
                    backgroundColor: 'rgba(255,255,255,0.1)',
                  }}
                >
                  <span>{formatNegativeDiff(hoveredCluster.subway_vs_avg)?.arrow}</span>
                  <span>{formatNegativeDiff(hoveredCluster.subway_vs_avg)?.text}m</span>
                </div>
              </div>
            </div>

            {/* Activity metrics */}
            <div className="border-t border-gray-700 pt-3 space-y-2 text-sm">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-gray-500 text-xs uppercase tracking-wide">Airbnb Listings</div>
                  <div className="font-semibold text-blue-400">{hoveredCluster.airbnb_total}</div>
                </div>
                <div
                  className="text-sm font-bold px-2 py-0.5 rounded-full flex items-center gap-1"
                  style={{
                    color: formatNeutralDiff(hoveredCluster.airbnb_vs_avg)?.color,
                    backgroundColor: 'rgba(255,255,255,0.1)',
                  }}
                >
                  <span>{formatNeutralDiff(hoveredCluster.airbnb_vs_avg)?.arrow}</span>
                  <span>{formatNeutralDiff(hoveredCluster.airbnb_vs_avg)?.text}</span>
                </div>
              </div>

              {hoveredCluster.airbnb_per_1k_units > 0 && (
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-gray-500 text-xs uppercase tracking-wide">Airbnb per 1k Units</div>
                    <div className="font-semibold text-blue-400">{hoveredCluster.airbnb_per_1k_units}</div>
                  </div>
                  <div
                    className="text-sm font-bold px-2 py-0.5 rounded-full flex items-center gap-1"
                    style={{
                      color: formatNeutralDiff(hoveredCluster.airbnb_density_vs_avg, 1)?.color,
                      backgroundColor: 'rgba(255,255,255,0.1)',
                    }}
                  >
                    <span>{formatNeutralDiff(hoveredCluster.airbnb_density_vs_avg, 1)?.arrow}</span>
                    <span>{formatNeutralDiff(hoveredCluster.airbnb_density_vs_avg, 1)?.text}</span>
                  </div>
                </div>
              )}

              <div className="flex items-center justify-between">
                <div>
                  <div className="text-gray-500 text-xs uppercase tracking-wide">Schools</div>
                  <div className="font-semibold text-yellow-400">{hoveredCluster.schools}</div>
                </div>
                <div
                  className="text-sm font-bold px-2 py-0.5 rounded-full flex items-center gap-1"
                  style={{
                    color: formatNeutralDiff(hoveredCluster.schools_vs_avg)?.color,
                    backgroundColor: 'rgba(255,255,255,0.1)',
                  }}
                >
                  <span>{formatNeutralDiff(hoveredCluster.schools_vs_avg)?.arrow}</span>
                  <span>{formatNeutralDiff(hoveredCluster.schools_vs_avg)?.text}</span>
                </div>
              </div>
            </div>

            {/* Crime density with vs avg */}
            <div className="border-t border-gray-700 pt-3 flex items-center justify-between">
              <div>
                <div className="text-gray-500 text-xs uppercase tracking-wide">Crime Rate (2024)</div>
                <div className="font-semibold">{hoveredCluster.crime_density}/ha</div>
              </div>
              <div
                className="text-sm font-bold px-2 py-0.5 rounded-full flex items-center gap-1"
                style={{
                  color: formatNegativeDiff(hoveredCluster.crime_vs_avg, 1)?.color,
                  backgroundColor: 'rgba(255,255,255,0.1)',
                }}
              >
                <span>{formatNegativeDiff(hoveredCluster.crime_vs_avg, 1)?.arrow}</span>
                <span>{formatNegativeDiff(hoveredCluster.crime_vs_avg, 1)?.text}/ha</span>
              </div>
            </div>

            {hoveredCluster.avg_sat && hoveredCluster.sat_vs_avg !== null && !isNaN(hoveredCluster.sat_vs_avg) && (
              <div className="border-t border-gray-700 pt-3 flex items-center justify-between">
                <div>
                  <div className="text-gray-500 text-xs uppercase tracking-wide">Avg SAT Score</div>
                  <div className="font-semibold">{hoveredCluster.avg_sat}</div>
                </div>
                <div
                  className="text-sm font-bold px-2 py-0.5 rounded-full flex items-center gap-1"
                  style={{
                    color: formatPositiveDiff(hoveredCluster.sat_vs_avg)?.color,
                    backgroundColor: 'rgba(255,255,255,0.1)',
                  }}
                >
                  <span>{formatPositiveDiff(hoveredCluster.sat_vs_avg)?.arrow}</span>
                  <span>{formatPositiveDiff(hoveredCluster.sat_vs_avg)?.text}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="absolute bottom-4 right-4 bg-gray-900/90 text-white px-4 py-3 rounded-xl text-xs">
        <div className="flex items-center justify-between mb-2">
          <span className="text-gray-500 uppercase tracking-wide">Sort by:</span>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="bg-gray-800 text-white text-xs rounded px-2 py-1 border border-gray-700 focus:outline-none focus:border-blue-500"
          >
            {SORT_OPTIONS.map((option) => (
              <option key={option.key} value={option.key}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        <div className="grid grid-cols-8 gap-1">
          {getSortedClusters().map((clusterId) => (
            <div
              key={clusterId}
              className="w-6 h-6 rounded flex items-center justify-center text-[10px] font-bold cursor-pointer transition-all hover:scale-110"
              style={{
                backgroundColor: CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length],
                boxShadow: legendHoveredCluster === clusterId ? '0 0 0 2px white' : 'none',
                transform: legendHoveredCluster === clusterId ? 'scale(1.2)' : 'scale(1)',
              }}
              onMouseEnter={() => highlightCluster(clusterId)}
              onMouseLeave={() => highlightCluster(null)}
            >
              {clusterId}
            </div>
          ))}
        </div>
        <div className="text-gray-600 text-[10px] mt-2 text-center">
          {sortBy !== 'cluster' && (
            <>
              {SORT_OPTIONS.find((o) => o.key === sortBy)?.ascending ? '← Lower' : '← Higher'} to{' '}
              {SORT_OPTIONS.find((o) => o.key === sortBy)?.ascending ? 'Higher →' : 'Lower →'}
            </>
          )}
        </div>
      </div>

      {/* Loading */}
      {!mapLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80">
          <div className="text-center">
            <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <div className="text-white text-xl">Loading map...</div>
          </div>
        </div>
      )}
    </main>
  );
}
