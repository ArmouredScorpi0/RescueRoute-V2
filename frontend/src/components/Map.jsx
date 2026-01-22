import React, { useEffect, useState } from 'react';
import DeckGL from '@deck.gl/react';
import { Map as MapLibre } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';

// Public Map Style
const MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

export default function Map({ initialView, layers = [], onClick }) {
  const [viewState, setViewState] = useState(initialView);

  // When the parent changes the city, update the camera
  useEffect(() => {
    if (initialView) {
      setViewState(prev => ({
        ...prev,
        ...initialView,
        transitionDuration: 2000 // Smooth fly effect
      }));
    }
  }, [initialView]);

  return (
    <DeckGL
      initialViewState={viewState}
      viewState={viewState}
      onViewStateChange={({viewState}) => setViewState(viewState)}
      controller={true}
      layers={layers}
      getTooltip={({object}) => object && object.tooltip}
      onClick={onClick}
    >
      <MapLibre
        mapStyle={MAP_STYLE}
        style={{width: '100vw', height: '100vh'}}
      />
    </DeckGL>
  );
}