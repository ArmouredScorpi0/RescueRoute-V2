import React, { useEffect, useState } from 'react';
import Map from './components/Map';
import { api } from './api';
import './App.css';

export default function App() {
  const [cityConfig, setCityConfig] = useState(null);
  const [selectedCity, setSelectedCity] = useState(null);
  const [viewState, setViewState] = useState(null);
  const [loading, setLoading] = useState(false);

  // 1. Load Config on Start
  useEffect(() => {
    async function boot() {
      try {
        const data = await api.getConfig();
        setCityConfig(data);
        // Select first city by default
        const firstCity = Object.keys(data)[0];
        handleCityChange(firstCity, data[firstCity]);
      } catch (err) {
        console.error("Backend Disconnected:", err);
      }
    }
    boot();
  }, []);

  // 2. Handle City Switch
  const handleCityChange = async (cityName, cityData) => {
    setLoading(true);
    setSelectedCity(cityName);
    
    // Update Map Camera
    setViewState({
      latitude: cityData.center[0],
      longitude: cityData.center[1],
      zoom: cityData.zoom,
      pitch: 45,
      bearing: 0
    });

    // Tell Backend to Load Graph
    await api.initCity(cityName);
    setLoading(false);
  };

  return (
    <div>
      {/* Map Layer */}
      {viewState && <Map initialView={viewState} />}

      {/* Sidebar UI */}
      <div className="control-panel">
        <h2>🚀 RescueRoute V2</h2>
        <div className="status-indicator">
          <span className={`dot ${selectedCity ? 'green' : 'red'}`}></span>
          {selectedCity ? "SYSTEM ONLINE" : "CONNECTING..."}
        </div>

        {/* City Selector */}
        {cityConfig && (
          <div className="section">
            <label>📍 Operation Theater</label>
            <select 
              value={selectedCity || ""} 
              onChange={(e) => handleCityChange(e.target.value, cityConfig[e.target.value])}
              disabled={loading}
            >
              {Object.keys(cityConfig).map(city => (
                <option key={city} value={city}>{city}</option>
              ))}
            </select>
          </div>
        )}

        {loading && <div className="loading-bar">LOADING TERRAIN DATA...</div>}
      </div>
    </div>
  );
}