import axios from 'axios';

// Ensure this matches your Backend Port
const API_URL = "/api";

export const api = {
    // 1. Get List of Cities
    getConfig: async () => {
        const res = await axios.get(`${API_URL}/config`);
        return res.data;
    },

    // 2. Initialize a City (Load Graph)
    initCity: async (cityName) => {
        const res = await axios.post(`${API_URL}/init-city`, { city_name: cityName });
        return res.data;
    },

    // 3. Flood Prediction
    predictFlood: async (cityName, rain) => {
        const res = await axios.post(`${API_URL}/predict-flood`, {
            city_name: cityName,
            rain_mm: rain
        });
        return res.data;
    },

    // 4. Scan Zones (AI Scout)
    scanZones: async (cityName, rain, target) => {
        const res = await axios.post(`${API_URL}/scan-zones`, {
            city_name: cityName,
            rain_mm: rain,
            target_coords: target
        });
        return res.data;
    },

    // 5. Calculate Route
    calculateRoute: async (payload) => {
        const res = await axios.post(`${API_URL}/calculate-route`, payload);
        return res.data;
    }
};