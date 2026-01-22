from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
import os
import networkx as nx
import osmnx as ox
import rasterio
from rasterio import features
import numpy as np
from scipy.ndimage import binary_dilation
from shapely.geometry import LineString
from geopy.distance import geodesic
import base64
import io
from PIL import Image

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. CONFIGURATION & DATABASE ---
CITY_DATABASE = {
    "🇮🇳 NEW DELHI (Yamuna Basin)": {
        "center": (28.6139, 77.2090),
        "zoom": 11,
        "graph_file": "new_delhi.graphml",
        "dem_file": "delhi_srtm.tif",     
        "hq_marker": (28.4500, 76.8200),
        "target_marker": (28.6280, 77.2750),
        "dem_sensitivity": 0.5,
        "landmarks": [
            {"name": "Jawaharlal Nehru Stadium", "coords": (28.5828, 77.2344)},
            {"name": "Indira Gandhi Airport", "coords": (28.5562, 77.1000)},
            {"name": "Connaught Place Hub", "coords": (28.6304, 77.2177)},
            {"name": "Rohini Sector 13", "coords": (28.7163, 77.1213)},
            {"name": "Dwarka Sector 21", "coords": (28.5524, 77.0583)}
        ]
    },
    "🇺🇸 PITTSBURGH (River Confluence)": {
        "center": (40.4406, -79.9959),
        "zoom": 12,
        "graph_file": "pittsburgh.graphml", 
        "dem_file": "output_USGS30m.tif",
        "hq_marker": (40.4914, -80.2328), # Airport
        "target_marker": (40.4417, -80.0128),
        "dem_sensitivity": 0.15,
        "landmarks": [
            {"name": "Heinz Field", "coords": (40.4468, -80.0158)},
            {"name": "Carnegie Mellon", "coords": (40.4432, -79.9428)},
            {"name": "Mount Washington", "coords": (40.4313, -80.0050)},
            {"name": "Highland Park", "coords": (40.4831, -79.9022)}
        ]
    }
}

# Global Store
LOADED_GRAPHS = {}

# --- 2. DATA LOADERS ---
def get_graph(city_name: str):
    """Retrieves graph from memory or loads it if missing."""
    if city_name in LOADED_GRAPHS:
        return LOADED_GRAPHS[city_name]
    
    city_data = CITY_DATABASE.get(city_name)
    if not city_data:
        return None

    filename = city_data['graph_file']
    
    # --- FULL POWER MODE (LOCAL 24GB RAM) ---
    if not os.path.exists(filename):
        print(f"⚠️ File {filename} not found locally.")
        print("Creating graph from OSM live download (this might take a moment)...")
        # If file missing, download full city (dist=10000 = 10km radius)
        G = ox.graph_from_point(city_data['center'], dist=10000, network_type='drive')
    else:
        print(f"🚀 Loading massive graph from {filename}...")
        # This will take ~5-10GB RAM
        G = ox.load_graphml(filename)
    
    # PHYSICS ENABLED
    print("⚡ Calculating edge speeds and travel times...")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    
    LOADED_GRAPHS[city_name] = G
    print(f"✅ Graph loaded successfully: {len(G.nodes)} nodes.")
    return G

# --- 3. PHYSICS & IMPACT ENGINE ---
def calculate_flood_logic(dem_path, rain_mm, sensitivity):
    if not os.path.exists(dem_path):
        return None, None, None, None

    with rasterio.open(dem_path) as src:
        elevation_data = src.read(1)
        transform = src.transform
        bounds = [
            [float(src.bounds.bottom), float(src.bounds.left)], 
            [float(src.bounds.top), float(src.bounds.right)]
        ]
        
        valid_mask = elevation_data > -500
        if not np.any(valid_mask):
            return None, None, None, None

        riverbed_level = np.percentile(elevation_data[valid_mask], 5)
        water_level = riverbed_level + ((rain_mm / 10.0) * sensitivity)
        flood_mask = (elevation_data < water_level) & valid_mask
        
        return flood_mask, transform, bounds, water_level

def assess_road_damage(G, flood_mask, transform, flood_shape):
    if flood_mask is None: return {}
    
    graph_nodes_data = {n: {'x': G.nodes[n]['x'], 'y': G.nodes[n]['y']} for n in G.nodes}

    rows, cols = flood_shape
    chaos_mask = binary_dilation(flood_mask, iterations=4) & ~flood_mask
    
    node_status = {}
    
    for node_id, data in graph_nodes_data.items():
        try:
            r, c = rasterio.transform.rowcol(transform, data['x'], data['y'])
            if 0 <= r < rows and 0 <= c < cols:
                if flood_mask[r, c]:
                    node_status[node_id] = "FLOOD"
                elif chaos_mask[r, c]:
                    node_status[node_id] = "CHAOS"
                else:
                    node_status[node_id] = "SAFE"
            else:
                node_status[node_id] = "SAFE"
        except:
            node_status[node_id] = "SAFE"
            
    return node_status

# --- 4. SMART SCOUT LOGIC ---
def logic_scan_staging_zones(G, node_status, hq_coords, target_coords, landmarks):
    safe_nodes = [n for n, status in node_status.items() if status == "SAFE"]
    if not safe_nodes: return "PROTOCOL_OMEGA"

    dist_hq_target = geodesic(hq_coords, target_coords).km
    
    top_connected = sorted(safe_nodes, key=lambda n: G.degree[n], reverse=True)[:500]
    
    valid_candidates = []
    
    for n in top_connected:
        node_coords = (G.nodes[n]['y'], G.nodes[n]['x'])
        dist_to_target = geodesic(node_coords, target_coords).km
        
        if dist_to_target < (dist_hq_target * 0.8):
            valid_candidates.append({
                "id": n,
                "coords": node_coords,
                "dist": dist_to_target,
                "degree": G.degree[n]
            })
            
    if not valid_candidates:
        return "PROTOCOL_OMEGA" 

    valid_candidates.sort(key=lambda x: x['dist'])
    best_picks = valid_candidates[:3]
    
    results = []
    for i, pick in enumerate(best_picks):
        best_name = f"Sector {chr(65+i)}"
        min_dist = 5.0 
        
        for lm in landmarks:
            d = geodesic(pick['coords'], lm['coords']).km
            if d < min_dist:
                best_name = f"Near {lm['name']}"
                min_dist = d
        
        results.append({
            "id": pick['id'],
            "name": best_name,
            "coords": pick['coords'],
            "dist_km": pick['dist']
        })
        
    return results

# --- 5. ROUTING LOGIC ---
def logic_get_route(G, node_status, manual_blocks, start_node, end_node):
    G_routing = G.copy()
    
    # 1. Apply Natural Disaster Penalties
    for u, v, k, data in G_routing.edges(keys=True, data=True):
        status_u = node_status.get(u, "SAFE")
        status_v = node_status.get(v, "SAFE")
        
        base_weight = data.get('length', 1)
        
        if status_u == "FLOOD" or status_v == "FLOOD":
            G_routing[u][v][k]['length'] = base_weight * 10000 
        elif status_u == "CHAOS" or status_v == "CHAOS":
            G_routing[u][v][k]['length'] = base_weight * 5
            
    # 2. Apply Manual Threat Penalties
    for block_lat, block_lon in manual_blocks:
        u_block, v_block, key_block = ox.nearest_edges(G, block_lon, block_lat)
        if G_routing.has_edge(u_block, v_block, key_block):
            G_routing[u_block][v_block][key_block]['length'] *= 10000
            
    try:
        path = nx.shortest_path(G_routing, start_node, end_node, weight='length')
        return path
    except nx.NetworkXNoPath:
        return None

def logic_calculate_confidence(G, route, node_status, manual_blocks):
    if not route: return 0, "No Route Possible"
    
    score = 100
    penalty_log = []
    
    chaos_count = 0
    total_nodes = len(route)
    
    for n in route:
        status = node_status.get(n, "SAFE")
        if status == "CHAOS":
            chaos_count += 1
            
    chaos_ratio = chaos_count / total_nodes
    if chaos_ratio > 0.1:
        deduction = int(chaos_ratio * 50)
        score -= deduction
        penalty_log.append(f"Traffic Risk (-{deduction})")
        
    threat_deduction = 0
    for lat, lon in manual_blocks:
        for n in route[::10]: 
            n_coords = (G.nodes[n]['y'], G.nodes[n]['x'])
            if geodesic(n_coords, (lat, lon)).km < 0.5:
                threat_deduction += 20
                break
    if threat_deduction > 0:
        score -= threat_deduction
        penalty_log.append(f"Threat Proximity (-{threat_deduction})")
        
    score = max(0, score)
    reason = "Optimal Conditions"
    if penalty_log:
        reason = ", ".join(penalty_log)
        
    return score, reason


# --- API REQUEST MODELS ---
class CityRequest(BaseModel):
    city_name: str

class FloodRequest(BaseModel):
    city_name: str
    rain_mm: float

class ScanRequest(BaseModel):
    city_name: str
    rain_mm: float
    target_coords: Tuple[float, float]

class RouteRequest(BaseModel):
    city_name: str
    rain_mm: float
    start_coords: Tuple[float, float]
    end_coords: Tuple[float, float]
    manual_blocks: List[Tuple[float, float]]

# --- API ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "active", "system": "RescueRoute-V2 Brain"}

@app.get("/config")
def get_config():
    return CITY_DATABASE

@app.post("/init-city")
def init_city(req: CityRequest):
    G = get_graph(req.city_name)
    if not G:
        raise HTTPException(status_code=404, detail="City not found")
    return {"status": "ready", "city": req.city_name}

@app.post("/predict-flood")
def predict_flood(req: FloodRequest):
    city_data = CITY_DATABASE.get(req.city_name)
    if not city_data: raise HTTPException(status_code=404, detail="City not found")

    flood_mask, transform, bounds, w_level = calculate_flood_logic(
        city_data['dem_file'], req.rain_mm, city_data['dem_sensitivity']
    )
    
    if flood_mask is None:
        return {"flood": False, "stats": 0}

    G = get_graph(req.city_name)
    node_status = assess_road_damage(G, flood_mask, transform, flood_mask.shape)
    
    flooded_count = sum(1 for v in node_status.values() if v == "FLOOD")
    
    return {
        "flood": True,
        "water_level": w_level,
        "flooded_nodes_count": flooded_count,
        "bounds": bounds
    }

@app.post("/scan-zones")
def scan_zones(req: ScanRequest):
    city_data = CITY_DATABASE.get(req.city_name)
    G = get_graph(req.city_name)
    
    flood_mask, transform, _, _ = calculate_flood_logic(
        city_data['dem_file'], req.rain_mm, city_data['dem_sensitivity']
    )
    node_status = assess_road_damage(G, flood_mask, transform, flood_mask.shape)
    
    result = logic_scan_staging_zones(
        G, node_status, city_data['hq_marker'], req.target_coords, city_data['landmarks']
    )
    
    return {"result": result}

@app.post("/calculate-route")
def calculate_route(req: RouteRequest):
    city_data = CITY_DATABASE.get(req.city_name)
    G = get_graph(req.city_name)
    
    flood_mask, transform, _, _ = calculate_flood_logic(
        city_data['dem_file'], req.rain_mm, city_data['dem_sensitivity']
    )
    node_status = assess_road_damage(G, flood_mask, transform, flood_mask.shape)
    
    start_node = ox.nearest_nodes(G, req.start_coords[1], req.start_coords[0])
    end_node = ox.nearest_nodes(G, req.end_coords[1], req.end_coords[0])
    
    route = logic_get_route(G, node_status, req.manual_blocks, start_node, end_node)
    
    if not route:
        return {"success": False, "reason": "No path found"}
        
    confidence, reason = logic_calculate_confidence(G, route, node_status, req.manual_blocks)
    
    path_coords = [[G.nodes[n]['y'], G.nodes[n]['x']] for n in route]
    
    return {
        "success": True,
        "path": path_coords,
        "confidence": confidence,
        "reason": reason
    }