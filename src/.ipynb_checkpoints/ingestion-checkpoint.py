"""
ingestion.py
------------
Pulls Sentinel-2 imagery from Google Earth Engine
for a target agricultural district in Maharashtra, India.
"""

import ee
import geemap
import numpy as np
import pandas as pd
import json


def initialize_gee(project_id: str = None):
    """Initialize Google Earth Engine."""
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        print("✅ GEE initialized successfully.")
    except Exception as e:
        print(f"❌ GEE init failed: {e}")
        print("Run: earthengine authenticate")
        raise


def get_study_area(district: str = "Wardha") -> ee.Geometry:
    """
    Returns a bounding box for a Maharashtra district.
    Default: Wardha (major cotton farming district in Vidarbha).
    You can replace with any district coordinates.
    
    Coordinates format: [west, south, east, north]
    """
    districts = {
        "Wardha":   [78.35, 20.40, 79.10, 21.10],
        "Yavatmal": [77.50, 19.80, 78.60, 20.80],
        "Amravati": [77.20, 20.50, 78.00, 21.30],
        "Nashik":   [73.50, 19.50, 74.50, 20.50],
    }
    coords = districts.get(district, districts["Wardha"])
    geometry = ee.Geometry.BBox(coords[0], coords[1], coords[2], coords[3])
    print(f"📍 Study area set to: {district}, Maharashtra")
    return geometry


def get_sentinel2_collection(
    geometry: ee.Geometry,
    start_date: str = "2023-10-01",
    end_date: str = "2024-03-31",
    cloud_threshold: int = 20
) -> ee.Image:
    """
    Fetches and composites Sentinel-2 SR imagery.
    Rabi season (Oct-Mar) is ideal for Maharashtra agriculture.
    
    Returns a cloud-masked median composite.
    """

    def mask_s2_clouds(image):
        """Mask clouds using Sentinel-2 QA band."""
        qa = image.select("QA60")
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = (
            qa.bitwiseAnd(cloud_bit_mask).eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )
        return image.updateMask(mask).divide(10000)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        .map(mask_s2_clouds)
    )

    count = collection.size().getInfo()
    print(f"🛰️  Found {count} Sentinel-2 scenes after cloud filtering.")

    composite = collection.median().clip(geometry)
    print(f"✅ Median composite created for {start_date} → {end_date}")
    return composite


def get_soilgrids_soc(geometry: ee.Geometry) -> ee.Image:
    """
    Fetches SoilGrids Soil Organic Carbon (SOC) baseline from GEE.
    Unit: g/kg at 0-30cm depth. 
    This acts as our ground-truth reference layer.
    """
    # SoilGrids SOC mean at 0-30cm (dg/kg * 10 = g/kg)
    soc = (
        ee.Image("projects/soilgrids-isric/soc_mean")
        .select("soc_0-30cm_mean")
        .clip(geometry)
    )
    print("🌱 SoilGrids SOC baseline loaded.")
    return soc


def sample_pixels(
    composite: ee.Image,
    soc_layer: ee.Image,
    geometry: ee.Geometry,
    n_samples: int = 500
) -> pd.DataFrame:
    """
    Samples pixel values from Sentinel-2 composite + SOC layer
    to create a training/inference dataset.
    
    Returns a pandas DataFrame with band values + SOC target.
    """
    bands = ["B2", "B3", "B4", "B8", "B11", "B12"]  # Blue, Green, Red, NIR, SWIR1, SWIR2
    
    combined = composite.select(bands).addBands(soc_layer.rename("SOC"))
    
    samples = combined.sample(
        region=geometry,
        scale=20,          # 20m resolution
        numPixels=n_samples,
        seed=42,
        geometries=True
    )

    # Convert to DataFrame
    features = samples.getInfo()["features"]
    
    records = []
    for f in features:
        props = f["properties"]
        coords = f["geometry"]["coordinates"]
        props["longitude"] = coords[0]
        props["latitude"] = coords[1]
        records.append(props)

    df = pd.DataFrame(records)
    df = df.dropna()
    print(f"✅ Sampled {len(df)} pixel records with {len(df.columns)} features.")
    return df


def export_composite_thumbnail(
    composite: ee.Image,
    geometry: ee.Geometry,
    output_path: str = "outputs/sentinel2_preview.png"
) -> str:
    """Export a quick RGB thumbnail for visualization."""
    viz_params = {
        "bands": ["B4", "B3", "B2"],
        "min": 0,
        "max": 0.3,
        "dimensions": 800,
        "region": geometry,
        "format": "PNG"
    }
    url = composite.getThumbURL(viz_params)
    
    import requests
    from PIL import Image
    from io import BytesIO
    
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save(output_path)
    print(f"🖼️  Thumbnail saved to {output_path}")
    return output_path


if __name__ == "__main__":
    initialize_gee()
    geometry = get_study_area("Wardha")
    composite = get_sentinel2_collection(geometry)
    soc = get_soilgrids_soc(geometry)
    df = sample_pixels(composite, soc, geometry, n_samples=500)
    df.to_csv("outputs/sampled_pixels.csv", index=False)
    print(df.head())
