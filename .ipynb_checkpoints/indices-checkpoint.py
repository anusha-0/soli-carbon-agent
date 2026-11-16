"""
indices.py
----------
Computes key spectral indices from Sentinel-2 bands
for land practice detection and SOC estimation.

Indices used:
- NDVI  : Vegetation health (NIR - Red) / (NIR + Red)
- BSI   : Bare Soil Index — detects exposed soil
- NDTI  : Normalized Difference Tillage Index — detects tillage
- NDWI  : Water content in vegetation
- EVI   : Enhanced Vegetation Index (less saturation than NDVI)
- CRC   : Crop Residue Cover — key for carbon estimation
"""

import pandas as pd
import numpy as np
import ee


# ─────────────────────────────────────────────
# GEE-based index computation (for map display)
# ─────────────────────────────────────────────

def compute_indices_gee(composite: ee.Image) -> ee.Image:
    """
    Adds all spectral indices as new bands to the composite.
    Used for map visualization in GEE/geemap.
    """
    B2  = composite.select("B2")   # Blue
    B3  = composite.select("B3")   # Green
    B4  = composite.select("B4")   # Red
    B8  = composite.select("B8")   # NIR
    B11 = composite.select("B11")  # SWIR1
    B12 = composite.select("B12")  # SWIR2

    # NDVI — Normalized Difference Vegetation Index
    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")

    # BSI — Bare Soil Index: ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
    bsi = (
        B11.add(B4).subtract(B8.add(B2))
        .divide(B11.add(B4).add(B8).add(B2))
        .rename("BSI")
    )

    # NDTI — Normalized Difference Tillage Index: (SWIR1 - SWIR2) / (SWIR1 + SWIR2)
    ndti = composite.normalizedDifference(["B11", "B12"]).rename("NDTI")

    # NDWI — Normalized Difference Water Index: (Green - NIR) / (Green + NIR)
    ndwi = composite.normalizedDifference(["B3", "B8"]).rename("NDWI")

    # EVI — Enhanced Vegetation Index
    evi = composite.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {"NIR": B8, "RED": B4, "BLUE": B2}
    ).rename("EVI")

    # CRC — Crop Residue Cover: (SWIR1 - Green) / (SWIR1 + Green)
    crc = composite.normalizedDifference(["B11", "B3"]).rename("CRC")

    return composite.addBands([ndvi, bsi, ndti, ndwi, evi, crc])


# ─────────────────────────────────────────────
# DataFrame-based index computation (for ML)
# ─────────────────────────────────────────────

def compute_indices_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes spectral indices from a sampled pixel DataFrame.
    Input columns expected: B2, B3, B4, B8, B11, B12
    Returns DataFrame with added index columns.
    """
    df = df.copy()

    eps = 1e-10  # avoid division by zero

    # NDVI
    df["NDVI"] = (df["B8"] - df["B4"]) / (df["B8"] + df["B4"] + eps)

    # BSI
    df["BSI"] = (
        (df["B11"] + df["B4"]) - (df["B8"] + df["B2"])
    ) / (
        (df["B11"] + df["B4"]) + (df["B8"] + df["B2"]) + eps
    )

    # NDTI
    df["NDTI"] = (df["B11"] - df["B12"]) / (df["B11"] + df["B12"] + eps)

    # NDWI
    df["NDWI"] = (df["B3"] - df["B8"]) / (df["B3"] + df["B8"] + eps)

    # EVI
    df["EVI"] = 2.5 * (
        (df["B8"] - df["B4"]) /
        (df["B8"] + 6 * df["B4"] - 7.5 * df["B2"] + 1 + eps)
    )

    # CRC
    df["CRC"] = (df["B11"] - df["B3"]) / (df["B11"] + df["B3"] + eps)

    print(f"✅ Computed 6 spectral indices. DataFrame shape: {df.shape}")
    return df


def classify_land_practice(row: pd.Series) -> str:
    """
    Rule-based land practice classifier using spectral indices.
    Used as labels for ML training + agent reasoning explanation.
    
    Classes:
    - 'No-Till'       : High crop residue, moderate NDVI
    - 'Cover Crop'    : High NDVI, low BSI
    - 'Conventional'  : High BSI (bare soil), low NDVI
    - 'Fallow'        : Low everything
    - 'Irrigated'     : High NDWI + NDVI
    """
    ndvi = row.get("NDVI", 0)
    bsi  = row.get("BSI", 0)
    ndwi = row.get("NDWI", 0)
    crc  = row.get("CRC", 0)
    ndti = row.get("NDTI", 0)

    if ndwi > 0.1 and ndvi > 0.4:
        return "Irrigated Crop"
    elif crc > 0.15 and ndvi > 0.2 and ndvi < 0.6:
        return "No-Till / Residue"
    elif ndvi > 0.5 and bsi < 0.0:
        return "Cover Crop / Dense Vegetation"
    elif bsi > 0.05 and ndvi < 0.2:
        return "Conventional Tilled / Bare Soil"
    elif ndvi < 0.15 and bsi < 0.0:
        return "Fallow Land"
    else:
        return "Mixed / Transitional"


def add_practice_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Apply practice classification to all rows."""
    df["LandPractice"] = df.apply(classify_land_practice, axis=1)
    
    practice_counts = df["LandPractice"].value_counts()
    print("\n🌾 Land Practice Distribution:")
    print(practice_counts.to_string())
    return df


def get_carbon_potential(practice: str) -> dict:
    """
    Returns carbon sequestration potential and recommendation
    for each land practice.
    Based on IPCC Tier 1 emission factors and literature values.
    """
    carbon_map = {
        "No-Till / Residue": {
            "soc_change_rate": +0.3,   # tC/ha/year
            "potential": "High",
            "recommendation": "Maintain no-till practice. Add legume cover crops to further boost SOC.",
            "credit_potential_per_ha": 2.5
        },
        "Cover Crop / Dense Vegetation": {
            "soc_change_rate": +0.5,
            "potential": "Very High",
            "recommendation": "Excellent practice. Diversify cover crop species for maximum carbon input.",
            "credit_potential_per_ha": 4.0
        },
        "Conventional Tilled / Bare Soil": {
            "soc_change_rate": -0.2,
            "potential": "Low (Carbon Loss)",
            "recommendation": "URGENT: Transition to no-till or cover cropping. Current practice is losing carbon.",
            "credit_potential_per_ha": 0.0
        },
        "Fallow Land": {
            "soc_change_rate": 0.0,
            "potential": "Moderate",
            "recommendation": "Introduce green manure or legume cover crops during fallow period.",
            "credit_potential_per_ha": 1.0
        },
        "Irrigated Crop": {
            "soc_change_rate": +0.1,
            "potential": "Moderate",
            "recommendation": "Integrate organic matter inputs. Avoid over-tillage between crops.",
            "credit_potential_per_ha": 1.5
        },
        "Mixed / Transitional": {
            "soc_change_rate": +0.05,
            "potential": "Variable",
            "recommendation": "Conduct field-level soil testing for precise recommendations.",
            "credit_potential_per_ha": 0.8
        }
    }
    return carbon_map.get(practice, carbon_map["Mixed / Transitional"])


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    n = 100
    test_df = pd.DataFrame({
        "B2":  np.random.uniform(0.02, 0.15, n),
        "B3":  np.random.uniform(0.03, 0.20, n),
        "B4":  np.random.uniform(0.02, 0.25, n),
        "B8":  np.random.uniform(0.10, 0.60, n),
        "B11": np.random.uniform(0.05, 0.40, n),
        "B12": np.random.uniform(0.03, 0.30, n),
        "SOC": np.random.uniform(3.0, 25.0, n)
    })

    df = compute_indices_df(test_df)
    df = add_practice_labels(df)
    print("\nSample rows:")
    print(df[["NDVI", "BSI", "NDTI", "LandPractice", "SOC"]].head(10))
