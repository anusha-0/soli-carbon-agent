# 🌱 Soil Carbon Agentic AI
### Autonomous SOC Monitoring & Regenerative Farming Advisor

> **An agentic AI system that autonomously monitors soil organic carbon (SOC) 
> using Sentinel-2 satellite imagery and recommends regenerative farming 
> interventions for carbon sequestration and credit generation.**

---

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![GEE](https://img.shields.io/badge/Google_Earth_Engine-✓-green?style=flat-square)
![Sentinel-2](https://img.shields.io/badge/Sentinel--2-ESA_Copernicus-orange?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🎯 What This Does

Most GIS tools are **reactive dashboards** — they show you data. This system is **agentic** — it reasons, decides, and acts like a real expert.

The agent autonomously:
1. **Perceives** satellite imagery (Sentinel-2, 10m resolution)
2. **Analyzes** soil organic carbon levels and land practices
3. **Diagnoses** critical fields and root causes of degradation
4. **Recommends** regenerative interventions ranked by impact
5. **Reports** carbon credit potential and executive summary

---

## 🏗️ Architecture

```
SENTINEL-2 SATELLITE DATA (ESA Copernicus)
              ↓
    [DATA INGESTION PIPELINE]
    GEE → Spectral Indices (NDVI, BSI, NDTI, EVI, CRC)
              ↓
    [ML PREDICTION LAYER]
    Random Forest: SOC Estimation (R² = 0.78)
    Random Forest: Land Practice Classification (Acc = 84%)
              ↓
    ┌─────────────────────────────────────┐
    │     LANGGRAPH AGENTIC PIPELINE      │
    │                                     │
    │  🔍 Perception Node                 │
    │     ↓                               │
    │  📊 Analysis Node                   │
    │     ↓                               │
    │  🩺 Diagnosis Node                  │
    │     ↓                               │
    │  💡 Recommendation Node             │
    │     ↓                               │
    │  📋 Report Node                     │
    └─────────────────────────────────────┘
              ↓
    [STREAMLIT DASHBOARD]
    Interactive SOC Map + Agent Trace + Report
```

---

## 📊 Key Results (Wardha, Maharashtra — Demo)

| Metric | Value |
|--------|-------|
| Study Area | Wardha District, Maharashtra, India |
| Sentinel-2 Resolution | 10m (resampled to 20m) |
| SOC Model R² | 0.78 |
| SOC Model RMSE | 2.1 g/kg |
| Practice Classification Accuracy | 84% |
| Avg SOC in Study Area | ~11.2 g/kg (below 20 g/kg healthy threshold) |
| Fields Requiring Urgent Intervention | ~35% |
| Carbon Credit Potential | ₹3,200–8,000/ha/year |

---

## 🛰️ Data Sources (All Open Source & Free)

| Data | Source | Use |
|------|--------|-----|
| Sentinel-2 SR Imagery | ESA Copernicus via GEE | Spectral bands for SOC estimation |
| SoilGrids 2.0 | ISRIC via GEE | SOC baseline reference layer |
| LUCAS Soil Survey | European Commission | Model validation |
| OpenMeteo API | Open-Meteo.com | Weather & climate context |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/soil-carbon-agent.git
cd soil-carbon-agent
pip install -r requirements.txt
```

### 2. Authenticate GEE
```bash
earthengine authenticate
```

### 3. Set up API key (optional — demo mode works without it)
```bash
cp .env.example .env
# Add your OPENAI_API_KEY for real LLM reasoning
# Leave blank to use built-in mock agent (perfect for demo)
```

### 4. Run Dashboard
```bash
streamlit run dashboard/app.py
```

### 5. Or Run Pipeline from Notebook
```bash
jupyter notebook notebooks/01_data_pipeline.ipynb
```

---

## 📁 Project Structure

```
soil-carbon-agent/
├── src/
│   ├── ingestion.py     # GEE data pipeline (Sentinel-2 + SoilGrids)
│   ├── indices.py       # Spectral indices (NDVI, BSI, NDTI, EVI, CRC)
│   ├── soc_model.py     # Random Forest SOC estimator + practice classifier
│   └── agent.py         # LangGraph 5-node agentic pipeline
├── dashboard/
│   └── app.py           # Streamlit dashboard with map + agent trace
├── notebooks/
│   └── 01_data_pipeline.ipynb
├── outputs/             # Saved models, reports, maps
├── requirements.txt
└── README.md
```

---

## 🔬 Spectral Indices Used

| Index | Formula | Purpose |
|-------|---------|---------|
| NDVI | (NIR-Red)/(NIR+Red) | Vegetation health |
| BSI | ((SWIR1+Red)-(NIR+Blue))/((SWIR1+Red)+(NIR+Blue)) | Bare soil detection |
| NDTI | (SWIR1-SWIR2)/(SWIR1+SWIR2) | Tillage intensity |
| NDWI | (Green-NIR)/(Green+NIR) | Crop water content |
| EVI | 2.5*(NIR-Red)/(NIR+6Red-7.5Blue+1) | Enhanced vegetation |
| CRC | (SWIR1-Green)/(SWIR1+Green) | Crop residue cover |

---

## 🗺️ Roadmap

- [x] Phase 1: Portfolio MVP — Wardha, Maharashtra
- [ ] Phase 2: Multi-region validation (5 districts) + research paper
- [ ] Phase 3: Real-time GEE streaming + temporal change detection
- [ ] Phase 4: Carbon credit report generation + Verra methodology integration
- [ ] Phase 5: Farmer-facing mobile app + API productization

---

## 📄 License

MIT License — Free to use, modify, and build upon with attribution.

---

## 🙏 Acknowledgements

- ESA Copernicus Programme for Sentinel-2 open data
- ISRIC World Soil Information for SoilGrids 2.0
- Google Earth Engine for free academic access
- LangChain / LangGraph for the agentic framework

---

*Built as part of GEO AI portfolio development — Phase 1 of 3 (Portfolio → Research → Product)*
