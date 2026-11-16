"""
agent.py
--------
The core agentic reasoning loop using LangGraph.

Agent Graph:
  START
    ↓
  [PerceptionNode]   — reads pixel data, summarizes field conditions
    ↓
  [AnalysisNode]     — interprets SOC scores + land practices
    ↓
  [DiagnosisNode]    — identifies problem fields, carbon loss risk
    ↓
  [RecommendationNode] — generates actionable interventions per field
    ↓
  [ReportNode]       — compiles final structured report
    ↓
  END

Each node reasons step-by-step, producing an auditable trace
that is displayed on the LinkedIn demo dashboard.
"""

import os
import json
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# ─── LangGraph imports ───────────────────────
from langgraph.graph import StateGraph, END

# ─── LangChain LLM ───────────────────────────
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


# ─────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    """Shared state passed between all agent nodes."""
    field_data: List[Dict]          # Input: list of field pixel summaries
    perception_summary: str         # Output of PerceptionNode
    analysis_summary: str           # Output of AnalysisNode
    diagnosis: str                  # Output of DiagnosisNode
    recommendations: List[Dict]     # Output of RecommendationNode
    final_report: str               # Output of ReportNode
    reasoning_trace: List[str]      # Full agent reasoning log for display
    region_name: str                # e.g., "Wardha, Maharashtra"


# ─────────────────────────────────────────────
# LLM Setup
# ─────────────────────────────────────────────

def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    """Returns configured LLM. Uses GPT-4o by default."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in .env file.\n"
            "Add: OPENAI_API_KEY=your_key_here to a .env file."
        )
    return ChatOpenAI(
        model="gpt-4o",
        temperature=temperature,
        api_key=api_key
    )


# ─────────────────────────────────────────────
# Agent Nodes
# ─────────────────────────────────────────────

def perception_node(state: AgentState) -> AgentState:
    """
    NODE 1: PERCEPTION
    Reads raw field data and produces a structured natural language
    summary of what the satellite imagery shows.
    """
    print("\n🔍 [AGENT] Perception Node running...")

    llm = get_llm()
    field_data = state["field_data"]
    region = state["region_name"]

    # Build a structured summary of the field data for the LLM
    field_summary_text = json.dumps(field_data[:15], indent=2)  # Top 15 fields

    messages = [
        SystemMessage(content="""You are a GIS Remote Sensing expert analyzing 
        satellite imagery of agricultural fields in India. 
        Be precise, technical, and structured in your analysis."""),
        HumanMessage(content=f"""
        I have sampled {len(field_data)} field locations in {region} from 
        Sentinel-2 satellite imagery (Rabi season 2023-24).
        
        Here is a sample of the data including spectral indices and 
        predicted Soil Organic Carbon (SOC) values:
        
        {field_summary_text}
        
        Please provide a PERCEPTION SUMMARY:
        1. What are the dominant spectral characteristics you observe?
        2. What season/crop cycle does this imagery likely represent?
        3. What is the overall vegetation density and soil exposure pattern?
        4. What is the range and distribution of SOC values?
        
        Keep your response focused, structured, and under 200 words.
        """)
    ]

    response = llm.invoke(messages)
    perception_summary = response.content

    state["perception_summary"] = perception_summary
    state["reasoning_trace"].append(
        f"**🔍 PERCEPTION:**\n{perception_summary}"
    )

    print("   ✅ Perception complete.")
    return state


def analysis_node(state: AgentState) -> AgentState:
    """
    NODE 2: ANALYSIS
    Deep analysis of land practices and carbon status.
    """
    print("📊 [AGENT] Analysis Node running...")

    llm = get_llm()
    field_data = state["field_data"]
    region = state["region_name"]

    # Count practice distribution
    practice_counts = {}
    soc_values = []
    for f in field_data:
        p = f.get("PredictedPractice", "Unknown")
        practice_counts[p] = practice_counts.get(p, 0) + 1
        soc_values.append(f.get("SOC_predicted", 0))

    avg_soc = round(sum(soc_values) / len(soc_values), 2) if soc_values else 0
    min_soc = round(min(soc_values), 2) if soc_values else 0
    max_soc = round(max(soc_values), 2) if soc_values else 0

    messages = [
        SystemMessage(content="""You are an agricultural carbon scientist 
        specializing in soil health in South Asian farming systems."""),
        HumanMessage(content=f"""
        PERCEPTION CONTEXT: {state['perception_summary']}
        
        ANALYSIS DATA for {region}:
        - Total fields analyzed: {len(field_data)}
        - Land Practice Distribution: {json.dumps(practice_counts, indent=2)}
        - SOC Statistics: Mean={avg_soc} g/kg, Min={min_soc}, Max={max_soc}
        
        Please provide a detailed ANALYSIS:
        1. Which land practices dominate and what does this imply for soil health?
        2. How does the average SOC ({avg_soc} g/kg) compare to healthy soil benchmarks (>20 g/kg)?
        3. What percentage of fields are losing vs gaining carbon?
        4. What are the primary drivers of carbon status in this region?
        5. What is the estimated total carbon stock per hectare?
        
        Be specific and quantitative. Under 250 words.
        """)
    ]

    response = llm.invoke(messages)
    analysis_summary = response.content

    state["analysis_summary"] = analysis_summary
    state["reasoning_trace"].append(
        f"**📊 ANALYSIS:**\n{analysis_summary}"
    )

    print("   ✅ Analysis complete.")
    return state


def diagnosis_node(state: AgentState) -> AgentState:
    """
    NODE 3: DIAGNOSIS
    Identifies the most critical problem fields and root causes.
    """
    print("🩺 [AGENT] Diagnosis Node running...")

    llm = get_llm()
    field_data = state["field_data"]

    # Find critical fields (lowest SOC + worst practice)
    critical_fields = [
        f for f in field_data
        if f.get("SOC_predicted", 99) < 8
        or f.get("PredictedPractice") == "Conventional Tilled / Bare Soil"
    ][:5]

    messages = [
        SystemMessage(content="""You are a precision agriculture diagnostic expert.
        Identify root causes and urgency levels for soil degradation."""),
        HumanMessage(content=f"""
        CONTEXT: {state['analysis_summary']}
        
        CRITICAL FIELDS IDENTIFIED (lowest SOC or worst practices):
        {json.dumps(critical_fields, indent=2)}
        
        Please provide a DIAGNOSIS:
        1. What is the primary diagnosis for soil carbon depletion in this region?
        2. Which fields need URGENT intervention (within 1 season)?
        3. What are the root causes — tillage intensity, lack of organic matter, 
           crop residue burning, or water stress?
        4. What is the estimated carbon loss per year if no action is taken?
        5. What is the confidence level of this diagnosis based on the satellite data?
        
        Use clinical/diagnostic language. Under 200 words.
        """)
    ]

    response = llm.invoke(messages)
    diagnosis = response.content

    state["diagnosis"] = diagnosis
    state["reasoning_trace"].append(
        f"**🩺 DIAGNOSIS:**\n{diagnosis}"
    )

    print("   ✅ Diagnosis complete.")
    return state


def recommendation_node(state: AgentState) -> AgentState:
    """
    NODE 4: RECOMMENDATION
    Generates specific, actionable interventions for each field cluster.
    """
    print("💡 [AGENT] Recommendation Node running...")

    llm = get_llm()
    region = state["region_name"]

    messages = [
        SystemMessage(content="""You are a regenerative agriculture advisor
        specializing in carbon farming in India. Give practical, 
        locally relevant recommendations that farmers can actually implement."""),
        HumanMessage(content=f"""
        FULL CONTEXT:
        Perception: {state['perception_summary']}
        Analysis: {state['analysis_summary']}
        Diagnosis: {state['diagnosis']}
        Region: {region}
        
        Generate STRUCTURED RECOMMENDATIONS as a JSON array with this exact format:
        [
          {{
            "priority": "HIGH/MEDIUM/LOW",
            "practice": "name of practice",
            "description": "what to do exactly",
            "carbon_benefit": "estimated tC/ha/year gain",
            "implementation_season": "Kharif/Rabi/Both",
            "estimated_cost_inr_per_ha": 1000,
            "credit_potential_inr_per_ha": 2000,
            "timeline": "immediate/1-season/2-seasons"
          }}
        ]
        
        Generate 5 recommendations ordered by priority and impact.
        Return ONLY the JSON array, no extra text.
        """)
    ]

    response = llm.invoke(messages)

    # Parse recommendations
    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        recommendations = json.loads(raw.strip())
    except Exception as e:
        print(f"   ⚠️ JSON parse warning: {e}. Using fallback.")
        recommendations = [{
            "priority": "HIGH",
            "practice": "Transition to No-Till Farming",
            "description": "Avoid mechanical tillage. Use zero-till seed drills for wheat/sorghum.",
            "carbon_benefit": "0.3-0.5 tC/ha/year",
            "implementation_season": "Rabi",
            "estimated_cost_inr_per_ha": 2500,
            "credit_potential_inr_per_ha": 3000,
            "timeline": "immediate"
        }]

    state["recommendations"] = recommendations
    state["reasoning_trace"].append(
        f"**💡 RECOMMENDATIONS:**\n{json.dumps(recommendations, indent=2)}"
    )

    print(f"   ✅ Generated {len(recommendations)} recommendations.")
    return state


def report_node(state: AgentState) -> AgentState:
    """
    NODE 5: REPORT
    Compiles everything into a final executive summary report.
    """
    print("📋 [AGENT] Report Node running...")

    llm = get_llm()
    region = state["region_name"]

    rec_summary = "\n".join([
        f"- [{r['priority']}] {r['practice']}: {r['carbon_benefit']} carbon gain"
        for r in state["recommendations"]
    ])

    messages = [
        SystemMessage(content="""You are writing an executive briefing report 
        for agricultural policymakers and carbon market investors."""),
        HumanMessage(content=f"""
        Create a concise EXECUTIVE REPORT for {region} soil carbon assessment.
        
        Include:
        1. Executive Summary (3 sentences)
        2. Key Findings (3 bullets)
        3. Carbon Credit Opportunity (quantified estimate)
        4. Top Recommendations Summary: {rec_summary}
        5. Confidence & Limitations
        6. Recommended Next Steps
        
        Tone: Professional, data-driven, actionable.
        Format with clear headers. Under 400 words total.
        """)
    ]

    response = llm.invoke(messages)
    final_report = response.content

    state["final_report"] = final_report
    state["reasoning_trace"].append(
        f"**📋 FINAL REPORT:**\n{final_report}"
    )

    print("   ✅ Report generated.")
    return state


# ─────────────────────────────────────────────
# Build the LangGraph Agent
# ─────────────────────────────────────────────

def build_agent() -> StateGraph:
    """Constructs and compiles the LangGraph agent workflow."""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("perception",     perception_node)
    workflow.add_node("analysis",       analysis_node)
    workflow.add_node("diagnosis",      diagnosis_node)
    workflow.add_node("recommendation", recommendation_node)
    workflow.add_node("report",         report_node)

    # Define linear flow
    workflow.set_entry_point("perception")
    workflow.add_edge("perception",     "analysis")
    workflow.add_edge("analysis",       "diagnosis")
    workflow.add_edge("diagnosis",      "recommendation")
    workflow.add_edge("recommendation", "report")
    workflow.add_edge("report",         END)

    return workflow.compile()


def run_agent(field_data: list, region_name: str = "Wardha, Maharashtra") -> AgentState:
    """
    Main entry point to run the full agent pipeline.
    
    Args:
        field_data: List of dicts with field spectral + SOC data
        region_name: Human-readable region name
        
    Returns:
        Final AgentState with all outputs and reasoning trace
    """
    app = build_agent()

    initial_state: AgentState = {
        "field_data": field_data,
        "perception_summary": "",
        "analysis_summary": "",
        "diagnosis": "",
        "recommendations": [],
        "final_report": "",
        "reasoning_trace": [],
        "region_name": region_name
    }

    print(f"\n{'═'*55}")
    print(f"  🤖 SOIL CARBON AGENT — {region_name}")
    print(f"  Analyzing {len(field_data)} field locations...")
    print(f"{'═'*55}")

    result = app.invoke(initial_state)

    print(f"\n{'═'*55}")
    print("  ✅ AGENT COMPLETE — All nodes executed.")
    print(f"{'═'*55}\n")

    return result


# ─────────────────────────────────────────────
# Mock Agent (no API key needed for demo)
# ─────────────────────────────────────────────

def run_mock_agent(field_data: list, region_name: str = "Wardha, Maharashtra") -> AgentState:
    """
    Mock agent that returns realistic outputs WITHOUT needing an OpenAI key.
    Perfect for portfolio demo and LinkedIn post.
    """
    import time

    n_fields = len(field_data)
    soc_values = [f.get("SOC_predicted", 12) for f in field_data]
    avg_soc = round(sum(soc_values) / len(soc_values), 1) if soc_values else 12.0

    practice_counts = {}
    for f in field_data:
        p = f.get("PredictedPractice", "Unknown")
        practice_counts[p] = practice_counts.get(p, 0) + 1

    dominant_practice = max(practice_counts, key=practice_counts.get) if practice_counts else "Mixed"
    pct_degraded = round(
        practice_counts.get("Conventional Tilled / Bare Soil", 0) / max(n_fields, 1) * 100, 1
    )

    reasoning_trace = []

    print(f"\n{'═'*55}")
    print(f"  🤖 SOIL CARBON AGENT (Demo Mode) — {region_name}")
    print(f"  Analyzing {n_fields} field locations...")
    print(f"{'═'*55}")

    time.sleep(0.3)
    print("🔍 [AGENT] Perception Node running...")
    perception = f"""Sentinel-2 imagery analysis for {region_name} (Rabi season 2023-24) reveals 
a heterogeneous agricultural landscape with {n_fields} sampled field locations. Dominant spectral 
signatures indicate a mix of crop residue-covered and bare-soil exposed fields. NDVI values range 
from 0.12 to 0.61, suggesting active cropping alongside significant post-harvest bare exposure. 
SWIR bands (B11/B12) show high reflectance in approximately {pct_degraded}% of pixels, indicating 
tillage activity. SOC distribution is skewed towards lower values (mean: {avg_soc} g/kg), 
well below the 20 g/kg threshold for healthy agricultural soils."""
    reasoning_trace.append(f"**🔍 PERCEPTION:**\n{perception}")
    print("   ✅ Perception complete.")

    time.sleep(0.3)
    print("📊 [AGENT] Analysis Node running...")
    analysis = f"""Land practice analysis reveals '{dominant_practice}' as the dominant management 
system ({practice_counts.get(dominant_practice, 0)} of {n_fields} fields = 
{round(practice_counts.get(dominant_practice,0)/max(n_fields,1)*100,1)}%). 
The average SOC of {avg_soc} g/kg is critically below the 20 g/kg benchmark for productive soils, 
indicating significant historical carbon depletion from intensive tillage. 
{pct_degraded}% of fields show active carbon loss patterns. Under current management, 
the region is estimated to be losing 0.15-0.25 tC/ha/year. Only {100-pct_degraded}% of fields 
show neutral or positive carbon accumulation. At current SOC levels, soil water-holding 
capacity and crop yields are likely 15-25% below their potential."""
    reasoning_trace.append(f"**📊 ANALYSIS:**\n{analysis}")
    print("   ✅ Analysis complete.")

    time.sleep(0.3)
    print("🩺 [AGENT] Diagnosis Node running...")
    diagnosis = f"""PRIMARY DIAGNOSIS: Moderate-to-severe soil organic carbon depletion driven by 
chronic conventional tillage and residue removal in {region_name}. 
ROOT CAUSES: (1) Deep plowing disrupting soil aggregate structure and exposing subsoil; 
(2) Crop residue burning post-harvest eliminating organic matter inputs; 
(3) Insufficient fallow management with no cover crops. 
URGENCY: HIGH — {pct_degraded}% of fields require immediate intervention. 
PROJECTED IMPACT (no action): At current loss rate of ~0.20 tC/ha/year, 
fields will cross the 5 g/kg critical threshold within 8-12 years, 
risking permanent productivity loss. Confidence: 78% based on spectral indices + SOC model."""
    reasoning_trace.append(f"**🩺 DIAGNOSIS:**\n{diagnosis}")
    print("   ✅ Diagnosis complete.")

    time.sleep(0.3)
    print("💡 [AGENT] Recommendation Node running...")
    recommendations = [
        {
            "priority": "HIGH",
            "practice": "Zero-Till / No-Till Transition",
            "description": "Replace conventional plowing with zero-till seed drills for wheat and chickpea. Maintain crop residue as mulch cover.",
            "carbon_benefit": "0.30-0.45 tC/ha/year",
            "implementation_season": "Rabi (immediate)",
            "estimated_cost_inr_per_ha": 2500,
            "credit_potential_inr_per_ha": 3200,
            "timeline": "immediate"
        },
        {
            "priority": "HIGH",
            "practice": "Cover Cropping with Legumes",
            "description": "Plant Sesbania or Dhaincha as green manure during fallow period. Incorporate before flowering to maximize N and C inputs.",
            "carbon_benefit": "0.40-0.60 tC/ha/year",
            "implementation_season": "Kharif fallow",
            "estimated_cost_inr_per_ha": 1800,
            "credit_potential_inr_per_ha": 4500,
            "timeline": "1 season"
        },
        {
            "priority": "MEDIUM",
            "practice": "Residue Retention (Stop Burning)",
            "description": "Retain 60-70% of crop residue on fields. Use happy seeder for in-situ incorporation. Critical for Wardha cotton fields.",
            "carbon_benefit": "0.15-0.25 tC/ha/year",
            "implementation_season": "Post-harvest (Kharif)",
            "estimated_cost_inr_per_ha": 800,
            "credit_potential_inr_per_ha": 1800,
            "timeline": "immediate"
        },
        {
            "priority": "MEDIUM",
            "practice": "Compost + Biochar Application",
            "description": "Apply 2-3 tonnes/ha compost or 0.5 t/ha biochar annually. Combine with vermicomposting of farm waste.",
            "carbon_benefit": "0.20-0.35 tC/ha/year",
            "implementation_season": "Pre-Kharif",
            "estimated_cost_inr_per_ha": 3500,
            "credit_potential_inr_per_ha": 2800,
            "timeline": "1 season"
        },
        {
            "priority": "LOW",
            "practice": "Agroforestry Integration",
            "description": "Plant Subabul or Moringa on field borders (5-8 trees/ha). Long-term SOC boost + additional income stream.",
            "carbon_benefit": "0.50-1.0 tC/ha/year",
            "implementation_season": "Kharif onset",
            "estimated_cost_inr_per_ha": 4000,
            "credit_potential_inr_per_ha": 8000,
            "timeline": "2 seasons"
        }
    ]
    reasoning_trace.append(f"**💡 RECOMMENDATIONS:**\n{json.dumps(recommendations, indent=2)}")
    print(f"   ✅ Generated {len(recommendations)} recommendations.")

    time.sleep(0.3)
    print("📋 [AGENT] Report Node running...")
    total_credit_potential = sum(r["credit_potential_inr_per_ha"] for r in recommendations[:3])
    final_report = f"""
## 🌱 SOIL CARBON ASSESSMENT REPORT
### {region_name} | Rabi Season 2023-24

---

### Executive Summary
Satellite-based analysis of {n_fields} agricultural fields in {region_name} reveals 
significant soil carbon depletion, with mean SOC at {avg_soc} g/kg — 
**{round((20 - avg_soc)/20*100)}% below the 20 g/kg healthy soil benchmark.** 
Approximately {pct_degraded}% of fields are actively losing carbon under conventional 
tillage practices, representing a major intervention opportunity for both yield recovery 
and carbon credit generation.

---

### Key Findings
- 📍 **{n_fields} fields analyzed** across {region_name} using Sentinel-2 10m imagery
- ⚠️ **Mean SOC: {avg_soc} g/kg** — critically below healthy threshold of 20 g/kg
- 🔴 **{pct_degraded}% fields** under carbon-depleting conventional tillage
- 💰 **Estimated carbon credit potential: ₹{total_credit_potential:,}/ha/year** (top 3 practices)
- 📈 **Projected SOC recovery: +0.35 tC/ha/year** with recommended interventions

---

### Carbon Credit Opportunity
At current degradation rates, a 1,000 ha intervention zone could generate:
- **Carbon sequestered: 350 tCO₂e/year**
- **Credit value at ₹500/tCO₂e: ₹1,75,000/year**
- **5-year cumulative value: ₹8,75,000** (conservative estimate)

---

### Confidence & Limitations
Model R² = 0.78 | RMSE = 2.1 g/kg | Validation: LUCAS soil database
Limitations: 20m pixel resolution; cloud cover <20% filter applied; 
seasonal variability not yet captured.

---

### Recommended Next Steps
1. Deploy field validation team for ground-truth SOC sampling (50 sites)
2. Enroll willing farmers in no-till pilot (50 ha trial)
3. Register with Verra or Gold Standard for carbon credit verification
4. Re-run satellite assessment in 6 months to track change
"""
    reasoning_trace.append(f"**📋 FINAL REPORT:**\n{final_report}")
    print("   ✅ Report generated.")

    print(f"\n{'═'*55}")
    print("  ✅ AGENT COMPLETE — All 5 nodes executed.")
    print(f"{'═'*55}\n")

    return {
        "field_data": field_data,
        "perception_summary": perception,
        "analysis_summary": analysis,
        "diagnosis": diagnosis,
        "recommendations": recommendations,
        "final_report": final_report,
        "reasoning_trace": reasoning_trace,
        "region_name": region_name
    }


if __name__ == "__main__":
    # Quick test with synthetic field data
    import sys
    sys.path.append(".")
    from soc_model import generate_synthetic_training_data
    from indices import compute_indices_df

    df = generate_synthetic_training_data(100)
    df = compute_indices_df(df)
    field_data = df.to_dict("records")[:50]

    result = run_mock_agent(field_data, "Wardha, Maharashtra")
    print("\n📋 FINAL REPORT:")
    print(result["final_report"])
