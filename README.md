# CP-Agent

**CP-Agent: Context-Aware Multimodal Reasoning for Cellular Morphological Profiling under Chemical Perturbations**

> 🔍 A modular, interpretable, and agentic framework for analyzing Cell Painting perturbation data using multimodal large language models (MLLMs) and image-based reasoning.

---

## 📌 Overview

CP-Agent is a modular multimodal agent system that supports human-interpretable reasoning for drug-induced cellular morphology changes captured by **Cell Painting** assays. It integrates high-content imaging, structured metadata, and agent-driven analysis to generate mechanism-aware reports.

At its core, CP-Agent includes:
- `CP-CLIP`: A contrastive learning module aligning Cell Painting images with structured experiment metadata.
- `CP-Agent`: A reasoning pipeline that synthesizes visual features and context into interpretable biological reports.
![CP-Agent Overview](figures/plot1.png)

---

## 🧠 Key Features

- ✅ **Context-aware representation** of experimental metadata (e.g., cell line, dose, time)
- ✅ **Multimodal alignment** of image and molecular descriptors via CP-CLIP
- ✅ **Mechanistic reasoning** using LLM agents and CellProfiler features
- ✅ **Zero-shot generalization** to unseen compounds


---

## 🧬 Applications

- Mechanism-of-action (MoA) inference
- Phenotypic screening and hit triaging
- Hypothesis generation for drug discovery
- Cross-study generalization in morphological profiling

---


## 📁 Project Structure
<pre><code>

cp-agent/

├── cpclip/ # CP-CLIP model and training

├── featureExtractor/ # Feature extraction wrappers

├── reasoning_utils/ # Agent modules (FeatRank, ReportGen, etc.)

├── segmentor/ # Segmentation pipeline (e.g., VISTA-2D)

├── metadata/ # Curated experimental metadata

├── results/ # Sample outputs and reports

├── example_images/ # Example Cell Painting images

├── cp-agent_demo.ipynb # Jupyter demo notebook

├── cpagent_utils.py # Utility functions

├── cpagent_reportGen.py # Report generation logic

├── config.py # Configuration settings

└── requirements.txt # Python dependencies

</code></pre>

