# CP-Agent

**CP-Agent: Context-Aware Multimodal Reasoning for Cellular Morphological Profiling under Chemical Perturbations**

> 🎉 **Accepted by ICLR 2026**

---

## Overview

CP-Agent is a modular multimodal agent system for analyzing drug-induced cellular morphology changes captured by **Cell Painting** assays. It integrates high-content imaging, structured metadata, and agent-driven analysis to generate mechanism-aware reports.

At its core, CP-Agent includes:
- 🔬 **CP-CLIP**: A contrastive learning module aligning Cell Painting images with structured experiment metadata.
- 🤖 **CP-Agent**: A reasoning pipeline that synthesizes visual features and context into interpretable biological reports.

![CP-Agent Overview](figures/plot1.png)

---

## Key Features

- **Context-aware representation** of experimental metadata (e.g., cell line, dose, time)
- **Multimodal alignment** of image and molecular descriptors via CP-CLIP
- **Mechanistic reasoning** using LLM agents and CellProfiler features
- **Zero-shot generalization** to unseen compounds

---

## Model Weights

The pre-trained CP-CLIP model weights are available on Hugging Face:

🤗 **[letitiaaa/CP-agent](https://huggingface.co/letitiaaa/CP-agent)**

| File | Description | Size |
|------|-------------|------|
| `cpclip_model.pt` | CP-CLIP model weight | 1.82 GB |
| `openclip_model.pt` | OpenCLIP baseline weight | 1.81 GB |
| `model_segmentor.pt` | fintuned-VISTA segmentation weight | 360 MB |

### Download via Python

```python
from huggingface_hub import hf_hub_download

# Download CP-CLIP model example
hf_hub_download(repo_id="letitiaaa/CP-agent", filename="cpclip_model.pt", local_dir="./weights")
```
---



## Usage Guide

You can run **CP-Agent** either as a script or step-by-step in a notebook:
To set up the environment and install all dependencies, run:

```bash
pip install -r requirements.txt
```
This will set up the full environment needed to run CP-Agent, including image processing, deep learning, and LLM integration components.

> ⚠️ Note: Some packages will be installed directly from GitHub (e.g., `segment_anything`). Make sure `git` is available in your environment.
All configurable parameters (e.g., input image paths, output locations, analysis options) can be customized in: `config.py`

### Run 

```bash
# Full end-to-end pipeline and generate a full report:
python cpagent_reportGen.py

# Or interactive notebook
jupyter notebook cp-agent_demo.ipynb
```
Sample Cell Painting images (control vs perturbation pairs) are provided in `example_images/` for testing.


