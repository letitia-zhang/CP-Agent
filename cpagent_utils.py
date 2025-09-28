import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import random
import math
import re
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from segmentor.vista_scripts.cell_sam_wrapper import CellSamWrapper
from segmentor.vista_scripts.components import LogitsToLabels
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, List, Union
import io
import yaml
import matplotlib.patches as patches
from typing import Any, Dict, List, Optional
from jinja2 import Template
from scipy.stats import mannwhitneyu
from google import genai
from google.genai import types
import openai
import base64

class DrugTextImageMatcher:
    def __init__(
        self,
        open_clip_path: str,
        pretrained_ckpt_path: str,
        text_jsonl: str,
        compound_npz_path: str,
        drug_csv_paths: list,
        device: str = None
    ):
        if open_clip_path not in sys.path:
            sys.path.insert(0, open_clip_path)
        import open_clip
        self.open_clip = open_clip

        self.pretrained_ckpt_path = pretrained_ckpt_path
        self.text_jsonl = text_jsonl
        self.compound_npz_path = compound_npz_path
        self.drug_csv_paths = drug_csv_paths
        self.device = device

        self.tokenizer = open_clip.get_tokenizer(
            model_name="ViT-B-16",
            context_length=256,
            tokenizer_type="hf:gpt2",
            additional_special_tokens=['<CONC_TOKEN>', '<TIME_TOKEN>', '<COMPOUND_TOKEN>']
        )

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name="ViT-B-16",
            pretrained=self.pretrained_ckpt_path,
            precision="fp32",
            device=self.device,
            tokenizer=self.tokenizer,
            output_dict=True,
            load_weights_only=True,
            special_tokens=['<CONC_TOKEN>', '<TIME_TOKEN>', '<COMPOUND_TOKEN>'],
            use_enhanced_clip=True
        )
        self.model.eval()

    class JsonlTextOnlyProcessor:
        def __init__(self, tokenizer=None):
            self.tokenize = tokenizer

        def process(self, sample):
            text = self.tokenize([str(sample['text'])])[0] if self.tokenize else str(sample['text'])
            concentration = torch.tensor(sample["Concentration"], dtype=torch.float32)
            time = torch.tensor(sample["Time"], dtype=torch.float32)
            compound_embedding = torch.tensor(sample["compound_embedding"], dtype=torch.float32)
            return text, concentration, time, compound_embedding, str(sample['text'])

    def load_image_tensor(self, image_path: str):
        full_img = Image.open(image_path).convert("L")
        full_tensor = torch.from_numpy(np.array(full_img)).unsqueeze(0).float() / 255.
        left_tensor = full_tensor[:, :, :512]
        right_tensor = full_tensor[:, :, 512:]
        image = torch.cat([left_tensor, right_tensor], dim=0)
        return image  

    def load_compound_embeddings(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        return data['compounds'], data['embeddings']

    def find_compound_name(self, embedding, compound_names, compound_embeddings):
        embedding = embedding.reshape(1, -1)
        sims = cosine_similarity(embedding, compound_embeddings)[0]
        return compound_names[np.argmax(sims)]

    def load_druginfo(self, paths):
        return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

    def infer_actual_concentration(self, compound_name_raw, max_mass_density_norm, step_index_norm, drug_df):
        max_mass_density_norm = max_mass_density_norm.item() if hasattr(max_mass_density_norm, "item") else max_mass_density_norm
        step_index_norm = step_index_norm.item() if hasattr(step_index_norm, "item") else step_index_norm

        def normalize_name(name):
            return re.sub(r'[^a-z0-9]', '', name.lower())

        if 'norm_name' not in drug_df.columns:
            drug_df['norm_name'] = drug_df['compound'].apply(normalize_name)

        compound_info = {
            row['norm_name']: row for _, row in drug_df.iterrows()
        }

        norm_name = normalize_name(compound_name_raw)
        if norm_name not in compound_info:
            return None

        info = compound_info[norm_name]
        MW = info['MolecularWeight']
        C_max = info['Max_Concentration']

        try:
            step_index = step_index_norm * 7
            return round(C_max / (10 ** (step_index * 0.5)), 6)
        except:
            return None

    def replace_tokens_in_text(self, original_text, time, concentrations, compound, drug_df):
        original_text = original_text.replace("<TIME_TOKEN>", f"<TIME_TOKEN> hours")
        original_text = original_text.replace("<CONC_TOKEN>", f"<TIME_TOKEN> μM")
        actual_time = math.ceil(time * 112)
        updated_text = original_text.replace("<TIME_TOKEN>", str(actual_time))

        actual_concs = []
        for conc in concentrations:
            actual = self.infer_actual_concentration(compound, conc[0], conc[1], drug_df)
            actual_concs.append(f"{actual:.4f}" if actual is not None else "N/A")

        updated_text = updated_text.replace("<CONC_TOKEN>", ", ".join(actual_concs))
        updated_text = updated_text.replace("<COMPOUND_TOKEN>", compound)
        return updated_text, actual_concs, actual_time, compound

    def run(self, image_path: str):
        image = self.load_image_tensor(image_path)

        with open(self.text_jsonl, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f]

        processor = self.JsonlTextOnlyProcessor(tokenizer=self.tokenizer)
        all_features, raw_texts, times, concentrations, compound_embs = [], [], [], [], []
        batch_size = 64

        for i in tqdm(range(0, len(samples), batch_size), desc="Encoding"):
            batch = samples[i:i + batch_size]
            text_batch, conc_batch, time_batch, compound_batch = [], [], [], []

            for s in batch:
                t, c, tm, ce, rt = processor.process(s)
                text_batch.append(t)
                conc_batch.append(c)
                time_batch.append(tm)
                compound_batch.append(ce)
                raw_texts.append(rt)
                times.append(tm)
                concentrations.append(c)
                compound_embs.append(ce)

            text_batch = torch.stack(text_batch).to(self.device)
            conc_batch = torch.stack(conc_batch).to(self.device)
            time_batch = torch.stack(time_batch).to(self.device)
            compound_batch = torch.stack(compound_batch).to(self.device)

            with torch.no_grad():
                features = self.model.encode_text(
                    text=text_batch,
                    concentration=conc_batch,
                    time=time_batch,
                    compound_embedding=compound_batch,
                    normalize=True
                ).float()

            all_features.append(features.cpu())

        all_features = torch.cat(all_features, dim=0)
        compound_embs = torch.stack(compound_embs)

        with torch.no_grad():
            image_features = self.model.encode_image(image.unsqueeze(0).to(self.device), normalize=True).float()

        text_norm = all_features / all_features.norm(dim=1, keepdim=True)
        image_norm = image_features / image_features.norm(dim=1, keepdim=True)
        similarity = torch.matmul(text_norm.cpu(), image_norm.cpu().T).squeeze()
        best_idx = torch.argmax(similarity).item()

        best_text = raw_texts[best_idx]
        best_time = times[best_idx]
        best_conc = concentrations[best_idx]
        best_compound_emb = compound_embs[best_idx]

        compound_names, compound_embeddings = self.load_compound_embeddings(self.compound_npz_path)
        best_compound_name = self.find_compound_name(best_compound_emb.numpy(), compound_names, compound_embeddings)
        drug_df = self.load_druginfo(self.drug_csv_paths)

        final_text, actual_concs, actual_time, actual_compound = self.replace_tokens_in_text(
            best_text,
            best_time,
            [best_conc.tolist()],
            best_compound_name,
            drug_df
        )

        left_image = image[0].numpy()
        right_image = image[1].numpy()

        print("\n The most matching drug condition listed as below:")
        print("-" * 40)
        print(f"Text:      {best_text}")
        print(f"Time:         {actual_time} hours")
        print(f"Concentration:{actual_concs[0]} uM")
        print(f"Compound:     {actual_compound}")
        print(f"Similarity score:   {similarity[best_idx].item():.4f}")
        print("-" * 40)

        return final_text, left_image, right_image



class CellSegmentor:
    def __init__(self, model_ckpt_path, device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = CellSamWrapper(checkpoint=None)
        self.model.to(self.device)

        ckpt = torch.load(model_ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def preprocess(self, img_np):
        img_np = img_np.astype(np.float32) / 255.0
        img_np = np.stack([img_np] * 3, axis=0)
        return torch.tensor(img_np).unsqueeze(0).to(self.device)

    def get_instance_mask(self, tensor):
        with torch.no_grad():
            output = self.model(tensor)[0]
            instance_mask, _ = LogitsToLabels()(logits=output)
            return instance_mask

    def segment_and_plot(self, left_image, right_image, show=True):
        left_img = (left_image * 255).astype(np.uint8)
        right_img = (right_image * 255).astype(np.uint8)

        left_tensor = self.preprocess(left_img)
        right_tensor = self.preprocess(right_img)

        left_mask = self.get_instance_mask(left_tensor)
        right_mask = self.get_instance_mask(right_tensor)

        if show:
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            axs[0, 0].imshow(left_img, cmap="gray")
            axs[0, 0].set_title("Control Image")

            axs[0, 1].imshow(left_mask, cmap="nipy_spectral")
            axs[0, 1].set_title("Instance Mask L")

            axs[1, 0].imshow(right_img, cmap="gray")
            axs[1, 0].set_title("Perturb Image")

            axs[1, 1].imshow(right_mask, cmap="nipy_spectral")
            axs[1, 1].set_title("Instance Mask R")

            for ax in axs.ravel():
                ax.axis("off")
            plt.tight_layout()
            plt.show()

        return left_mask, right_mask, fig



class CellProfilerFeatureExtractor:
    def __init__(
        self,
        pipeline_paths: dict,
        output_base_dir: str,
        temp_input_base_dir: str
    ):
        self.pipeline_paths = {k: Path(v) for k, v in pipeline_paths.items()}
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        self.temp_input_base_dir = Path(temp_input_base_dir)
        self.temp_input_base_dir.mkdir(parents=True, exist_ok=True)

    def run_cp_on_pair(
        self,
        img: Union[np.ndarray, Image.Image],
        mask: Union[np.ndarray, Image.Image],
        tag: str,
        channel_type: str
    ) -> Union[Path, None]:
        assert channel_type in self.pipeline_paths, f"Unknown channel: {channel_type}"
        pipeline_path = self.pipeline_paths[channel_type]

        sample_id = tag
        condition = "default"
        mask_type = "nuclei" if channel_type == "DNA" else "cell"

        # Temp input dir
        temp_input_dir = self.temp_input_base_dir / sample_id / channel_type / condition
        if temp_input_dir.exists():
            shutil.rmtree(temp_input_dir)
        temp_input_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        image_dst_dir = temp_input_dir / "image" / channel_type
        image_dst_dir.mkdir(parents=True, exist_ok=True)
        img_name = f"{sample_id}_{channel_type}.png"
        img_path = image_dst_dir / img_name
        if img.dtype in [np.float32, np.float64]:
            img = (img * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img)
        img.save(img_path)

        # Save mask
        mask_dst_dir = temp_input_dir / "mask" / mask_type
        mask_dst_dir.mkdir(parents=True, exist_ok=True)
        mask_name = f"{sample_id}_{mask_type}.png"
        mask_path = mask_dst_dir / mask_name
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
        mask.save(mask_path)

        # Output dir
        output_dir = self.output_base_dir / sample_id / channel_type / condition
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        command = [
            "cellprofiler",
            "-c",
            "-r",
            "-p", str(pipeline_path),
            "-i", str(temp_input_dir),
            "-o", str(output_dir)
        ]

        print(f"\nRunning CellProfiler for: {tag} | Channel: {channel_type}")
        print(f"Image: {img_path}")
        print(f"Mask:  {mask_path}")
        print("Command:", " ".join(command))

        try:
            subprocess.run(command, check=True)
            print(f"CellProfiler finished for: {tag}")
        except subprocess.CalledProcessError as e:
            print(f"CellProfiler failed: {e}")
            return None

        # Return first CSV file
        csv_list = list(output_dir.glob("*.csv"))
        if not csv_list:
            print("No output CSV found.")
            return None

        print(f"Feature CSV saved: {csv_list[0]}")
        return csv_list[0]

    def extract_features(
        self,
        channel_type: str,
        control_tag: str = "control",
        perturb_tag: str = "perturb",
        feature_level: str = "cell"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:

        control_csv = self.output_base_dir / control_tag / channel_type / "default" / f"Expt_{feature_level}.csv"
        perturb_csv = self.output_base_dir / perturb_tag / channel_type / "default" / f"Expt_{feature_level}.csv"

        print(f"\nLoading Control CSV: {control_csv}")
        print(f"Loading Perturb CSV: {perturb_csv}")

        if not control_csv.exists() or not perturb_csv.exists():
            raise FileNotFoundError(" Control or Perturbation CSV not found.")

        control_df = pd.read_csv(control_csv)
        perturb_df = pd.read_csv(perturb_csv)

        all_columns = control_df.columns.tolist()
        feature_columns = all_columns[4:]  # skip metadata columns like ImageNumber, ObjectNumber, etc.

        filtered_columns = [
            col for col in feature_columns
            if not (col.startswith("Texture") and "00" in col)
        ]
        filtered_features = [
            col for col in filtered_columns
            if "Intensity" not in col
        ]

        print(f"\nFiltered {len(filtered_features)} features from channel: {channel_type}")
        return control_df, perturb_df, filtered_features, control_csv, perturb_csv




class LLMFeatureAnalyzer:
    def __init__(
        self,
        api_key: str,
        prompt_yaml_path_step1: str,
        prompt_yaml_path_step3: str,
        final_text: str,
        feature_names: List[str],
        control_profiler_csv: str,
        perturb_profiler_csv: str,
        left_img: np.ndarray,
        right_img: np.ndarray,
        model: str = "Claude-Sonnet-4",
        base_url: str = "https://api.poe.com/v1",
        use_data_url_images: bool = True,  
        left_image_url: Optional[str] = None,  
        right_image_url: Optional[str] = None, 
    ):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

        self.final_text = final_text
        self.feature_names = feature_names
        self.control_profiler_csv = control_profiler_csv
        self.perturb_profiler_csv = perturb_profiler_csv
        self.left_img = left_img
        self.right_img = right_img
        self.model = model

        self.use_data_url_images = use_data_url_images
        self.left_image_url = left_image_url
        self.right_image_url = right_image_url

        self.prompts_step1 = self._load_prompts(prompt_yaml_path_step1)
        self.prompts_step3 = self._load_prompts(prompt_yaml_path_step3)

        self.background_json = {}
        self.llm_input = {}
        self.prediction_response = {}
        self.summary_df = pd.DataFrame()

    def _load_prompts(self, yaml_path: str) -> Dict[str, Any]:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, list):
            return {it["name"]: it for it in data}
        return data


    def _render_prompt(self, template_str: str, variables: Dict[str, Any]) -> str:
        return Template(template_str).render(**variables)

    def _extract_json_from_markdown(self, text: str) -> dict:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            match2 = re.search(r"(\{.*\})", text, re.DOTALL)
            if not match2:
                raise ValueError("No JSON object found in model output.")
            json_str = match2.group(1)
        return json.loads(json_str)
    

    def _call_llm(self, prompt: str, sys_prompt: str = "", model: Optional[str] = None) -> Dict[str, str]:
        m = model or self.model
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=m,
            messages=messages,
            temperature=0.0,
        )
        answer = resp.choices[0].message.content if resp.choices else ""
        return {"thoughts": "", "answer": (answer or "").strip()}

    def _call_llm_multimodal(self, prompt: str, sys_prompt: str) -> Dict[str, str]:
        def ndarray_to_png_bytes(arr: np.ndarray) -> bytes:
            if arr.ndim != 2:
                raise ValueError("Expect a 2D grayscale ndarray.")
            a = np.ascontiguousarray(arr)
            amin, amax = float(a.min()), float(a.max())
            if amax <= 1.0 and amin >= 0.0:
                a8 = (a * 255.0).clip(0, 255).astype(np.uint8)
            else:
                a_norm = (a - amin) / (amax - amin) if amax != amin else np.zeros_like(a)
                a8 = (a_norm * 255.0).clip(0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(a8, mode="L").save(buf, format="PNG")
            return buf.getvalue()
        def png_bytes_to_data_url(png_bytes: bytes) -> str:
            return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")
        parts = []
        if sys_prompt:
            system_msg = {"role": "system", "content": sys_prompt}
        else:
            system_msg = None

        if self.use_data_url_images:
            left_bytes = ndarray_to_png_bytes(self.left_img)
            right_bytes = ndarray_to_png_bytes(self.right_img)
            left_url = png_bytes_to_data_url(left_bytes)
            right_url = png_bytes_to_data_url(right_bytes)
        else:
            if not (self.left_image_url and self.right_image_url):
                raise ValueError("Please provide left_image_url and right_image_url when use_data_url_images=False.")
            left_url = self.left_image_url
            right_url = self.right_image_url

        user_content = [
            {"type": "text", "text": "Image A (Control)"},
            {"type": "image_url", "image_url": {"url": left_url}},
            {"type": "text", "text": "Image B (Perturbed)"},
            {"type": "image_url", "image_url": {"url": right_url}},
            {"type": "text", "text": prompt},
        ]

        messages = []
        if system_msg:
            messages.append(system_msg)
        messages.append({"role": "user", "content": user_content})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        answer = resp.choices[0].message.content if resp.choices else ""
        return {"thoughts": "", "answer": (answer or "").strip()}

    def _extract_names_from_llm_answer(self, answer: Any) -> Tuple[List[str], Optional[dict]]:
        try:
            obj = self._extract_json_from_markdown(answer)
            feats = obj.get("features_ranked", [])
            names = [f["name"] for f in feats if isinstance(f, dict) and "name" in f]
            return list(dict.fromkeys(names)), obj
        except Exception:
            return [], None

    def step1_generate_background_and_features(self):
        sys_prompt = self.prompts_step1["featRank_sys"]["prompt"]
        background_template = self.prompts_step1["background_curation"]["prompt"]
        feature_template = self.prompts_step1["featRank_user"]["prompt"]

        background_prompt = self._render_prompt(background_template, {
            "perturbation_condition": self.final_text
        })

        print("\n[Step 1A] Generating background...")
        background_response = self._call_llm(background_prompt, sys_prompt)
        self.background_json = self._extract_json_from_markdown(background_response["answer"])

        feature_prompt = self._render_prompt(feature_template, {
            "perturbation_condition": self.final_text,
            "feature_names_json": self.feature_names,
            "background_curation_json": json.dumps(self.background_json, indent=2)
        })

        print("\n[Step 1B] Predicting features...")
        prediction_response = self._call_llm(feature_prompt, sys_prompt)
        self.prediction_response = prediction_response

        return prediction_response

    def step2_compute_feature_statistics(self):
        names, _ = self._extract_names_from_llm_answer(self.prediction_response["answer"])
        control_df = pd.read_csv(self.control_profiler_csv)
        perturb_df = pd.read_csv(self.perturb_profiler_csv)

        existing_names = [n for n in names if n in control_df.columns and n in perturb_df.columns]
        control_filtered = control_df[existing_names]
        perturb_filtered = perturb_df[existing_names]

        def bootstrap_delta_ci(a, b, n=1000, ci=95):
            rng = np.random.default_rng(42)
            deltas = [np.median(rng.choice(b, size=len(b), replace=True)) -
                      np.median(rng.choice(a, size=len(a), replace=True)) for _ in range(n)]
            return np.percentile(deltas, (100 - ci) / 2), np.percentile(deltas, 100 - (100 - ci) / 2)

        def cliffs_delta(a, b):
            n, m = len(a), len(b)
            more = sum(x > y for x in a for y in b)
            less = sum(x < y for x in a for y in b)
            return (more - less) / (n * m)

        summary_data = []
        for feature in tqdm(existing_names, desc="Computing statistics"):
            a = control_filtered[feature].dropna().values
            b = perturb_filtered[feature].dropna().values
            if len(a) < 5 or len(b) < 5:
                continue
            q_a = np.percentile(a, [10, 25, 50, 75, 90])
            q_b = np.percentile(b, [10, 25, 50, 75, 90])
            delta = np.median(b) - np.median(a)
            ci_low, ci_up = bootstrap_delta_ci(a, b)
            cd = cliffs_delta(a, b)
            try:
                _, p = mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                p = np.nan
            summary_data.append({
                "feature": feature,
                "n_control": len(a),
                "n_perturb": len(b),
                "median_control": np.median(a),
                "median_perturb": np.median(b),
                "mad_control": np.median(np.abs(a - np.median(a))),
                "mad_perturb": np.median(np.abs(b - np.median(b))),
                "p10_control": q_a[0],
                "p25_control": q_a[1],
                "p50_control": q_a[2],
                "p75_control": q_a[3],
                "p90_control": q_a[4],
                "p10_perturb": q_b[0],
                "p25_perturb": q_b[1],
                "p50_perturb": q_b[2],
                "p75_perturb": q_b[3],
                "p90_perturb": q_b[4],
                "delta_median": delta,
                "bootstrap_ci_lower": ci_low,
                "bootstrap_ci_upper": ci_up,
                "cliffs_delta": cd,
                "p_value": p
            })

        df = pd.DataFrame(summary_data)
        if df.empty:
            self.summary_df = df
            return {"number_of_cells_control": 0, "number_of_cells_perturb": 0, "records": []}

        df = df.sort_values("p_value")
        df["rank"] = np.arange(1, len(df) + 1)
        df["q_value"] = np.minimum.accumulate((df["p_value"] * len(df) / df["rank"])[::-1])[::-1]
        self.summary_df = df

        df["direction"] = np.where(df["delta_median"] > 0, "increase",
                                   np.where(df["delta_median"] < 0, "decrease", "ambiguous"))
        df["abs_cd"] = df["cliffs_delta"].abs()
        df_sorted = df.sort_values(["q_value", "abs_cd", "delta_median"],
                                   ascending=[True, False, False]).head(20)

        # cols = ["feature", "median_control", "median_perturb", "delta_median",
        #         "bootstrap_ci_lower", "bootstrap_ci_upper", "cliffs_delta", "p_value", "q_value", "direction"]
        # self.llm_input = {
        #     "number_of_cells_control": int(df["n_control"].iloc[0]),
        #     "number_of_cells_perturb": int(df["n_perturb"].iloc[0]),
        #     "records": df_sorted[cols].to_dict(orient="records")
        # }
        self.llm_input = {
            "number_of_cells_control": int(df["n_control"].iloc[0]),
            "number_of_cells_perturb": int(df["n_perturb"].iloc[0]),
            "records": df_sorted.to_dict(orient="records")  
        }

        return self.llm_input

    def step3_generate_consistency_prediction(self):
        sys_prompt = self.prompts_step3["mechConsistency_sys"]["prompt"]
        user_prompt = self.prompts_step3["mechConsistency_user"]["prompt"]
        prompt_text = self._render_prompt(user_prompt, {
            "perturbation_condition": self.final_text,
            "background_curation_json": json.dumps(self.background_json),
            "summary_of_features_json": json.dumps(self.llm_input, indent=2)
        })
        print("\n[Step 2] Generating report...")
        response = self._call_llm_multimodal(prompt_text, sys_prompt)
        self.prediction_response = response
        return response

    def step4_plot_supporting_features(
        self,
        figsize: Tuple[int, int] = (5, 3),
        color_control: str = 'skyblue',
        color_perturb: str = 'salmon',
    ) -> List[plt.Figure]:
        llm_json = self._extract_json_from_markdown(self.prediction_response.get("answer", "{}"))
        if llm_json.get("mechanism_consistency") != "supports":
            print("Mechanism NOT supported. No plots will be generated.")
            return []

        support_features = [
            f["name"] for f in llm_json.get("features_ranked", [])
            if f.get("supports_proposed_mechanism") == "supports"
        ]

        figures = []

        for feature in support_features:
            row = self.summary_df[self.summary_df["feature"] == feature]
            if row.empty:
                print(f"[Warning] Feature {feature} not found.")
                continue
            row = row.iloc[0]
            x_labels = ["Control", "Perturb"]
            x_pos = [0.0, 0.55]
            stats = {}
            for group in ["control", "perturb"]:
                stats[group] = {k: row[f"{k}_{group}"] for k in ["median", "mad", "p10", "p25", "p50", "p75", "p90"]}

            fig, ax = plt.subplots(figsize=figsize)

            for i, group in enumerate(["control", "perturb"]):
                x = x_pos[i]
                color = color_control if group == "control" else color_perturb

                # p25-p75 box
                box = patches.Rectangle(
                    (x - 0.15, stats[group]["p25"]),
                    width=0.3,
                    height=stats[group]["p75"] - stats[group]["p25"],
                    edgecolor='black',
                    facecolor=color,
                    alpha=0.5,
                    label="p25-p75" if i == 0 else None
                )
                ax.add_patch(box)

                # median line
                ax.plot([x - 0.15, x + 0.15],
                        [stats[group]["median"]] * 2,
                        color='black',
                        linewidth=2,
                        label="median" if i == 0 else None)

                # p10-p90 whiskers with caps
                ax.plot([x, x],
                        [stats[group]["p10"], stats[group]["p90"]],
                        color=color,
                        linewidth=1.5,
                        label="p10-p90" if i == 0 else None)
                ax.plot([x - 0.1, x + 0.1], [stats[group]["p10"]] * 2,
                        color=color, linewidth=1.5)
                ax.plot([x - 0.1, x + 0.1], [stats[group]["p90"]] * 2,
                        color=color, linewidth=1.5)

                # ±MAD dashed line
                ax.plot([x, x],
                        [stats[group]["median"] - stats[group]["mad"],
                         stats[group]["median"] + stats[group]["mad"]],
                        color='gray',
                        linestyle='--',
                        linewidth=3,
                        zorder=10,
                        label="± MAD" if i == 0 else None)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
            ax.set_title(f"{feature}")
            ax.set_ylabel("Feature Value")
            ax.grid(axis="y", linestyle="--", alpha=0.6)

            plt.legend(
                loc='upper left',
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.,
                frameon=True
            )

            fig.tight_layout()
            figures.append(fig)

        return figures