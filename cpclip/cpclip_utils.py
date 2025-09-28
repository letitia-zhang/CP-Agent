import os
import re
import json
import math
import random
import numpy as np
import pandas as pd
import torch
import textwrap
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def normalize_name(name: str) -> str:
    return "".join(filter(str.isalnum, str(name).lower()))

def load_druginfo(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["norm_name"] = df["compound"].apply(normalize_name)
    return df

def infer_actual_concentration(compound_name_raw, max_mass_density_norm, step_index_norm, drug_df):
    norm_name = normalize_name(compound_name_raw)
    compound_info = {row["norm_name"]: row for _, row in drug_df.iterrows()}
    if norm_name not in compound_info:
        return -1
    info = compound_info[norm_name]
    try:
        C_max = info["Max_Concentration"]
        step_index = step_index_norm * 7
        return round(C_max / (10 ** (step_index * 0.5)), 6)
    except:
        return -1

def find_compound_name(embedding, compound_names, compound_embeddings):
    embedding = embedding.reshape(1, -1)
    sims = cosine_similarity(embedding, compound_embeddings)[0]
    return compound_names[np.argmax(sims)]

def replace_tokens_in_text(original_text, time, concentrations, compound, drug_df):
    actual_time = math.ceil(time.item() * 112) - 1
    actual_concs = []
    for conc in concentrations:
        val = infer_actual_concentration(compound, conc[0], conc[1], drug_df)
        actual_concs.append(f"{val:.4f} μM" if val >= 0 else "N/A")
    updated_text = original_text
    updated_text = updated_text.replace("<TIME_TOKEN>", f"{actual_time:.1f} hours")
    updated_text = updated_text.replace("<CONC_TOKEN>", ", ".join(actual_concs))
    updated_text = updated_text.replace("<COMPOUND_TOKEN>", compound)
    return updated_text

def extract_from_text(text: str, key: str) -> str:
    pattern = rf"{re.escape(key)}\s*(.*?)\;"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else "unknown"

def normalize_token(s: str) -> str:
    s = s.lower().replace(".", "dot")
    return re.sub(r"[^\w]", "", s)

def get_filename_from_final_text(final_text: str) -> str:
    cell_line = extract_from_text(final_text, "Cell line is")
    channel = extract_from_text(final_text, "Image channel is")
    concentration = extract_from_text(final_text, "The concentration is").split()[0]
    time_val = extract_from_text(final_text, "The observation time is").split()[0]
    compound = extract_from_text(final_text, "The perturbation compound is")

    parts = [
        normalize_token(cell_line),
        normalize_token(channel),
        normalize_token(concentration),
        normalize_token(time_val),
        normalize_token(compound)
    ]
    return "_".join(parts) + ".png"

def generate_example_image(
    jsonl_path: str,
    image_root: str,
    compound_npz_path: str,
    drug_csv_paths: list,
    save_dir: str
):
    with open(jsonl_path, "r") as f:
        lines = f.readlines()
        sample = json.loads(random.choice(lines))

    image_path = os.path.join(image_root, sample["image"])
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = Image.open(image_path).convert("L")
    img_np = np.array(image)
    left = img_np[:, :512]
    right = img_np[:, 512:]
    combined = np.concatenate([left, right], axis=1)
    raw_text = sample["text"]
    concentration = torch.tensor(sample["Concentration"], dtype=torch.float32)
    time = torch.tensor(sample["Time"], dtype=torch.float32)
    compound_embedding = torch.tensor(sample["compound_embedding"], dtype=torch.float32)
    npz = np.load(compound_npz_path, allow_pickle=True)
    compound_names = npz["compounds"]
    compound_embeddings = npz["embeddings"]

    compound_name = find_compound_name(compound_embedding.numpy(), compound_names, compound_embeddings)

    drug_df = load_druginfo(drug_csv_paths)
    final_text = replace_tokens_in_text(raw_text, time, [concentration.tolist()], compound_name, drug_df)

    print("\n=== Final Text ===\n")
    print(textwrap.fill(final_text, width=80))

    filename = get_filename_from_final_text(final_text)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    Image.fromarray(img_np).save(save_path)

    # plt.figure(figsize=(6, 3))
    # plt.imshow(combined, cmap="gray")
    # plt.axis("off")
    # plt.tight_layout()
    # plt.savefig(save_path, dpi=300)
    # plt.close()

    print(f"\nImage saved: {save_path}")