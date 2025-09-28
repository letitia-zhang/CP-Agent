import os
import json
import torch
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader



class CompoundMatcher:
    def __init__(
        self,
        jsonl_path: str,
        npz_path: str,
        image_base_dir: str,
        tokenizer,
        model,
        device,
        target_compound: str,
        target_moa: str = None,
        num_samples: int = 100,
        similarity_threshold: float = 0.85,
        batch_size: int = 16,
        seed: int = 42
    ):
        self.jsonl_path = jsonl_path
        self.npz_path = npz_path
        self.image_base_dir = image_base_dir
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.target_compound = target_compound
        self.target_moa = target_moa
        self.num_samples = num_samples
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.seed = seed

        self.matched_samples = self._get_matched_samples()
        self.dataset = self._build_dataset()
        self._prepare_batches()

    def _get_matched_samples(self):
        np.random.seed(self.seed)
        npz_data = np.load(self.npz_path, allow_pickle=True)
        compound_names_all = list(npz_data['compounds'])
        compound_embeddings_all = npz_data['embeddings']
        name_to_index = {name: idx for idx, name in enumerate(compound_names_all)}

        assert self.target_compound in name_to_index, f"❌ Compound '{self.target_compound}' not found."

        target_embedding = compound_embeddings_all[name_to_index[self.target_compound]].reshape(1, -1)

        matched = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                emb = sample.get("compound_embedding")
                if emb is None:
                    continue
                emb_np = np.array(emb, dtype=np.float32).reshape(1, -1)

                sim = cosine_similarity(emb_np, target_embedding)[0][0]
                if sim >= self.similarity_threshold:
                    if self.target_moa:
                        text = sample.get("text", "")
                        if f"The mechanism of action for this compound is {self.target_moa}" not in text:
                            continue
                    matched.append(sample)
                    if len(matched) >= self.num_samples:
                        break

        print(f"Found {len(matched)} samples for compound '{self.target_compound}' (sim ≥ {self.similarity_threshold})")
        return matched

    def _build_dataset(self):
        class JsonlDataset_unseen(Dataset):
            def __init__(self, samples, transforms=None, tokenizer=None, image_base_dir=""):
                self.samples = samples
                self.transforms = transforms
                self.tokenize = tokenizer
                self.image_base_dir = image_base_dir

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                sample = self.samples[idx]
                image_path = os.path.join(self.image_base_dir, sample['image'])
                full_img = Image.open(image_path).convert("L")
                full_tensor = torch.from_numpy(np.array(full_img)).unsqueeze(0).float() / 255.

                left_tensor = full_tensor[:, :, :512]
                right_tensor = full_tensor[:, :, 512:]
                image = torch.cat([left_tensor, right_tensor], dim=0)

                if self.tokenize:
                    text = self.tokenize([str(sample['text'])])[0]
                else:
                    text = str(sample['text'])

                concentration = torch.tensor(sample["Concentration"], dtype=torch.float32)
                time = torch.tensor(sample["Time"], dtype=torch.float32)
                compound_embedding = torch.tensor(sample["compound_embedding"], dtype=torch.float32)

                return image, text, concentration, time, compound_embedding, str(sample['text'])

        return JsonlDataset_unseen(
            samples=self.matched_samples,
            tokenizer=self.tokenizer,
            image_base_dir=self.image_base_dir,
        )

    def _prepare_batches(self):
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,        
            pin_memory=True       
        )

        all_images = []
        all_texts = []
        all_concs = []
        all_times = []
        all_compounds = []
        all_raw_texts = []

        for batch in tqdm(dataloader, desc="Loading batches"):
            image, text, conc, time, compound, raw_text = batch
            all_images.append(image)
            all_texts.append(text)
            all_concs.append(conc)
            all_times.append(time)
            all_compounds.append(compound)
            all_raw_texts.extend(raw_text)

        self.image_batch = torch.cat(all_images).to(self.device)
        self.text_batch = torch.cat(all_texts).to(self.device)
        self.conc_batch = torch.cat(all_concs).to(self.device)
        self.time_batch = torch.cat(all_times).to(self.device)
        self.compound_batch = torch.cat(all_compounds).to(self.device)
        self.all_raw_texts = all_raw_texts

    def compute_features(self):
        all_image_features = []
        all_text_features = []
        model = self.model
        model.eval()

        with torch.no_grad():
            num_samples = len(self.image_batch)
            for i in tqdm(range(0, num_samples, self.batch_size), desc="Encoding batches"):
                s = slice(i, i + self.batch_size)

                image = self.image_batch[s]
                text = self.text_batch[s]
                conc = self.conc_batch[s]
                time_ = self.time_batch[s]
                compound = self.compound_batch[s]

                text_feat = model.encode_text(
                    text=text,
                    concentration=conc,
                    time=time_,
                    compound_embedding=compound,
                    normalize=True
                ).float().cpu()

                image_feat = model.encode_image(
                    image=image,
                    normalize=True
                ).float().cpu()

                all_text_features.append(text_feat)
                all_image_features.append(image_feat)

        self.text_features = torch.cat(all_text_features, dim=0)
        self.image_features = torch.cat(all_image_features, dim=0)

        print("Feature extraction complete.")
        print("Text features shape:", self.text_features.shape)
        print("Image features shape:", self.image_features.shape)

    def compute_similarity_stats(self):
        cos_sim = F.cosine_similarity(self.text_features, self.image_features, dim=1)
        mean_sim = cos_sim.mean().item()
        std_sim = cos_sim.std().item()

        print(f"Cosine Similarity Mean: {mean_sim:.4f}")
        print(f"Cosine Similarity Std:  {std_sim:.4f}")

    def run_all(self):
        self.compute_features()
        self.compute_similarity_stats()


class CompoundMatcher_orig:
    def __init__(
        self,
        jsonl_path: str,
        image_base_dir: str,
        tokenizer,
        model,
        device,
        target_compound: str,
        target_moa: str = None,
        num_samples: int = 100,
        batch_size: int = 16,
        seed: int = 42
    ):
        self.jsonl_path = jsonl_path
        self.image_base_dir = image_base_dir
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.target_compound = target_compound.lower()
        self.target_moa = target_moa.lower() if target_moa else None
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed

        self.matched_samples = self._get_matched_samples()
        self.dataset = self._build_dataset()
        self._prepare_batches()

    def _get_matched_samples(self):
        matched = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)

                text = sample.get("text", "").lower()
                if f"perturbation compound is {self.target_compound}" not in text:
                    continue

                if self.target_moa:
                    if f"mechanism of action for this compound is {self.target_moa}" not in text:
                        continue

                matched.append(sample)
                if len(matched) >= self.num_samples:
                    break

        print(f"Found {len(matched)} samples with compound '{self.target_compound}'"
              + (f" and MoA '{self.target_moa}'" if self.target_moa else "")
              + ".")
        return matched

    def _build_dataset(self):
        class JsonlDatasetSimple(Dataset):
            def __init__(self, samples, tokenizer=None, image_base_dir=""):
                self.samples = samples
                self.tokenizer = tokenizer
                self.image_base_dir = image_base_dir

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                sample = self.samples[idx]
                image_path = os.path.join(self.image_base_dir, sample['image'])
                full_img = Image.open(image_path).convert("L")
                full_tensor = torch.from_numpy(np.array(full_img)).unsqueeze(0).float() / 255.

                # Split left/right channels
                left_tensor = full_tensor[:, :, :512]
                right_tensor = full_tensor[:, :, 512:]
                image = torch.cat([left_tensor, right_tensor], dim=0)  # [2, 512, 512]

                text = sample['text']
                if self.tokenizer:
                    text_tensor = self.tokenizer([text])[0]
                else:
                    text_tensor = text

                return image, text_tensor, text  # image, tokenized_text, raw_text

        return JsonlDatasetSimple(
            samples=self.matched_samples,
            tokenizer=self.tokenizer,
            image_base_dir=self.image_base_dir
        )

    def _prepare_batches(self):
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        self.image_batches = []
        self.text_batches = []
        self.raw_texts = []

        for image, text, raw_text in tqdm(dataloader, desc="Loading batches"):
            self.image_batches.append(image.to(self.device))
            self.text_batches.append(text.to(self.device))
            self.raw_texts.extend(raw_text)

    def compute_features(self):
        all_image_features = []
        all_text_features = []
        model = self.model
        model.eval()

        with torch.no_grad():
            for image_batch, text_batch in tqdm(zip(self.image_batches, self.text_batches),
                                                total=len(self.image_batches),
                                                desc="Encoding features"):
                text_feat = model.encode_text(text=text_batch, normalize=True).float().cpu()
                image_feat = model.encode_image(image=image_batch, normalize=True).float().cpu()

                all_text_features.append(text_feat)
                all_image_features.append(image_feat)

        self.text_features = torch.cat(all_text_features, dim=0)
        self.image_features = torch.cat(all_image_features, dim=0)

        print("Feature extraction complete.")
        print("Text features shape:", self.text_features.shape)
        print("Image features shape:", self.image_features.shape)

    def compute_similarity_stats(self):
        cos_sim = F.cosine_similarity(self.text_features, self.image_features, dim=1)
        mean_sim = cos_sim.mean().item()
        std_sim = cos_sim.std().item()

        print(f"Cosine Similarity Mean: {mean_sim:.4f}")
        print(f"Cosine Similarity Std:  {std_sim:.4f}")

    def run_all(self):
        self.compute_features()
        self.compute_similarity_stats()