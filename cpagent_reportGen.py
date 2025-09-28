# cpagent_reportGen.py

import os
import sys
import json
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from contextlib import suppress

from config import get_config
from cpagent_utils import DrugTextImageMatcher, CellSegmentor, CellProfilerFeatureExtractor, LLMFeatureAnalyzer
# from segmentor_utils import CellSegmentor
# from cellprofiler_utils import CellProfilerFeatureExtractor
# from reasoning_utils import LLMFeatureAnalyzer


def run_experiment(cfg):
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg["device"].split(":")[-1]
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    if cfg["open_clip_path"] not in sys.path:
        sys.path.insert(0, cfg["open_clip_path"])
    import open_clip  # noqa

    print(f"\nRunning experiment: {cfg['experiment']}")
    output_dir = os.path.join(cfg["output_base"], cfg["experiment"])
    os.makedirs(output_dir, exist_ok=True)

    matcher = DrugTextImageMatcher(
        open_clip_path=cfg["open_clip_path"],
        pretrained_ckpt_path=cfg["pretrained_ckpt_path"],
        text_jsonl=cfg["text_jsonl"],
        compound_npz_path=cfg["compound_npz_path"],
        drug_csv_paths=cfg["drug_csv_paths"],
        device=device
    )

    # === Match image to text ===
    final_text, left_img, right_img = matcher.run(cfg["image_path"])

    # Save match result
    with open(os.path.join(output_dir, "match_result.txt"), "w", encoding="utf-8") as f:
        f.write("Best Match:\n" + final_text + "\n")

    # === Segment cells ===
    segmenter = CellSegmentor(
        model_ckpt_path=cfg["segmentor_ckpt"],
        device=device
    )

    left_mask, right_mask, fig = segmenter.segment_and_plot(left_img, right_img)
    fig.savefig(os.path.join(output_dir, "segmentation_result.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # === Extract features ===
    match = re.search(r"Image channel is ([^;]+)", final_text)
    channel_type = match.group(1).strip()
    feature_level = "nuclei" if channel_type == "DNA" else "cell"

    extractor = CellProfilerFeatureExtractor(
        pipeline_paths=cfg["pipeline_paths"],
        output_base_dir=os.path.join(output_dir, "feature_output"),
        temp_input_base_dir=os.path.join(output_dir, "temp_input")
    )

    extractor.run_cp_on_pair(left_img, left_mask, "control", channel_type)
    extractor.run_cp_on_pair(right_img, right_mask, "perturb", channel_type)

    control_df, perturb_df, selected_features, control_csv, perturb_csv = extractor.extract_features(
        channel_type=channel_type,
        feature_level=feature_level
    )

    # === LLM Reasoning ===
    for model_name in cfg["models"]:
        print(f"{cfg['experiment']} | Model: {model_name}")

        analyzer = LLMFeatureAnalyzer(
            api_key=cfg["api_key"],
            prompt_yaml_path_step1=cfg["prompt_yaml_path_step1"],
            prompt_yaml_path_step3=cfg["prompt_yaml_path_step3"],
            final_text=final_text,
            feature_names=selected_features,
            control_profiler_csv=control_csv,
            perturb_profiler_csv=perturb_csv,
            left_img=left_img,
            right_img=right_img,
            model=model_name,
            base_url="https://api.poe.com/v1",
            use_data_url_images=True,
        )

        feature_response = analyzer.step1_generate_background_and_features()
        with open(os.path.join(output_dir, f"{model_name}_feature_response.json"), "w", encoding="utf-8") as f:
            json.dump(feature_response, f, indent=2, ensure_ascii=False)

        stats = analyzer.step2_compute_feature_statistics()
        with open(os.path.join(output_dir, f"{model_name}_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        report_response = analyzer.step3_generate_consistency_prediction()
        report_copy = dict(report_response)
        answer_text = report_copy.pop("answer", "")

        with open(os.path.join(output_dir, f"{model_name}_report_response.json"), "w", encoding="utf-8") as f:
            json.dump(report_copy, f, indent=2, ensure_ascii=False)

        with open(os.path.join(output_dir, f"{model_name}_llm_answer.txt"), "w", encoding="utf-8") as f:
            f.write(answer_text)

        with suppress(Exception):
            parsed = analyzer._extract_json_from_markdown(answer_text)
            with open(os.path.join(output_dir, f"{model_name}_llm_answer_parsed.json"), "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)

        figures = analyzer.step4_plot_supporting_features()
        for j, fig in enumerate(figures):
            fig_path = os.path.join(output_dir, f"{model_name}_fig_{j + 1}.pdf")
            fig.savefig(fig_path, format='pdf')
            plt.close(fig)

        print(f"{model_name} completed for {cfg['experiment']}")


if __name__ == "__main__":
    config = get_config()
    run_experiment(config)