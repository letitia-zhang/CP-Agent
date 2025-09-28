import os

def get_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))  
    return {
        "device": "cuda:7",
        "experiment": "data4",
        "image_path": os.path.join(base_dir, "example_images", "mcf7_dna_100dot0000_24dot0_alln.png"),

        "open_clip_path": os.path.join(base_dir, "cpclip/model/src"),
        "pretrained_ckpt_path": os.path.join(base_dir, "cpclip", "cpclip_vitB_descriptor.pt"),
        "text_jsonl": os.path.join(base_dir, "cpclip", "context_dedup.jsonl"),
        "compound_npz_path": os.path.join(base_dir, "metadata", "all_compounds_embedding.npz"),
        "drug_csv_paths": [
            os.path.join(base_dir, "metadata", "bbbc021_drugExperiment.csv"),
            os.path.join(base_dir, "metadata", "cpjump_drugExperiment.csv"),
            os.path.join(base_dir, "metadata", "rxrx3_drugExperiment.csv")
        ],

        "segmentor_ckpt": os.path.join(base_dir, "segmentor", "model_tuned.pt"),
        "pipeline_paths": {
            "DNA": os.path.join(base_dir, "featureExtractor/generalPipelines", "dnaprofiler.cppipe"),
            "AGP": os.path.join(base_dir, "featureExtractor/generalPipelines", "agpprofiler.cppipe"),
            "ER": os.path.join(base_dir, "featureExtractor/generalPipelines", "erprofiler.cppipe"),
            "Mito": os.path.join(base_dir, "featureExtractor/generalPipelines", "mitoprofiler.cppipe"),
            "RNA": os.path.join(base_dir, "featureExtractor/generalPipelines", "rnaprofiler.cppipe"),
            "Actin": os.path.join(base_dir, "featureExtractor/generalPipelines", "actinprofiler.cppipe"),
            "Tubulin": os.path.join(base_dir, "featureExtractor/generalPipelines", "tubulinprofiler.cppipe"),
        },
        "output_base": os.path.join(base_dir, "results"),

        "models": ["GPT-5"],
        "api_key": "eZG8-qVwGLMfIJwJy94dEX44v3p9vwolxLn-HgtRIug",  
        "prompt_yaml_path_step1": os.path.join(base_dir, "reasoning_utils", "featRank.yaml"),
        "prompt_yaml_path_step3": os.path.join(base_dir, "reasoning_utils", "reportGen.yaml")
    }