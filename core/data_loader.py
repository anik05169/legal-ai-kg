import os
import json
import zipfile
from datasets import load_dataset

class LocalCUADDataset:
    """Wraps a list of parsed JSON dataset items to mimic HuggingFace's Dataset interface."""
    def __init__(self, items):
        self.items = items
    
    def __getitem__(self, key):
        if isinstance(key, str):
            # Support column-based indexing like dataset["context"]
            return [item[key] for item in self.items]
        # Support integer indexing like dataset[0]
        return self.items[key]
    
    def __iter__(self):
        return iter(self.items)
    
    def __len__(self):
        return len(self.items)

def load_cuad_dataset():
    """
    Loads the CUAD dataset. First checks locally in 'data/CUADv1.zip' or 'data/CUADv1.json'.
    If neither is found, falls back to Hugging Face datasets hub.
    """
    zip_path = os.path.join("data", "CUADv1.zip")
    json_path = os.path.join("data", "CUADv1.json")
    
    # 1. Try Loading from Local ZIP (Highly recommended for Git/GitHub Actions to keep repo light)
    if os.path.exists(zip_path):
        print(f"📦 Found local compressed dataset: {zip_path}. Loading...")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                json_files = [f for f in z.namelist() if f.endswith(".json")]
                if not json_files:
                    raise FileNotFoundError("❌ No .json file found inside CUADv1.zip")
                with z.open(json_files[0]) as f:
                    cuad_data = json.load(f)
            return _parse_squad_json(cuad_data)
        except Exception as e:
            print(f"⚠️ Failed to load zip: {e}. Falling back...")
            
    # 2. Try Loading from Local JSON
    if os.path.exists(json_path):
        print(f"📄 Found local raw JSON dataset: {json_path}. Loading...")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                cuad_data = json.load(f)
            return _parse_squad_json(cuad_data)
        except Exception as e:
            print(f"⚠️ Failed to load raw JSON: {e}. Falling back...")
            
    # 3. Fallback to Hugging Face Datasets Hub
    print("🌐 No local dataset found in 'data/'. Falling back to Hugging Face...")
    return load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)

def _parse_squad_json(cuad_data):
    """Parses SQuAD-formatted CUAD JSON and maps it to a HuggingFace-compatible dataset format."""
    items = []
    for doc in cuad_data.get("data", []):
        for paragraph in doc.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                item = {
                    "context": context,
                    "question": qa.get("question", ""),
                    "answers": {
                        "text": [ans.get("text", "") for ans in qa.get("answers", [])],
                        "answer_start": [ans.get("answer_start", 0) for ans in qa.get("answers", [])]
                    },
                    "id": qa.get("id", ""),
                    "is_impossible": qa.get("is_impossible", False)
                }
                items.append(item)
    return LocalCUADDataset(items)

def get_cuad_contracts(num_samples=5):
    """
    Pulls the CUAD dataset and extracts unique contract texts.
    """
    dataset = load_cuad_dataset()
    
    unique_contexts = list(set(dataset["context"]))
    print(f"📊 Total unique contracts found in CUAD: {len(unique_contexts)}")
    
    samples = unique_contexts[:num_samples]
    print(f"🚀 Returning {len(samples)} contracts for processing.")
    
    return samples

if __name__ == "__main__":
    contracts = get_cuad_contracts(num_samples=1)
    print(f"Sample contract length: {len(contracts[0])} characters")