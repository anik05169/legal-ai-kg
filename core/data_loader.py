from datasets import load_dataset

def get_cuad_contracts(num_samples=5):
    """
    Pulls the CUAD dataset and extracts unique contract texts.
    """
    print("📥 Downloading/Loading CUAD dataset from Hugging Face...")
    dataset = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)
    
    unique_contexts = list(set(dataset["context"]))
    print(f"📊 Total unique contracts found in CUAD: {len(unique_contexts)}")
    
    samples = unique_contexts[:num_samples]
    print(f"🚀 Returning {len(samples)} contracts for processing.")
    
    return samples

if __name__ == "__main__":
    contracts = get_cuad_contracts(num_samples=1)
    print(f"Sample contract length: {len(contracts[0])} characters")