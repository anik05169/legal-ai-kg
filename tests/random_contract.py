from datasets import load_dataset
import random

def preview_random_contract():
    print("📥 Downloading/Loading CUAD dataset...")
    
    # Updated to use the reliable path from your data loader!
    dataset = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)

    print("🧠 Grouping questions by contract...")
    contracts_dict = {}

    # Group all questions and answers by their associated contract text
    for item in dataset:
        context = item['context']
        question = item['question']
        
        # Safely handle the SQuAD 2.0 answers dictionary format
        if 'answers' in item and isinstance(item['answers'], dict) and len(item['answers'].get('text', [])) > 0:
            answer = item['answers']['text'][0]
        else:
            answer = "[No clause found for this question in this contract]"

        if context not in contracts_dict:
            contracts_dict[context] = []
            
        contracts_dict[context].append({
            "question": question,
            "answer": answer
        })

    # Pick a random contract from our grouped dictionary
    random_contract_text = random.choice(list(contracts_dict.keys()))
    associated_qa = contracts_dict[random_contract_text]

    # --- PRINT THE RESULTS ---
    print("\n" + "="*80)
    print("📄 RANDOM CONTRACT PREVIEW (First 1500 characters):")
    print("="*80)
    print(random_contract_text[:1500].strip() + "\n\n... [CONTRACT CONTINUES] ...")
    
    print("\n" + "="*80)
    print(f"❓ EXPERT LEGAL QUESTIONS FOR THIS CONTRACT ({len(associated_qa)} total):")
    print("="*80)

    # Print up to 10 questions to keep the terminal readable
    for i, qa in enumerate(associated_qa[:10], 1):
        print(f"\nQ{i}: {qa['question']}")
        
        # Clean up newlines in the answer for a cleaner printout
        clean_answer = str(qa['answer']).replace('\n', ' ')
        
        # Truncate really long answers so they don't flood the screen
        if len(clean_answer) > 150:
            print(f"A{i}: {clean_answer[:150]}...")
        else:
            print(f"A{i}: {clean_answer}")

    if len(associated_qa) > 10:
        print(f"\n... and {len(associated_qa) - 10} more questions.")

if __name__ == "__main__":
    preview_random_contract()