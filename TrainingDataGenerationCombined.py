import json
import os
import random
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm

def save_jsonl(data: List[Dict], output_path: str):
    """Save data in JSONL format"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def combine_enriched_data(corpora: List[str]) -> List[Dict]:
    """Combine all enriched JSON files into one dataset"""
    combined_data = []
    for corpus in tqdm(corpora, desc="Combining corpora"):
        input_path = f"CorpusAugmented/{corpus}_enriched_qa_dataset.json"
        with open(input_path, 'r', encoding='utf-8') as f:
            enriched_data = json.load(f)
            combined_data.extend(enriched_data)
    return combined_data

def prepare_combined_training_data(corpora: List[str], train_ratio: float = 0.8, seed: int = 42):
    """Process and split combined QA pairs"""
    # Combine all data
    enriched_data = combine_enriched_data(corpora)
    print(f"\nTotal QA pairs: {len(enriched_data)}")
    
    # Process QA pairs
    training_pairs = []
    for item in tqdm(enriched_data, desc="Processing QA pairs"):
        if 'qa_pair' in item and item['qa_pair']:
            qa = item['qa_pair']
            thoughts = "\n".join([f"- {t}" for t in qa.get('thought_process', [])])
            concepts = ", ".join(qa.get('related_concepts', []))
            
            training_item = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant providing clear explanations about data science and programming concepts."
                    },
                    {
                        "role": "user", 
                        "content": qa['question']
                    },
                    {
                        "role": "assistant",
                        "content": f"Let me explain step by step:\n\n{thoughts}\n\nAnswer:\n{qa['answer']}\n\nRelated concepts: {concepts}"
                    }
                ]
            }
            training_pairs.append(training_item)
    
    # Split and save data
    random.seed(seed)
    random.shuffle(training_pairs)
    split_idx = int(len(training_pairs) * train_ratio)
    
    # Create output directory
    output_dir = Path("CorpusQAforTraining/combined")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "valid.jsonl"
    
    save_jsonl(training_pairs[:split_idx], str(train_path))
    save_jsonl(training_pairs[split_idx:], str(val_path))
    print(f"\nSaved {split_idx} training and {len(training_pairs)-split_idx} validation pairs")

if __name__ == "__main__":
    corpora = ["AIML", "BigData", "BizStats", "DataViz", "IntroToBA", "PrescriptiveAnalysis"]
    prepare_combined_training_data(corpora)