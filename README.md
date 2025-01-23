# Technical Report: Low-Resource Data Science Teaching Assistant Using Qwen-1.5B
https://huggingface.co/RoffyS/Qwen-1.5B-DataScienceTA

Note: The current Qwen-1.5B-DataScienceTA is an early-stage model and is not yet ready for deployment. The model is still in the fine-tuning phase and is being tested for its performance and reliability. The dataset used for fine-tuning has not yet been finalized as I am still experimenting with the teacher model and the raw corpora. Also, the corpora are not going to be open-sourced due to privacy concerns.

## Project Overview
- **Objective**: Develop locally-deployable teaching assistant for data science concepts
- **Base Model**: Qwen-2.5-Coder-1.5B-Instruct
- **Method**: LoRA fine-tuning with guided data generation of a larger model
- **Target**: Teaching assistant for conceptual understanding in exam settings

## Data Pipeline

### 1. **Data Collection & Processing**
- **Source Materials**:
  - PDF lecture notes (6 domains)
  - Markdown documentation
  - Jupyter notebooks
- **Extraction**: Regex-based text mining
- **Domains Covered**:
  - AI/ML,
  - Big Data,
  - Business Statistics
  - Data Visualization,
  - Business Analytics,
  - Prescriptive Analytics

### 2. **Data Generation & Enrichment**
- **Large Model Assistance**: DeepSeek v3
- **Process**:
  1. Raw text extraction from course materials
  2. QA pair generation using DeepSeek v3
  3. Chain-of-Thought augmentation
  4. Concept enrichment and linking
- **Inspired by**: AdaptLLM's approach to domain adaptation https://github.com/microsoft/LMOps/tree/main/adaptllm

### 3. **Training Format**
```json
{
    "qa_pair": {
        "question": "Data science concept",
        "thought_process": ["Step 1", "Step 2", "Step 3"],
        "answer": "Conceptual explanation",
        "related_concepts": ["concept1", "concept2"]
    }
}
```

## Technical Implementation
### 1. **Fine-tuning Configuration**
- LoRA rank: 8
- Max sequence length: 8192
- Trainable parameters: 0.071% (1.090M/1543.714M)
- Training steps: 2000

### 2. **Model Variants**
- Base Model (Qwen-2.5-Coder-1.5B-Instruct)
- Fused Model (Qwen-1.5B-DataScienceTA)

## Results Analysis
### 1. **Model Behavior Comparison**
When using identical system prompt and temperature settings:
```bash
mlx_lm.generate \
--model base/fused \
--max-tokens 4000 \
--temp 0.8 \
--system-prompt "You are a data science expert to answer conceptual questions. You should not generate code." \
--prompt "Explain logit regression."
```

| Aspect | Base Model | Finetuned Fused Model |
|--------|------------|----------------------|
| Response Style | Academic | Teaching |
| Math Formulation | LaTeX format | Plain text |
| Explanation Structure | Detailed paragraphs | Step-by-step bullets |
| Token Count | ~450 | ~250 |
| Focus Areas | Mathematical derivation | Intuitive understanding |
| Answer Pattern | Comprehensive lecture | Digestible chunks |
| Related Concepts | Embedded in text | Explicitly listed |

### 2. **Key Observations**
- Base model maintains academic rigor with LaTeX equations
- Fine-tuned model adopts a teaching style with clear steps
- Both models respect "no code" instruction
- Finetuned model demonstrates more structured responses
- Base model provides deeper mathematical explanations
- Fine-tuned model focuses on concept accessibility

## Recommendations

### 1. **Production Deployment**
- Use fused model for teaching scenarios where code snippets are not appropriate and logical reasoning might be inspiring, i.e., conceptual learning, exams
- Use base model for daily use, where both code snippets and explanations are required

### 2. **Future Improvements**
- Strengthen teaching constraints in training
- Experiment with different LoRA ranks
- Improve augmentation of raw corpora by adding both code-heavy and concept-heavy data to develop a general-purpose model
- Fine-tune larger models that can capture more complex data science concepts and provide more detailed explanations, which is expected to be deployed in a cloud environment rather than students' local machines.

# Getting Started

## Prerequisites
- Python 3.10+
- DeepSeek API key
- Device with Apple Silicon chip and MLX environment
  - FYI, I used **MacBook Pro with M3 Max chip, 64GB RAM**
- Input documents in PDF/MD/IPYNB formats

## Project Structure
```
DS-fine-tuning-Qwen/
├── 

CorpusRetriever.py


├── 

CorpusAugmentation_DeepSeek_async.py


├── 

TrainingDataGenerationCombined.py


├── CorpusRepo (a folder for your documents)/
├── CorpusProcessed
├── CorpusAugmented/
└── CorpusQAforTraining/
```

### Step 1: Raw Corpus Extraction
CorpusRetriever.py 

This script:
- Processes documents using regex patterns
- Extracts relevant text segments

### Step 2: Corpus Augmentation
CorpusAugmentation_DeepSeek_async.py

This script:
- Processes raw text using DeepSeek v3 asynchronously with parallel API calls
- Generates QA pairs with Chain-of-Thought
- Enriches with related concepts

### Step 3: Training Data Preparation
TrainingDataGenerationCombined.py 

This script:
- Combines all augmented corpora
- Formats into JSONL for MLX
- Creates train/validation splits

## Common Issues
1. Rate limiting with DeepSeek API
   - Solution: Adjust batch_size and delay
2. Memory usage during fine-tuning
   - Solution: Process in smaller batches, see mlx_lm.lora documentation for recommendations
3. File encoding issues
   - Solution: Ensure UTF-8 encoding

