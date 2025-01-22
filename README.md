# Technical Report: Low-Resource Data Science Teaching Assistant Using Qwen-1.5B
Note: The current Qwen-1.5B-DataScienceTA is an early-stage model and is not yet ready for deployment. The model is still in the fine-tuning phase and is being tested for its performance and reliability. The dataset used for fine-tuning is not yet finalized as I am still experimenting with the teacher model and the raw corpora. Also, the corpora are not going to be open-sourced due to privacy concerns.

## Project Overview
- **Objective**: Develop locally-deployable teaching assistant for data science concepts
- **Base Model**: Qwen-2.5-Coder-1.5B-Instruct
- **Method**: LoRA fine-tuning with knowledge distillation
- **Target**: Teaching assistant for conceptual understanding in exam settings

## Data Pipeline

### 1. **Data Collection & Processing**
- **Source Materials**:
  - PDF lecture notes (6 domains)
  - Markdown documentation
  - Jupyter notebooks
- **Extraction**: Regex-based text mining
- **Domains Covered**:
  - AI/ML, Big Data, Business Statistics
  - Data Visualization, Analytics, Prescriptive Analysis

### 2. **Knowledge Distillation**
- **Teacher Model**: DeepSeek v3
- **Process**:
  1. Raw text extraction
  2. QA pair generation with Chain-of-Thought
  3. Concept enrichment
- **Inspired by**: AdaptLLM methodology, where text mining (summarization, word-to-text, natural language inference, common sense reasoning, paragraph detection, text completion) is used to create domain-specific corpora.

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
### 1. Fine-tuning Configuration
- LoRA rank: 8
- Max sequence length: 8192
- Trainable parameters: 0.071% (1.090M/1543.714M)
- Training steps: 2000
- 
### 2. Model Variants
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
- Finetuned model adopts teaching style with clear steps
- Both models respect "no code" instruction
- Finetuned model demonstrates more structured responses
- Base model provides deeper mathematical explanations
- Finetuned model focuses on concept accessibility

## Recommendations

### 1. **Production Deployment**
- Use fused model for teaching scenarios where code snippets are not appropriate and logical reasoning might be inspiring, i.e., conceptual learning, exams
- Use base model for daily use, where both code snippets and explanations are required

### 2. **Future Improvements**
- Strengthen teaching constraints in training
- Experiment with different LoRA ranks
- Improve augmentation of raw corpora by adding both code-heavy and concept-heavy data
- Fine-tune larger models that can capture more complex data science concepts and provide more detailed explanations, which is expected to be deployed in a cloud environment rather than students' local machines.
