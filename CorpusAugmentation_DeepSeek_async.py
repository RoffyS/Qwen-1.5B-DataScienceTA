import os
import json
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
import asyncio
from contextlib import asynccontextmanager

@dataclass
class KnowledgeUnit:
    content: str
    code_snippets: List[str]
    formulas: List[str]
    related_concepts: List[str]

class ContentProcessor:
    def __init__(self):
        self.code_pattern = r'```[\w]*\n(.*?)\n```'
        self.formula_pattern = r'\$(.*?)\$'

    def extract_code_snippets(self, content: str) -> List[str]:
        return re.findall(self.code_pattern, content, re.DOTALL)

    def extract_formulas(self, content: str) -> List[str]:
        return re.findall(self.formula_pattern, content)

    def create_knowledge_unit(self, section: Dict) -> KnowledgeUnit:
        content = section['content']
        code_snippets = self.extract_code_snippets(content)
        formulas = self.extract_formulas(content)

        return KnowledgeUnit(
            content=content,
            code_snippets=code_snippets,
            formulas=formulas,
            related_concepts=[]
        )

class AsyncDeepSeekEnricher:
    def __init__(self, model_name: str = "default", client_id: int = 0):
        self.client = AsyncOpenAI(
            base_url="https://api.deepseek.com/v1", 
            api_key="your_api_key"
        )
        self.model = model_name
        self.client_id = client_id
        self.max_length = 8192
        self.semaphore = asyncio.Semaphore(1)
        self.system_prompt = r"""You are a helpful assistant that generates structured QA pairs for fine tuning smaller models based on corpus related to data science and programming.
Return a JSON object in this EXACT format:
{
    "question": "clear question",
    "thought_process": ["step 1", "step 2"],
    "answer": "brief answer",
    "type": "concept/code/application",
    "related_concepts": ["concept1"]
}"""

    def _validate_format(self, data: Dict) -> bool:
        """Validate response format"""
        required = {
            "question": str,
            "thought_process": list,
            "answer": str,
            "type": str,
            "related_concepts": list
        }
        return all(isinstance(data.get(k), t) for k, t in required.items())

    def _clean_response(self, content: str) -> str:
        """Clean DeepSeek response for JSON parsing"""
        try:
            # Remove any non-JSON content
            content = (content
                .replace("```json", "")
                .replace("```", "")
                .replace("<|im_end|>", "")
                .strip())
            
            # Find JSON boundaries
            start = content.find('{')
            end = content.rfind('}') + 1
            if start == -1 or end == 0:
                return ""
                
            # Clean JSON string
            json_str = content[start:end]
            
            # Handle escapes and formatting
            json_str = (json_str
                .replace('\n', ' ')
                .replace('\r', ' ')
                .replace('\t', ' ')
                .replace('\\', '')
                .replace('\\"', '"'))
                
            return json_str.strip()
            
        except Exception as e:
            print(f"Cleaning error: {str(e)}")
            return ""

    async def _generate_response(self, prompt: str) -> Optional[str]:
        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=1.0,
                    max_tokens=self.max_length
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"API error: {str(e)}")
                return None

    async def generate_qa_pairs(self, ku: KnowledgeUnit) -> List[Dict]:
        prompts = self._create_enrichment_prompts(ku)
        tasks = [self._generate_response(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        
        qa_pairs = []
        for response in responses:
            if response:
                cleaned = self._clean_response(response)
                try:
                    parsed = json.loads(cleaned)
                    if self._validate_format(parsed):
                        qa_pairs.append(parsed)
                except json.JSONDecodeError as e:
                    print(f"JSON error: {str(e)}")
        return qa_pairs
    
    def _create_enrichment_prompts(self, ku: KnowledgeUnit) -> List[str]:
        """Create prompts for QA generation"""
        templates = [
            "Explain this content with step-by-step reasoning.",
            "If code is present, explain its implementation.",
            "Compare the concepts with related approaches.",
            "If code exists, suggest optimizations: {code}.",
            "Apply these concepts to solve real problems."
        ]

        return [
            f"""### Knowledge Unit\n{ku.content}\n
### Task\n{template.format(
    code=ku.code_snippets[0] if ku.code_snippets else ''
)}\n
### Format\n{{
    "question": "detailed question",
    "thought_process": ["step 1", "step 2", "..."],
    "answer": "comprehensive answer",
    "type": "concept/code/application",
    "related_concepts": ["related concept 1", "related concept 2"]
}}"""
            for template in templates
        ]

class AsyncCorpusAugmentor:
    def __init__(self, model_name: str = "default", num_clients: int = 5):
        self.content_processor = ContentProcessor()
        self.enrichers = [
            AsyncDeepSeekEnricher(model_name, i) 
            for i in range(num_clients)
        ]

    async def process_corpus(self, input_path: str, output_path: str, enricher_id: int):
        enricher = self.enrichers[enricher_id]
        print(f"Processing {input_path} with client {enricher_id}")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                corpus = json.load(f)
        except Exception as e:
            print(f"Error loading corpus: {str(e)}")
            return

        tasks = []
        for section in corpus:
            if 'section' in section and 'content' in section['section']:
                ku = self.content_processor.create_knowledge_unit(section['section'])
                tasks.append((section, enricher.generate_qa_pairs(ku)))

        enriched_qa_pairs = []
        for section, future in tqdm(tasks, desc=f"Client {enricher_id}"):
            try:
                qa_pairs = await future
                for qa in qa_pairs:
                    enriched_qa_pairs.append({
                        "source_file": section.get('file', ''),
                        "content": section['section']['content'],
                        "qa_pair": qa
                    })
            except Exception as e:
                print(f"Error in client {enricher_id}: {str(e)}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_qa_pairs, f, indent=2)

async def main():
    model_name = "deepseek-chat"
    augmentor = AsyncCorpusAugmentor(model_name, num_clients=5)
    
    corpora = ["BigData", "BizStats", "DataViz", "IntroToBA", "PrescriptiveAnalysis"]
    tasks = []
    
    for i, corpus_name in enumerate(corpora):
        input_path = f"CorpusProcessed/{corpus_name}_processed_corpus.json"
        output_path = f"CorpusAugmented/{corpus_name}_enriched_qa_dataset.json"
        
        os.makedirs("CorpusAugmented", exist_ok=True)
        print(f"\nAssigning {corpus_name} to client {i}")
        tasks.append(augmentor.process_corpus(input_path, output_path, i))
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
