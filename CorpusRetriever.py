from abc import ABC, abstractmethod
import os
import re
import glob
import json
import logging
import markdown
from bs4 import BeautifulSoup
import PyPDF2
from typing import Dict, List, Optional, Union
import nbformat
from nbformat.reader import NotJSONError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentExtractor:
    """Handles document text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        try:
            text_content = []
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
            return "\n".join(text_content)
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {str(e)}")
            raise

    @staticmethod
    def extract_text_from_markdown(md_path: str) -> str:
        try:
            with open(md_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
            html = markdown.markdown(md_content)
            text = BeautifulSoup(html, "html.parser").get_text()
            return text
        except Exception as e:
            logger.error(f"Failed to process Markdown {md_path}: {str(e)}")
            raise
    
    @staticmethod
    def extract_text_from_notebook(nb_path: str) -> str:
        try:
            with open(nb_path, 'r', encoding='utf-8') as file:
                nb = nbformat.read(file, as_version=4)
                
            content_blocks = []
            current_code_block = []
            
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    # Add to current code block
                    if cell.source.strip():
                        current_code_block.append(cell.source)
                elif cell.cell_type == 'markdown':
                    # Save any accumulated code block
                    if current_code_block:
                        content_blocks.append(f"```python\n{chr(10).join(current_code_block)}\n```")
                        current_code_block = []
                    # Add markdown content
                    if cell.source.strip():
                        content_blocks.append(cell.source)
            
            # Handle remaining code block
            if current_code_block:
                content_blocks.append(f"```python\n{chr(10).join(current_code_block)}\n```")
            
            return "\n\n".join(content_blocks)
            
        except Exception as e:
            logger.error(f"Failed to process notebook {nb_path}: {str(e)}")
            raise


class TextProcessor(ABC):
    @abstractmethod
    def process(self, content: str) -> List[Dict]:
        pass
    
    def clean_text(self, text: str) -> str:
        # Remove extra whitespace and normalize line endings
        return " ".join(text.split())

class SectionProcessor(TextProcessor):
    def __init__(self, min_section_length: int = 50):
        self.min_section_length = min_section_length
        
    def process(self, content: str) -> List[Dict]:
        # Split by code blocks first
        code_pattern = r'```[\w]*\n(.*?)\n```'
        code_blocks = re.finditer(code_pattern, content, re.DOTALL)
        
        sections = []
        last_end = 0
        
        for match in code_blocks:
            # Add text before code block
            text_before = content[last_end:match.start()].strip()
            if text_before and len(text_before) >= self.min_section_length:
                sections.append({
                    "content": self.clean_text(text_before),
                    "type": "section"
                })
            
            # Add code block
            code = match.group(1).strip()
            if code:
                sections.append({
                    "content": match.group(0),  # Include markers
                    "type": "code"
                })
            
            last_end = match.end()
        
        # Add remaining text
        text_after = content[last_end:].strip()
        if text_after and len(text_after) >= self.min_section_length:
            sections.append({
                "content": self.clean_text(text_after),
                "type": "section"
            })
        
        return sections

class CodeProcessor(TextProcessor):
    def process(self, content: str) -> List[Dict]:
        code_blocks = []
        current_block = []
        in_code_block = False
        
        # Split content into lines for processing
        lines = content.split('\n')
        
        for line in lines:
            # Check for code block markers
            if '```' in line:
                if not in_code_block:
                    # Start of code block
                    in_code_block = True
                    # Remove language identifier if present
                    current_block = []
                else:
                    # End of code block, save if not empty
                    in_code_block = False
                    if current_block:
                        code_blocks.append({
                            "content": '\n'.join(current_block).strip(),
                            "type": "code"
                        })
                    current_block = []
                continue
                
            # Inside code block
            if in_code_block:
                current_block.append(line)
            # Check for indented code
            elif line.startswith('    '):
                current_block.append(line.strip())
            # End of indented block
            elif current_block and not line.strip():
                code_blocks.append({
                    "content": '\n'.join(current_block).strip(),
                    "type": "code"
                })
                current_block = []
        
        # Handle any remaining code block
        if current_block:
            code_blocks.append({
                "content": '\n'.join(current_block).strip(),
                "type": "code"
            })
            
        return code_blocks

    def clean_code(self, code: str) -> str:
        """Clean code while preserving structure"""
        # Remove leading/trailing whitespace
        code = code.strip()
        # Remove markdown code markers
        code = re.sub(r'^```\w*\s*|\s*```$', '', code)
        return code

class CorpusProcessor:
    def __init__(self):
        self.section_processor = SectionProcessor()
        self.code_processor = CodeProcessor()
        self.document_extractor = DocumentExtractor()
    
    def process_document(self, file_path: str) -> List[Dict]:
        # Get content based on file type
        if file_path.endswith('.pdf'):
            content = self.document_extractor.extract_text_from_pdf(file_path)
        elif file_path.endswith('.md'):
            content = self.document_extractor.extract_text_from_markdown(file_path)
        elif file_path.endswith('.ipynb'):
            content = self.document_extractor.extract_text_from_notebook(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
            
        # Process sections and code blocks
        sections = self.section_processor.process(content)
        processed_content = []
        
        for section in sections:
            # Extract code blocks from section
            code_blocks = self.code_processor.process(section['content'])
            
            # Add non-empty sections and code blocks
            if section['content'].strip():
                processed_content.append(section)
            if code_blocks:
                processed_content.extend(code_blocks)
                
        return processed_content

def build_finetuning_dataset(folder_path: str, output_path: str = None) -> List[Dict]:
    """Process corpus files and save to JSON"""
    processor = CorpusProcessor()
    document_extractor = DocumentExtractor()
    dataset = []
    
    try:
        # Process all supported files
        for ext in ['pdf', 'md', 'ipynb']:
            for file_path in glob.iglob(os.path.join(folder_path, '**', f'*.{ext}'), recursive=True):
                try:
                    processed = processor.process_document(file_path)
                    dataset.extend([{
                        "file": file_path,
                        "section": item,
                        "format": ext
                    } for item in processed])
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
        
        # Save dataset if output path is provided
        if output_path and dataset:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            
            # Save to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            logger.info(f"Dataset saved to {os.path.abspath(output_path)}")
            
        return dataset
    
    except Exception as e:
        logger.error(f"Error processing corpus: {str(e)}")
        return []

if __name__ == "__main__":
    corpus_path = "CorpusRepo/PrescriptiveAnalysis"
    output_path = "CorpusProcessed/PrescriptiveAnalysis_processed_corpus.json"
    
    print(f"Processing corpus from {corpus_path}...")
    dataset = build_finetuning_dataset(corpus_path, output_path)
    print(f"Processed {len(dataset)} sections")