# %% [markdown]
# # extractor

# %%
#|default_exp extract.extractor

# %%
#|hide
import nblite; from nbdev.showdoc import show_doc; nblite.nbl_export()

# %%
#|export
from typing import Any, Dict, Union, Type, Optional, List
from pathlib import Path
import json
import pymupdf
import io
from datetime import datetime
import inspect

import adulib.llm
from pydantic import BaseModel, Field as PydanticField, create_model
from magpy.core.config import get_llm_config
from magpy.extract.schema import Field, _create_dynamic_model, _pydantic_to_response_format_param


# %%
#|export
def extract_structured(
    text: Optional[str] = None,
    texts: Optional[List[str]] = None,
    path: Optional[Union[str, Path]] = None,
    paths: Optional[List[Union[str, Path]]] = None,
    schema: Dict[str, Union[Type, Field]]|Type[BaseModel] = None,
    temperature: float = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Extract structured data from unstructured text using a target schema.
    
    Args:
        text: Single text string to extract from
        texts: List of text strings to extract from
        path: Single file path to extract from
        paths: List of file paths to extract from. The supported file formats are `.pdf`, `.txt`, and `.md`.
        schema: Target schema defining fields to extract
        
    Returns:
        Dictionary or list of dictionaries with extracted structured data
    """
    if not schema:
        raise ValueError("Schema must be provided")
    
    # Determine input type and normalize to list of texts
    input_texts = []
    
    if text:
        input_texts = [text]
    elif texts:
        if not type(texts) in [list, tuple]: raise ValueError("`texts` argument must be a list or tuple")
        input_texts = texts
    elif path:
        input_texts = [_load_text_from_file(Path(path))]
    elif paths:
        if not type(paths) in [list, tuple]: raise ValueError("`paths` argument must be a list or tuple")
        input_texts = [_load_text_from_file(Path(p)) for p in paths]
    else:
        raise ValueError("Must provide one of: text, texts, path, or paths")
    
    # Create Pydantic model from schema
    pydantic_model = _create_dynamic_model(schema) if not issubclass(schema, BaseModel) else schema
    api_schema = _pydantic_to_response_format_param(pydantic_model)
    
    config = get_llm_config()
    
    results = []
    
    for input_text in input_texts:
        # Create extraction prompt
        prompt = _create_extraction_prompt(input_text)
        
        # Call LLM with structured response format
        # Disable caching when using response_format since Pydantic models can't be pickled
        response = adulib.llm.single(
            prompt=prompt,
            model=config.get('model', 'gpt-4o'),
            temperature=config.get('temperature', 0.1),
            response_format=api_schema,
            **{k: v for k, v in config.items() 
               if k not in ['model', 'temperature', 'cache_enabled', 'response_format']}
        )
        
        # Parse structured response
        try:
            result = pydantic_model.model_validate_json(response)
            results.append(result.model_dump())
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}\nResponse: {response}")
    
    # Return single result or list based on input
    if len(results) == 1 and (text or path):
        return results[0]
    else:
        return results

def _load_text_from_file(file_path: Path) -> str:
    """Load text content from various file formats."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == '.pdf':
        return _extract_text_from_pdf(file_path)
    elif suffix in ['.txt', '.md']:
        return file_path.read_text(encoding='utf-8')
    else:
        # Try to read as text
        try:
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            raise ValueError(f"Unsupported file format: {suffix}")

def _extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from PDF file."""
    pages = []
    pdf_document = pymupdf.open(file_path)   
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        page_text = page.get_text().strip()
        pages.append(f"Page {page_num+1}:\n{page_text}")
    return "\n\n\n".join(pages)

def _create_extraction_prompt(text: str) -> str:
    """Create prompt for structured data extraction."""
    
    prompt = inspect.cleandoc(f"""Extract structured data from the following text. Follow these guidelines:

    1. Extract information that matches the requested fields
    2. For optional fields, include them only if the information is clearly present in the text
    3. For datetime fields, use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
    4. For numerical fields, extract only the numeric value without currency symbols or other formatting
    5. If a required field cannot be found, use null
    6. Be precise and only extract information that is explicitly stated in the text

    TEXT TO ANALYZE:
    {text}""")

    return prompt
