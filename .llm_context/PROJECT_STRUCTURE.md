# Project structure

Here is all the relevant information you need to understand the structure of the repository, and for its development.

## `.llm_context`

This folder contains various guides and documentation.

**.llm_context/PROJECT_LOG.md:** A log to write down what you have done.

**.llm_context/PROJECT_STRUCTURE.md:** This document.

**.llm_context/PLAN.md:** Description of the overall project, as well as explanations of the sequence of milestones for how we complete it.

**.llm_context/CONVENTIONS.md:** Various rules on how to make contributions to the repository.

**.llm_context/nblite_README.md:** The README.md of the package `nblite`, which is used to structure the code in this repository.

**.llm_context/adulib_skeleton.md:** A 'skeleton' of the library `adulib`, which is a package that contains various useful LLM-related functions.

**.llm_context/adulib.llm.completions.pct.py:** A copy of the source code of `adulib.llm.completions`, which is useful as you'll be using the library in the development. Note that the code is in the `py:percent` notebook format, and it mixes both module source code (exported using `#|export`) and example code.

## Module structure

```
magpy/
│
├── __init__.py
│
├── core/
│   ├── __init__.py
│   ├── config.py          # set_llm_config and model management
│   ├── llm.py             # LLM API / local model interface
│   └── utils.py           # Shared tools
│
├── extract/
│   ├── __init__.py
│   ├── schema.py          # Field class, schema validation, optional fields
│   └── extractor.py       # extract_structured and helper functions
│
├── ocr/
│   ├── __init__.py
│   └── ocr.py             # Text extraction from images / PDFs
│
├── search/
│   ├── __init__.py
│   └── fuzzy.py           # Fuzzy/semantic search
│
├── entities/
│   ├── __init__.py
│   ├── recognizer.py      # NER and category tagging
│   └── dedup.py           # Entity deduplication
│
├── connect/
│   ├── __init__.py
│   └── linker.py          # Connect entities via relationships mined from text
```

## Tests

```
tests/
│
├── extract/
│   ├── test_extractor.py
│   ├── test_schema.py
│   └── test_cache.py
│
├── core/
│   ├── test_config.py
│   ├── test_llm.py
│   └── test_utils.py
│
├── ocr/
│   └── test_ocr.py
│
├── search/
│   └── test_fuzzy.py
│
├── entities/
│   ├── test_recognizer.py
│   └── test_dedup.py
│
├── connect/
│   └── test_linker.py
```