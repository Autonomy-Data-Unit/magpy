# Project Milestones

**MagPy** is a modular Python toolkit that empowers data journalists and researchers to extract structured intelligence from messy, unstructured documents. Built for usability and extensibility, it bridges the gap between large language models (LLMs) and real-world investigative workflows.

The package provides:

- Schema-based extraction of structured data from natural language.
- Fuzzy and semantic search across document corpora.
- OCR capabilities for scanned PDFs and images.
- Named entity recognition and deduplication.
- Tools to infer and map relationships between entities across texts.

`magpy` is designed to be LLM-agnostic (OpenAI, Claude, or local models like LLaMA), reproducible (via built-in caching), and lightweight enough for Python users with modest experience. It supports core use cases like document tagging, political donation analysis, FOI log processing, and network mapping of influence or corruption.

In what follows we outline the development milestones for the `magpy` Python package, organized by module. Each milestone builds on the previous to create a powerful, modular toolkit for extracting structured intelligence from unstructured text, tailored for data journalists.

## Milestones

### Milestone 1: `core` and `extract`

**Objective**: Establish the foundational infrastructure and implement the structured data extraction pipeline.

#### Modules
- `core`: Configuration, LLM integration, and caching.
- `extract`: Schema definition, extraction functions, and text-to-structure logic.

#### Tasks
- Implement `Field` class supporting type enforcement, optional fields, and nesting.
- Develop `extract_structured()` to extract structured data from one or more unstructured texts using a target schema.
- Add `set_llm_config()` for model selection, temperature, caching, and API key management.
- Build a local cache to reduce redundant LLM usage and improve reproducibility.
- Write usage examples and unit tests for schema and extraction logic.

#### Outcome
Users can define field-level schemas and extract structured information from text with reproducibility and cost-efficiency.


### Milestone 2: `search`

**Objective**: Add semantic and fuzzy search capabilities across large corpora using LLM prompts or embeddings.

#### Module
- `search`: Prompt-driven and embedding-enhanced search tools.

#### Tasks
- Create `fuzzy_search()` that takes a natural language prompt and corpus, returning relevant matches.
- Support both LLM-based reasoning and embedding similarity for search.
- Handle large document sets with batching or streaming results.
- Add examples like “Find documents mentioning political entities” or “Highlight corruption-related passages.”

#### Outcome
Journalists can explore and filter text corpora with natural-language queries, improving investigative efficiency.



### Milestone 3: `ocr`

**Objective**: Add functionality to extract text from scanned PDFs, images, and other non-digital sources.

#### Module
- `ocr`: Optical character recognition from documents.

#### Tasks
- Integrate Tesseract or multi-modal LLMs (e.g. GPT-4o, LLaVA) for OCR.
- Normalize and clean OCR output for compatibility with `extract_structured()`.
- Support batch processing of images and PDFs.
- Implement caching of OCR output to avoid reprocessing.

#### Outcome
Users can incorporate unstructured, image-based documents into their data extraction workflows.



### Milestone 4: `entities`

**Objective**: Provide named entity recognition (NER), categorization, and deduplication to enhance structure and linkage.

#### Module
- `entities`: Tools for extracting and cleaning named entities.

#### Tasks
- Build `recognize_entities()` for identifying people, organizations, locations, etc.
- Implement `deduplicate_entities()` to resolve entity variants and aliases.
- Use both traditional models (e.g., spaCy) and LLM-backed recognition.
- Evaluate and test with real-world datasets (e.g., company filings, FOI logs).

#### Outcome
Users can reliably extract and clean entity information from text, enabling entity-centric analyses and reports.



### Milestone 5: `connect`

**Objective**: Enable entity relationship extraction from text and link entities into contextual networks.

#### Module
- `connect`: Functions to infer and describe relationships between entities.

#### Tasks
- Create `link_entities()` to take a list of entities and related texts, returning inferred links.
- Use proximity, co-occurrence, and LLM inference to detect relationships.
- Support output as structured JSON and exportable graph formats (e.g. GraphML, CSV edge lists).
- Add examples of donor–recipient links, corporate structures, or thematic relationships.

#### Outcome
Users can construct contextual maps of influence, association, or control by connecting entities through relationship inference.


## General notes

- Make use of the library `adulib`. In particular, use `adulib.llm`. It contains various useful tools to do LLM-related functions (see `adulib_skeleton.md` and `adulib.llm.completions.pct.py`). Do not reinvent the wheel.

