#!/usr/bin/env python3
"""
Agentic RAG Evaluation Script for ViDoRe V3 Benchmark

This script demonstrates in-context reasoning-driven retrieval using DoclingDocument
and evaluates it against the ViDoRe V3 benchmark. The approach leverages document
structure and AI-generated summaries for intelligent Q&A.

Pipeline:
1. Convert PDFs to DoclingDocument objects (JSON format)
2. Fix heading levels using DoclingEditingAgent
3. Enrich documents with AI-generated summaries using DoclingEnrichingAgent
4. Perform intelligent Q&A using DoclingRAGAgent and evaluate with NDCG@10

Usage:
    python perfs/agentic_rag_eval.py --config perfs/agentic_rag_eval_config.yaml --step 1
    python perfs/agentic_rag_eval.py --config perfs/agentic_rag_eval_config.yaml --step 2
    python perfs/agentic_rag_eval.py --config perfs/agentic_rag_eval_config.yaml --step 3
    python perfs/agentic_rag_eval.py --config perfs/agentic_rag_eval_config.yaml --step 4
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Final, Literal

import pandas as pd
import yaml
from docling.document_converter import DocumentConverter
from docling_core.transforms.serializer.markdown import MarkdownParams
from docling_core.types.doc.document import (
    DoclingDocument,
    ImageRefMode,
    SectionHeaderItem,
)
from tqdm import tqdm

from docling_agent.agents import (
    DoclingEditingAgent,
    DoclingEnrichingAgent,
    ReasoningBasedPageSelector,
    TreeGuidedPageSelector,
)
from docling_agent.backends.factory import create_backend
from docling_agent.task_model import BackendConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Silence verbose HTTP logs from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

# Try to import ranx, but provide fallback if not available
try:
    from ranx import Qrels, Run, evaluate

    HAS_RANX = True
except ImportError:
    HAS_RANX = False
    logger.warning("ranx library not found. Install with: pip install ranx")


MD_PARAMS: Final = MarkdownParams(
    image_mode=ImageRefMode.PLACEHOLDER,
    image_placeholder="",
    escape_underscores=False,
    escape_html=False,
    compact_tables=True,
    traverse_pictures=True,
)


def _count_sentences(text: str) -> int:
    """Count sentences in text, handling abbreviations properly."""
    # Protects: w.w. patterns (U.S., 3.14) and Title. abbreviations (Dr., Mr.)
    parts = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text.strip())
    return len([s for s in parts if s.strip()])


# ReasoningBasedPageSelector and TreeGuidedPageSelector are imported from
# docling_agent.agents (defined in docling_agent/agent/rag.py).


class AgenticRAGEvaluator:
    """Evaluator for agentic RAG approach on ViDoRe V3 benchmark."""

    def __init__(
        self,
        dataset_path: Path,
        output_base_dir: Path,
        backend_config: BackendConfig,
        page_level: bool = False,
        summarization_style: Literal["sentences", "keyphrases"] = "sentences",
        selector_algorithm: Literal["batch", "tree"] = "batch",
        eval_top_k: int = 10,
        eval_batch_size: int = 30,
        eval_early_stopping: float = 0.95,
        eval_max_iterations: int = 8,
    ):
        """Initialize the evaluator.

        Args:
            dataset_path: Path to ViDoRe dataset (e.g., /path/to/vidore_v3_finance_en)
            output_base_dir: Base directory for all outputs
            backend_config: Backend configuration for LLM inference
            page_level: Whether to use page-level summarization (default: False)
            summarization_style: "sentences" stores summaries in meta.summary;
                                 "keyphrases" stores keyword lists in meta.keywords (default: "sentences")
            selector_algorithm: "batch" uses ReasoningBasedPageSelector (flat page batches);
                                 "tree" uses TreeGuidedPageSelector (hierarchical heading traversal,
                                 requires element-level step-3 enrichment) (default: "batch")
            eval_top_k: Maximum pages to retrieve per query (default: 10)
            eval_batch_size: Pages per reasoning iteration for batch selector (default: 30)
            eval_early_stopping: Early stopping threshold for batch selector (default: 0.95)
            eval_max_iterations: Max drill-down iterations for tree selector (default: 8)
        """
        self.dataset_path = Path(dataset_path)
        self.output_base_dir = Path(output_base_dir)
        self.backend_config = backend_config
        self.page_level = page_level
        self.summarization_style: Literal["sentences", "keyphrases"] = summarization_style
        self.selector_algorithm: Literal["batch", "tree"] = selector_algorithm
        self.eval_top_k = eval_top_k
        self.eval_batch_size = eval_batch_size
        self.eval_early_stopping = eval_early_stopping
        self.eval_max_iterations = eval_max_iterations

        # Create backend instance
        self.backend = create_backend(backend_config)

        # Define output directories for each step
        self.step1_dir = self.output_base_dir / "step1_converted"
        self.step2_dir = self.output_base_dir / "step2_hierarchical"
        self.step3_dir = self.output_base_dir / "step3_enriched"
        self.step4_dir = self.output_base_dir / "step4_evaluation"

        # Input directories from dataset
        self.pdfs_dir = self.dataset_path / "pdfs"
        self.queries_dir = self.dataset_path / "queries"
        self.qrels_dir = self.dataset_path / "qrels"

        # Validate dataset structure
        if not self.pdfs_dir.exists():
            raise ValueError(f"PDFs directory not found: {self.pdfs_dir}")
        if not self.queries_dir.exists():
            raise ValueError(f"Queries directory not found: {self.queries_dir}")
        if not self.qrels_dir.exists():
            raise ValueError(f"Qrels directory not found: {self.qrels_dir}")

    def step1_convert_pdfs_to_docling(self) -> None:
        """Step 1: Convert PDF files to DoclingDocument objects in JSON format."""
        logger.info("=" * 80)
        logger.info("STEP 1: Converting PDFs to DoclingDocument (JSON)")
        logger.info("=" * 80)

        self.step1_dir.mkdir(parents=True, exist_ok=True)

        # Get all PDF files
        pdf_files = sorted(self.pdfs_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to convert")

        # Initialize Docling converter
        converter = DocumentConverter()

        # Convert each PDF
        for pdf_path in tqdm(pdf_files, desc="Converting PDFs"):
            try:
                # Skip metadata.csv if present
                if pdf_path.stem == "metadata":
                    continue

                output_path = self.step1_dir / f"{pdf_path.stem}.json"

                # Skip if already converted
                if output_path.exists():
                    logger.info(f"Skipping {pdf_path.name} (already converted)")
                    continue

                logger.info(f"Converting {pdf_path.name}...")
                start_time = time.time()

                # Convert PDF to DoclingDocument
                result = converter.convert(pdf_path)
                document = result.document

                # Save as JSON
                document.save_as_json(output_path)

                elapsed = time.time() - start_time
                logger.info(f"Converted {pdf_path.name} in {elapsed:.2f}s ({len(document.pages)} pages)")

            except Exception as e:
                logger.error(f"Failed to convert {pdf_path.name}: {e}", exc_info=True)

        logger.info(f"Step 1 complete. Output saved to: {self.step1_dir}")

    def step2_fix_heading_levels(self) -> None:
        """Step 2: Fix heading levels using DoclingEditingAgent."""
        logger.info("=" * 80)
        logger.info("STEP 2: Fixing heading levels with DoclingEditingAgent")
        logger.info("=" * 80)

        self.step2_dir.mkdir(parents=True, exist_ok=True)

        # Get all JSON files from step 1
        json_files = sorted(self.step1_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} documents to process")

        # Initialize editing agent with backend
        agent = DoclingEditingAgent(backend=self.backend, tools=[])

        # Process each document
        for json_path in tqdm(json_files, desc="Fixing heading levels"):
            try:
                output_path = self.step2_dir / json_path.name

                # Skip if already processed
                if output_path.exists():
                    logger.info(f"Skipping {json_path.name} (already processed)")
                    continue

                logger.info(f"Processing {json_path.name}...")
                start_time = time.time()

                # Load document
                document = DoclingDocument.load_from_json(json_path)

                # Check if heading levels need fixing
                # Typically, Docling assigns the same level to all headings with PDFs
                heading_levels = set()
                for item, _ in document.iterate_items():
                    if isinstance(item, SectionHeaderItem):
                        heading_levels.add(item.level)

                if len(heading_levels) <= 1:
                    logger.info(f"Document has flat heading structure (levels: {heading_levels}). Fixing with agent...")

                    # Use agent to fix heading levels
                    task = (
                        "Ensure that the section headings have the correct hierarchical levels "
                        "based on the document structure. Analyze the document and adjust heading "
                        "levels so that main sections have lower level numbers and subsections "
                        "have higher level numbers."
                    )
                    document = agent.run(task=task, document=document)

                    # Call _hierarchize to reorganize document elements
                    document._hierarchize()
                    document.validate_tree(document.body, raise_on_error=True)

                    logger.info("Heading levels fixed and document hierarchized")
                else:
                    logger.info(f"Document already has hierarchical structure (levels: {heading_levels})")
                    # Call _hierarchize to reorganize document elements
                    document._hierarchize()
                    document.validate_tree(document.body, raise_on_error=True)

                    logger.info("Document hierarchized")

                # Save the hierarchical document
                document.save_as_json(output_path)
                document.save_as_markdown(output_path.with_suffix(".md"))

                elapsed = time.time() - start_time
                logger.info(f"Processed {json_path.name} in {elapsed:.2f}s")

            except Exception as e:
                logger.error(f"Failed to process {json_path.name}: {e}", exc_info=True)

        logger.info(f"Step 2 complete. Output saved to: {self.step2_dir}")

    def step3_enrich_with_summaries(self) -> None:
        """Step 3: Enrich documents with AI-generated summaries.

        Supports two enrichment levels:
        - Element-level (default): Summarize individual document elements
        - Page-level (--page-level): Create one enrichment entry per page

        Each level supports two styles:
        - "sentences": Full-sentence summaries stored in meta.summary (SummaryMetaField)
        - "keyphrases": Keyword lists stored in meta.keywords (KeywordsMetaField)
        """
        if self.page_level:
            self._enrich_page_level()
        else:
            self._enrich_element_level()

    def _enrich_element_level(self) -> None:
        """Enrich documents with element-level summaries or keyphrases."""
        logger.info("=" * 80)
        logger.info(f"STEP 3: Enriching documents at element level (style={self.summarization_style!r})")
        logger.info("=" * 80)

        self.step3_dir.mkdir(parents=True, exist_ok=True)

        # Get all JSON files from step 2
        json_files = sorted(self.step2_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} documents to enrich")

        # Initialize enriching agent with backend
        agent = DoclingEnrichingAgent(backend=self.backend, tools=[])

        # Process each document
        for json_path in tqdm(json_files, desc="Enriching documents"):
            try:
                output_path = self.step3_dir / json_path.name

                # Skip if already processed
                if output_path.exists():
                    logger.info(f"Skipping {json_path.name} (already enriched)")
                    continue

                logger.info(f"Enriching {json_path.name}...")
                start_time = time.time()

                # Load document
                document = DoclingDocument.load_from_json(json_path)

                if self.summarization_style == "keyphrases":
                    # Extract keyphrases into meta.keywords for each element
                    document = agent._find_search_keywords(document=document)
                else:
                    # Generate sentence summaries into meta.summary for each element
                    document = agent._summarize_items(document=document)

                # Save the enriched document
                document.save_as_json(output_path)

                elapsed = time.time() - start_time
                logger.info(f"Enriched {json_path.name} in {elapsed:.2f}s")

            except Exception as e:
                logger.error(f"Failed to enrich {json_path.name}: {e}", exc_info=True)

        logger.info(f"Step 3 complete. Output saved to: {self.step3_dir}")

    def _enrich_page_level(self) -> None:
        """Enrich documents with page-level summaries or keyphrases."""
        logger.info("=" * 80)
        logger.info(f"STEP 3: Enriching documents at page level (style={self.summarization_style!r})")
        logger.info("=" * 80)

        self.step3_dir.mkdir(parents=True, exist_ok=True)

        # Get all JSON files from step 2
        json_files = sorted(self.step2_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} documents to enrich")

        # Initialize enriching agent
        agent = DoclingEnrichingAgent(backend=self.backend, tools=[])

        # Process each document
        for json_path in tqdm(json_files, desc="Enriching documents"):
            try:
                output_path = self.step3_dir / json_path.name

                # Load document (from output if exists for resume, otherwise from step2)
                if output_path.exists():
                    logger.info(f"Resuming enrichment for {json_path.name}...")
                    document = DoclingDocument.load_from_json(output_path)
                else:
                    logger.info(f"Enriching {json_path.name}...")
                    document = DoclingDocument.load_from_json(json_path)

                start_time = time.time()

                # Use agent's page summarization with save callback for fault tolerance
                def save_after_page(doc: DoclingDocument, page_no: int) -> None:
                    doc.save_as_json(output_path)
                    logger.debug(f"  Saved after page {page_no}")

                document = agent._summarize_pages(
                    document=document,
                    style=self.summarization_style,
                    loop_budget=5,
                    save_callback=save_after_page,
                    document_summary_pages=3,
                )

                # Final save
                document.save_as_json(output_path)

                elapsed = time.time() - start_time
                logger.info(f"Enriched {json_path.name} in {elapsed:.2f}s")

            except Exception as e:
                logger.error(f"Failed to enrich {json_path.name}: {e}", exc_info=True)

        logger.info(f"Step 3 complete. Output saved to: {self.step3_dir}")

    def step4_evaluate_rag(self) -> None:
        """Step 4: Evaluate RAG with NDCG@10 metric on ViDoRe benchmark.

        This method implements chunkless RAG evaluation by:
        1. Loading enriched documents with page and document summaries
        2. For each query, first selecting relevant documents (unbiased)
        3. Then using reasoning model to iteratively select top-K pages
        4. Comparing selected pages with ground truth using NDCG@10 metric
        5. Saving per-query results for resume capability
        """
        logger.info("=" * 80)
        logger.info("STEP 4: Evaluating Agentic RAG with NDCG@10")
        logger.info("=" * 80)

        self.step4_dir.mkdir(parents=True, exist_ok=True)

        # Per-query results file for resume capability
        query_results_file = self.step4_dir / "query_results.jsonl"

        # Load already processed queries if resuming
        processed_queries = set()
        previous_results: dict[str, dict[str, float]] = {}
        previous_times: list[float] = []

        if query_results_file.exists():
            logger.info("Found existing results file, loading processed queries...")
            with open(query_results_file) as f:
                for line in f:
                    result = json.loads(line)
                    query_id = result["query_id"]
                    processed_queries.add(query_id)

                    # Reconstruct retrieval results from previous run
                    if "predicted_pages" in result and not result.get("error"):
                        page_scores = {}
                        for page in result["predicted_pages"]:
                            corpus_id = page["corpus_id"]
                            score = page["score"]
                            page_scores[corpus_id] = score
                        previous_results[query_id] = page_scores

                    # Load previous timing
                    if "processing_time_seconds" in result:
                        previous_times.append(result["processing_time_seconds"])

            logger.info(f"Resuming: {len(processed_queries)} queries already processed")
            logger.info(f"Loaded {len(previous_results)} previous retrieval results")
            logger.info(f"Previous processing time: {sum(previous_times):.2f}s")

        # Load enriched documents
        json_files = sorted(self.step3_dir.glob("*.json"))
        if not json_files:
            logger.error(f"No enriched documents found in {self.step3_dir}")
            logger.error("Please run Step 3 first")
            return

        logger.info(f"Loading {len(json_files)} enriched documents...")
        documents: dict[str, DoclingDocument] = {}
        doc_summaries: dict[str, str] = {}
        for json_path in json_files:
            doc = DoclingDocument.load_from_json(json_path)
            # Use stem without extension as doc_id (e.g., "morgan_stanley_2024")
            doc_id = json_path.stem
            documents[doc_id] = doc
            doc_summaries[doc_id] = self._extract_document_summary(doc)
            logger.info(f"  Loaded {doc_id} ({len(doc.pages)} pages)")

        # Load ViDoRe corpus to get doc_id mapping
        logger.info("Loading corpus data for doc_id mapping...")
        corpus_files = sorted((self.dataset_path / "corpus").glob("*.parquet"))
        corpus_dfs = [pd.read_parquet(f) for f in corpus_files]
        corpus_df = pd.concat(corpus_dfs, ignore_index=True)
        logger.info(f"Loaded {len(corpus_df)} corpus entries")

        # Load ViDoRe queries
        queries_file = self.queries_dir / "test-00000-of-00001.parquet"
        if not queries_file.exists():
            logger.error(f"Queries file not found: {queries_file}")
            return

        queries_df = pd.read_parquet(queries_file)
        logger.info(f"Loaded {len(queries_df)} queries")

        # Load ViDoRe qrels (ground truth)
        qrels_file = self.qrels_dir / "test-00000-of-00001.parquet"
        if not qrels_file.exists():
            logger.error(f"Qrels file not found: {qrels_file}")
            return

        qrels_df = pd.read_parquet(qrels_file)
        logger.info(f"Loaded {len(qrels_df)} qrels entries")

        # Initialize page selector with backend
        if self.selector_algorithm == "tree":
            logger.info("Initializing TreeGuidedPageSelector...")
            page_selector: ReasoningBasedPageSelector | TreeGuidedPageSelector = TreeGuidedPageSelector(
                backend=self.backend,
                k=self.eval_top_k,
                max_iterations=self.eval_max_iterations,
                summarization_style=self.summarization_style,
            )
        else:
            logger.info("Initializing ReasoningBasedPageSelector...")
            page_selector = ReasoningBasedPageSelector(
                backend=self.backend,
                k=self.eval_top_k,
                batch_size=self.eval_batch_size,
                early_stopping_threshold=self.eval_early_stopping,
                summarization_style=self.summarization_style,
            )

        # Process queries and build retrieval results
        logger.info(
            f"Processing queries with {self.selector_algorithm!r} page selector (style={self.summarization_style!r})..."
        )

        # Start with previous results if resuming
        retrieval_results: dict[str, dict[str, float]] = previous_results.copy()

        # Track timing (include previous times)
        total_start_time = time.time()
        query_times = previous_times.copy()

        # Open results file in append mode
        with open(query_results_file, "a") as results_file:
            for idx, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Running queries"):
                query_id = str(row["query_id"])
                query_text = row["query"]

                # Skip if already processed
                if query_id in processed_queries:
                    logger.debug(f"Skipping already processed query {query_id}")
                    continue

                # Log the query at INFO level
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Query {query_id}: {query_text}")
                logger.info(f"{'=' * 80}")

                # Start timing for this query
                query_start_time = time.time()

                try:
                    # Step 1: Select relevant documents (unbiased - no peeking at qrels)
                    relevant_doc_ids = page_selector.select_relevant_documents(
                        query=query_text,
                        documents=documents,
                        doc_summaries=doc_summaries,
                    )

                    # Step 2: For each relevant document, select top pages
                    all_selected_pages: dict[str, list[tuple[int, float]]] = {}
                    page_summaries_by_doc: dict[str, dict[int, str]] = {}

                    for doc_id in relevant_doc_ids:
                        if doc_id not in documents:
                            continue

                        document = documents[doc_id]
                        doc_summary = doc_summaries[doc_id]

                        # Use reasoning-based page selector to find top-K pages
                        selected_pages = page_selector.select_pages(
                            query=query_text,
                            document=document,
                            doc_summary=doc_summary,
                        )

                        if selected_pages:
                            all_selected_pages[doc_id] = selected_pages
                            # Extract page enrichment for potential re-ranking (batch selector only)
                            if isinstance(page_selector, ReasoningBasedPageSelector):
                                page_summaries_by_doc[doc_id] = page_selector._extract_page_enrichment(document)

                    # Step 3: Re-rank pages across documents if multiple documents selected
                    # (only supported by ReasoningBasedPageSelector)
                    if len(all_selected_pages) > 1 and isinstance(page_selector, ReasoningBasedPageSelector):
                        logger.info(
                            f"Multiple documents selected ({len(all_selected_pages)}), re-ranking pages across documents"
                        )
                        reranked_pages = page_selector.rerank_across_documents(
                            query=query_text,
                            all_selected_pages=all_selected_pages,
                            doc_summaries=doc_summaries,
                            page_summaries_by_doc=page_summaries_by_doc,
                        )

                        # Convert reranked results to the expected format
                        page_scores = {}
                        predicted_pages = []
                        for doc_id, page_num, score in reranked_pages:
                            # Find corpus_id for this page
                            matching_rows = corpus_df[
                                (corpus_df["doc_id"] == doc_id)
                                & (corpus_df["page_number_in_doc"] == page_num - 1)  # Convert to 0-indexed
                            ]
                            if not matching_rows.empty:
                                corpus_id = str(matching_rows.iloc[0]["corpus_id"])
                                page_scores[corpus_id] = float(score)
                                predicted_pages.append(
                                    {
                                        "doc_id": doc_id,
                                        "page_num": page_num,
                                        "corpus_id": corpus_id,
                                        "score": float(score),
                                    }
                                )
                    else:
                        # Single document - no need to re-rank
                        page_scores = {}
                        predicted_pages = []
                        for doc_id, pages in all_selected_pages.items():
                            for page_num, score in pages:
                                # Find corpus_id for this page
                                matching_rows = corpus_df[
                                    (corpus_df["doc_id"] == doc_id)
                                    & (corpus_df["page_number_in_doc"] == page_num - 1)  # Convert to 0-indexed
                                ]
                                if not matching_rows.empty:
                                    corpus_id = str(matching_rows.iloc[0]["corpus_id"])
                                    page_scores[corpus_id] = float(score)
                                    predicted_pages.append(
                                        {
                                            "doc_id": doc_id,
                                            "page_num": page_num,
                                            "corpus_id": corpus_id,
                                            "score": float(score),
                                        }
                                    )

                    retrieval_results[query_id] = page_scores

                    # Get ground truth for this query
                    query_qrels = qrels_df[qrels_df["query_id"] == row["query_id"]]
                    ground_truth_pages = []
                    for _, qrel_row in query_qrels.iterrows():
                        corpus_id = str(qrel_row["corpus_id"])
                        score = int(qrel_row["score"])
                        # Find doc_id and page_num for this corpus_id
                        corpus_row = corpus_df[corpus_df["corpus_id"] == int(corpus_id)]
                        if not corpus_row.empty:
                            gt_doc_id = corpus_row.iloc[0]["doc_id"]
                            gt_page_num = int(corpus_row.iloc[0]["page_number_in_doc"]) + 1  # Convert to 1-indexed
                            ground_truth_pages.append(
                                {
                                    "doc_id": gt_doc_id,
                                    "page_num": gt_page_num,
                                    "corpus_id": corpus_id,
                                    "relevance": score,
                                }
                            )

                    # Calculate query processing time
                    query_elapsed = time.time() - query_start_time
                    query_times.append(query_elapsed)

                    # Save per-query result
                    query_result = {
                        "query_id": query_id,
                        "query_text": query_text,
                        "selected_documents": relevant_doc_ids,
                        "predicted_pages": predicted_pages,
                        "ground_truth_pages": ground_truth_pages,
                        "num_predicted": len(predicted_pages),
                        "num_ground_truth": len(ground_truth_pages),
                        "processing_time_seconds": round(query_elapsed, 2),
                    }
                    results_file.write(json.dumps(query_result) + "\n")
                    results_file.flush()  # Ensure it's written immediately

                    logger.info(f"Predicted {len(predicted_pages)} pages from {len(relevant_doc_ids)} document(s)")
                    logger.info(f"Ground truth: {len(ground_truth_pages)} relevant pages")
                    logger.info(f"Processing time: {query_elapsed:.2f}s")

                except Exception as e:
                    query_elapsed = time.time() - query_start_time
                    query_times.append(query_elapsed)

                    logger.error(f"Failed to process query {query_id}: {e}", exc_info=True)
                    retrieval_results[query_id] = {}
                    # Save error result
                    error_result = {
                        "query_id": query_id,
                        "query_text": query_text,
                        "error": str(e),
                        "predicted_pages": [],
                        "ground_truth_pages": [],
                        "processing_time_seconds": round(query_elapsed, 2),
                    }
                    results_file.write(json.dumps(error_result) + "\n")
                    results_file.flush()

        # Calculate total processing time
        current_run_time = time.time() - total_start_time
        total_query_time = sum(query_times)  # Includes previous + current
        avg_time_per_query = total_query_time / len(query_times) if query_times else 0

        logger.info("\n" + "=" * 80)
        logger.info("TIMING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Current run time: {current_run_time:.2f}s ({current_run_time / 60:.2f} minutes)")
        logger.info(f"Total query processing time: {total_query_time:.2f}s ({total_query_time / 60:.2f} minutes)")
        logger.info(f"Queries processed (total): {len(query_times)}")
        logger.info(f"Queries processed (this run): {len(query_times) - len(previous_times)}")
        logger.info(f"Average time per query: {avg_time_per_query:.2f}s")
        if query_times:
            logger.info(f"Min time: {min(query_times):.2f}s")
            logger.info(f"Max time: {max(query_times):.2f}s")

        # Prepare qrels for evaluation
        logger.info("Preparing ground truth qrels...")
        qrels_dict: dict[str, dict[str, int]] = {}
        for _, row in qrels_df.iterrows():
            query_id = str(row["query_id"])
            corpus_id = str(row["corpus_id"])
            score = int(row["score"])

            if query_id not in qrels_dict:
                qrels_dict[query_id] = {}
            qrels_dict[query_id][corpus_id] = score

        # Compute NDCG@10
        logger.info("Computing NDCG@10 metric...")
        logger.info(f"Queries in ground truth: {len(qrels_dict)}")
        logger.info(f"Queries in retrieval results: {len(retrieval_results)}")

        qrels_obj = Qrels(qrels_dict)
        run_obj = Run(retrieval_results)

        # Use make_comparable=True to handle partial results (e.g., when testing with subset of queries)
        # This adds empty results for queries missing from the run and removes those not in qrels
        ndcg_10 = evaluate(qrels_obj, run_obj, metrics=["ndcg@10"], make_comparable=True)

        logger.info("=" * 80)
        logger.info(f"RESULTS: Agentic RAG NDCG@10 = {ndcg_10:.4f}")
        logger.info("=" * 80)

        # Save results
        results = {
            "dataset": self.dataset_path.name,
            "backend_type": self.backend_config.type,
            "reasoning_model": self.backend_config.models.reasoning if self.backend_config.models else "unknown",
            "writing_model": self.backend_config.models.writing if self.backend_config.models else "unknown",
            "num_queries": len(queries_df),
            "num_queries_processed": len(query_times),
            "num_queries_this_run": len(query_times) - len(previous_times),
            "num_documents": len(documents),
            "ndcg@10": float(ndcg_10),
            "current_run_time_seconds": round(current_run_time, 2),
            "current_run_time_minutes": round(current_run_time / 60, 2),
            "total_query_processing_time_seconds": round(total_query_time, 2),
            "total_query_processing_time_minutes": round(total_query_time / 60, 2),
            "average_time_per_query_seconds": round(avg_time_per_query, 2),
            "min_time_per_query_seconds": round(min(query_times), 2) if query_times else 0,
            "max_time_per_query_seconds": round(max(query_times), 2) if query_times else 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        results_file = self.step4_dir / "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {results_file}")

        # Save detailed retrieval results
        detailed_file = self.step4_dir / "retrieval_results.json"
        with open(detailed_file, "w") as f:
            json.dump(retrieval_results, f, indent=2)

        logger.info(f"Detailed retrieval results saved to: {detailed_file}")
        logger.info(f"Step 4 complete. Output saved to: {self.step4_dir}")

    def _extract_document_summary(self, document: DoclingDocument) -> str:
        """Extract a document-level context string from the body meta.

        For style="sentences" this is the prose summary stored in ``body.meta.summary``.
        For style="keyphrases" there is no document-level summary (only page-level
        keywords), so the method falls back to assembling a short string from
        ``body.meta.keywords`` when present, and returns an empty string otherwise.

        Args:
            document: The enriched DoclingDocument

        Returns:
            Document-level context text, or empty string if not found
        """
        if not (hasattr(document, "body") and document.body):
            return ""
        if not (hasattr(document.body, "meta") and document.body.meta):
            return ""

        meta = document.body.meta

        # Try meta.summary first (present for "sentences" style and as doc-level
        # summary from _summarize_pages when style="sentences")
        if isinstance(meta, dict):
            summary_data = meta.get("summary", {})
            if isinstance(summary_data, dict):
                text = summary_data.get("text", "")
                if text:
                    return text
            # Fall back to keywords on body (unlikely but defensive)
            kw_data = meta.get("keywords", {})
            if isinstance(kw_data, dict):
                values = kw_data.get("values", [])
                if values:
                    return "; ".join(str(v) for v in values)
        else:
            if hasattr(meta, "summary") and meta.summary and hasattr(meta.summary, "text"):
                text = meta.summary.text
                if text:
                    return text
            if hasattr(meta, "keywords") and meta.keywords and hasattr(meta.keywords, "values"):
                kw_values = meta.keywords.values  # noqa: PD011
                if kw_values:
                    return "; ".join(str(v) for v in kw_values)

        return ""


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(config).__name__}")

    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Agentic RAG Evaluation on ViDoRe V3 Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with YAML config
  python perfs/agentic_rag_eval.py --config perfs/agentic_rag_eval_config.yaml --step 1

  # Override dataset path
  python perfs/agentic_rag_eval.py --config perfs/agentic_rag_eval_config.yaml --step 2 --dataset /path/to/dataset
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file (see perfs/agentic_rag_eval_config.yaml)",
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4],
        required=True,
        help="Which step to run (1: convert PDFs, 2: fix headings, 3: enrich, 4: evaluate)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to ViDoRe dataset directory (overrides config file)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for all steps (overrides config file)",
    )
    parser.add_argument(
        "--page-level",
        action="store_true",
        help="Use page-level summarization in Step 3 instead of element-level (default: False)",
    )
    parser.add_argument(
        "--summarization-style",
        choices=["sentences", "keyphrases"],
        default=None,
        help=(
            "Style for Step 3 enrichment: 'sentences' stores summaries in meta.summary; "
            "'keyphrases' stores keyword lists in meta.keywords (overrides config file)"
        ),
    )
    parser.add_argument(
        "--selector-algorithm",
        choices=["batch", "tree"],
        default=None,
        help=(
            "Step 4 page selector: 'batch' uses flat batch reasoning (ReasoningBasedPageSelector); "
            "'tree' uses hierarchical heading traversal (TreeGuidedPageSelector, "
            "requires element-level step-3 enrichment) (overrides config file)"
        ),
    )

    args = parser.parse_args()

    # Load configuration from YAML
    config = load_config(args.config)

    # Extract backend configuration
    if "backend" not in config:
        raise ValueError("Config file must contain 'backend' section")

    backend_config = BackendConfig.model_validate(config["backend"])

    # Get dataset path (command line overrides config)
    dataset_path = args.dataset if args.dataset else config.get("dataset")
    if dataset_path is None:
        raise ValueError("Dataset path must be provided via config or --dataset flag")

    # Get output path (command line overrides config)
    output_path = args.output if args.output else config.get("output", Path("scratch/agentic_rag_eval"))

    # Get page-level flag (command line overrides config)
    page_level = args.page_level or config.get("page_level", False)

    # Get summarization style (command line overrides config); validate and narrow the type
    _style_raw = args.summarization_style or config.get("summarization_style", "sentences")
    if _style_raw not in ("sentences", "keyphrases"):
        raise ValueError(f"summarization_style must be 'sentences' or 'keyphrases', got {_style_raw!r}")
    summarization_style: Literal["sentences", "keyphrases"] = _style_raw  # type: ignore[assignment]

    # Get selector algorithm (command line overrides config); validate and narrow the type
    _algo_raw = args.selector_algorithm or config.get("selector_algorithm", "batch")
    if _algo_raw not in ("batch", "tree"):
        raise ValueError(f"selector_algorithm must be 'batch' or 'tree', got {_algo_raw!r}")
    selector_algorithm: Literal["batch", "tree"] = _algo_raw  # type: ignore[assignment]

    # Get evaluation parameters from config
    eval_config = config.get("evaluation", {})
    eval_top_k = eval_config.get("top_k", 10)
    eval_batch_size = eval_config.get("batch_size", 30)
    eval_early_stopping = eval_config.get("early_stopping_threshold", 0.95)
    eval_max_iterations = eval_config.get("max_iterations", 8)

    # Initialize evaluator
    evaluator = AgenticRAGEvaluator(
        dataset_path=dataset_path,
        output_base_dir=output_path,
        backend_config=backend_config,
        page_level=page_level,
        summarization_style=summarization_style,
        selector_algorithm=selector_algorithm,
        eval_top_k=eval_top_k,
        eval_batch_size=eval_batch_size,
        eval_early_stopping=eval_early_stopping,
        eval_max_iterations=eval_max_iterations,
    )

    # Run the specified step
    if args.step == 1:
        evaluator.step1_convert_pdfs_to_docling()
    elif args.step == 2:
        evaluator.step2_fix_heading_levels()
    elif args.step == 3:
        evaluator.step3_enrich_with_summaries()
    elif args.step == 4:
        evaluator.step4_evaluate_rag()


if __name__ == "__main__":
    main()
