#!/usr/bin/env python3
"""
Batch document enrichment script.

This script processes multiple documents from an input directory, enriches them
with summaries and hierarchical headings, and saves the results to an output directory.

Features:
- Supports any document format (PDF, JSON, etc.)
- Automatically detects if a file is already a DoclingDocument JSON
- Fixes heading hierarchy and adds section summaries
- Handles errors gracefully without stopping the entire process
- Shows progress for both file-level and document-level processing
"""

import os
import sys
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument
from mellea.backends import model_ids
from tqdm import tqdm

from docling_agent.agents import DoclingEnrichingAgent, logger


def load_document(file_path: Path) -> DoclingDocument | None:
    """
    Load a document from a file path.

    First tries to load as a DoclingDocument JSON. If that fails,
    converts the file using DocumentConverter.

    Args:
        file_path: Path to the document file

    Returns:
        DoclingDocument if successful, None otherwise
    """
    try:
        # Try loading as DoclingDocument JSON first
        doc = DoclingDocument.load_from_json(file_path)
        logger.info(f"Loaded as DoclingDocument JSON: {file_path.name}")
        return doc
    except Exception as json_exc:
        logger.debug(f"Not a DoclingDocument JSON, attempting conversion: {json_exc}")

        # Try converting with DocumentConverter
        try:
            converter = DocumentConverter()
            conv_result = converter.convert(file_path)

            if conv_result.status == ConversionStatus.SUCCESS:
                logger.info(f"Successfully converted: {file_path.name}")
                return conv_result.document
            else:
                logger.error(f"Conversion failed for {file_path.name}: {conv_result.status}")
                return None

        except Exception as conv_exc:
            logger.error(f"Error converting {file_path.name}: {conv_exc}")
            return None


def enrich_document(
    document: DoclingDocument,
    agent: DoclingEnrichingAgent,
    file_name: str,
) -> DoclingDocument | None:
    """
    Enrich a document with summaries and hierarchical headings.

    Args:
        document: The document to enrich
        agent: The enriching agent to use
        file_name: Name of the file (for logging)

    Returns:
        Enriched document if successful, None otherwise
    """
    try:
        # Use the _summarize_items operation directly with explicit parameters
        # This ensures hierarchical headings and adds summaries
        enriched_doc = agent.run(
            task="",  # Empty task since we're using explicit operations
            document=document,
            operations=["summarize_items"],  # Use the operation directly
        )
        logger.info(f"Successfully enriched: {file_name}")
        return enriched_doc

    except Exception as exc:
        logger.error(f"Error enriching {file_name}: {exc}")
        return None


def get_input_files(input_dir: Path, pattern: str = "*") -> list[Path]:
    """
    Get all files from input directory matching the pattern.

    Args:
        input_dir: Input directory path
        pattern: Glob pattern for file matching (default: all files)

    Returns:
        List of file paths
    """
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    # Get all files (not directories) matching the pattern
    files = [f for f in input_dir.glob(pattern) if f.is_file()]

    # Sort for consistent processing order
    return sorted(files)


def process_documents(
    input_dir: Path,
    output_dir: Path,
    model_id=model_ids.OPENAI_GPT_OSS_20B,
    file_pattern: str = "*",
) -> dict[str, int]:
    """
    Process all documents in the input directory.

    Args:
        input_dir: Directory containing input documents
        output_dir: Directory to save enriched documents
        model_id: Model identifier for the enriching agent
        file_pattern: Glob pattern for file matching

    Returns:
        Dictionary with processing statistics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of input files
    input_files = get_input_files(input_dir, file_pattern)

    if not input_files:
        logger.warning(f"No files found in {input_dir} matching pattern '{file_pattern}'")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    logger.info(f"Found {len(input_files)} file(s) to process")

    # Initialize the enriching agent
    agent = DoclingEnrichingAgent(model_id=model_id, tools=[])

    # Statistics
    stats = {"total": len(input_files), "success": 0, "failed": 0, "skipped": 0}

    # Process each file with progress bar
    for file_path in tqdm(input_files, desc="Processing documents", unit="file"):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {file_path.name}")
        logger.info(f"{'=' * 60}")

        # Define output path (same name but with .json extension)
        output_path = output_dir / f"{file_path.stem}.json"

        # Skip if output already exists (optional - remove this check if you want to overwrite)
        if output_path.exists():
            logger.info(f"Output already exists, skipping: {output_path.name}")
            stats["skipped"] += 1
            continue

        # Step 1: Load the document
        logger.info("Step 1/3: Loading document...")
        document = load_document(file_path)

        if document is None:
            logger.error(f"Failed to load document: {file_path.name}")
            stats["failed"] += 1
            continue

        # Step 2: Enrich the document
        logger.info("Step 2/3: Enriching document (this may take a while)...")
        enriched_doc = enrich_document(document, agent, file_path.name)

        if enriched_doc is None:
            logger.error(f"Failed to enrich document: {file_path.name}")
            stats["failed"] += 1
            continue

        # Step 3: Save the enriched document
        logger.info("Step 3/3: Saving enriched document...")
        try:
            enriched_doc.save_as_json(filename=output_path)
            logger.info(f"Successfully saved: {output_path.name}")
            stats["success"] += 1
        except Exception as exc:
            logger.error(f"Error saving {output_path.name}: {exc}")
            stats["failed"] += 1

    return stats


def main():
    """Main entry point for the script."""
    # Configuration
    input_dir = Path("examples/data/papers")  # Change this to your input directory
    output_dir = Path("scratch/example_06")  # Change this to your output directory
    model_id = model_ids.OPENAI_GPT_OSS_20B
    file_pattern = "*"  # Process all files; use "*.pdf" for PDFs only, etc.

    # You can also get these from command line arguments
    if len(sys.argv) > 1:
        input_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    if len(sys.argv) > 3:
        file_pattern = sys.argv[3]

    logger.info("=" * 60)
    logger.info("Batch Document Enrichment")
    logger.info("=" * 60)
    logger.info(f"Input directory:  {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"File pattern:     {file_pattern}")
    logger.info(f"Model:            {model_id}")
    logger.info("=" * 60)

    # Process all documents
    stats = process_documents(
        input_dir=input_dir,
        output_dir=output_dir,
        model_id=model_id,
        file_pattern=file_pattern,
    )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Processing Summary")
    logger.info("=" * 60)
    logger.info(f"Total files:      {stats['total']}")
    logger.info(f"Successfully processed: {stats['success']}")
    logger.info(f"Failed:           {stats['failed']}")
    logger.info(f"Skipped:          {stats['skipped']}")
    logger.info("=" * 60)

    if stats["success"] > 0:
        logger.info(f"\nEnriched documents saved to: {output_dir}")


if __name__ == "__main__":
    main()
