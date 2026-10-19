# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "docling>=2.140.0",
#     "pypdf>=6.21.1",
#     "python-docx>=1.2.0",
# ]
# ///
"""
Stage 1: Raw Document (PDF / DOCX) -> Hierarchical Document AST Extractor
-------------------------------------------------------------------------
In a real production pipeline, you never hand pre-sliced JSON clauses to an LLM.
You ingest raw PDFs and DOCX files, extract their structural hierarchy (Sections -> Subsections -> 
Clauses + Embedded Markdown Tables), deterministically regex-extract explicit structural links 
("Section 4.3", "Part #VLV-200-SS"), and emit a clean Document AST JSON for Stage 2 (dLLM Graph Compilation).

Supports:
  1. IBM Docling (`pip install docling`) for layout-aware PDF/DOCX/Table AST extraction.
  2. Built-in zero-dependency fallback (`pypdf` + `python-docx`) so you can run this immediately.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Deterministic regex patterns for zero-cost structural edge discovery before calling the LLM
SECTION_HEADER_REGEX = re.compile(r"^(?:VOLUME\s+[IVX]+|[1-9]\.[0-9])\s+(.+)$")
CLAUSE_HEADER_REGEX = re.compile(r"^([1-9]\.[1-9][0-9]*)\s+(.+)$")
CROSS_REF_REGEX = re.compile(r"(?:Section|SEC-|Clause)\s*([1-9]\.[0-9]+)", re.IGNORECASE)
PART_SKU_REGEX = re.compile(r"#([A-Z0-9]+-[A-Z0-9]+(?:-[A-Z0-9]+)?)")


def extract_deterministic_anchors(text: str, self_clause_id: str) -> Dict[str, List[str]]:
    """
    Extracts hard cross-references (e.g., 'Section 4.3' -> 'SEC-4.3') and part SKUs
    deterministically via regex so the downstream dLLM has pinned boundary anchors.
    """
    raw_refs = CROSS_REF_REGEX.findall(text)
    cross_refs = sorted(
        list({f"SEC-{ref}" for ref in raw_refs if f"SEC-{ref}" != self_clause_id and not ref.endswith(".0")})
    )
    part_skus = sorted(list({f"PART:{sku}" for sku in PART_SKU_REGEX.findall(text)}))
    return {"deterministic_cross_refs": cross_refs, "deterministic_part_skus": part_skus}


def table_to_markdown(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(str(c).strip() for c in rows[0]) + " |"
    sep = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(str(c).strip() for c in r) + " |" for r in rows[1:]]
    return "\n".join([header, sep] + body)


def parse_with_docling(file_path: Path, doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Primary path: Uses Docling's DocumentConverter to parse PDF/DOCX layout trees and tables.
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        return None

    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    md_text = result.document.export_to_markdown()
    return parse_markdown_stream_into_ast(md_text, doc_id, file_path.name)


def parse_docx_into_markdown_stream(file_path: Path) -> str:
    """Extracts headings, paragraphs, and tables in reading order from a DOCX file."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(file_path))
    blocks: List[str] = []

    for child in doc.element.body.iterchildren():
        if child.tag.endswith("p"):
            para = Paragraph(child, doc)
            text = para.text.strip()
            if not text:
                continue
            style_name = para.style.name if para.style else ""
            if "Heading 1" in style_name or "Title" in style_name:
                blocks.append(f"## {text}")
            elif "Heading 2" in style_name:
                blocks.append(f"### {text}")
            else:
                blocks.append(text)
        elif child.tag.endswith("tbl"):
            tbl = Table(child, doc)
            rows = [[cell.text.strip() for cell in row.cells] for row in tbl.rows]
            blocks.append(table_to_markdown(rows))

    return "\n\n".join(blocks)


def parse_pdf_into_markdown_stream(file_path: Path) -> str:
    """Extracts text blocks from PDF pages while stripping running headers/footers."""
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    lines: List[str] = []
    for page in reader.pages:
        raw = page.extract_text() or ""
        for line in raw.splitlines():
            clean = line.strip()
            if not clean:
                continue
            if "RESTRICTED TECHNICAL SPECIFICATION" in clean or "BOREALIS-SPEC-2026-REV4" in clean:
                continue
            lines.append(clean)
    return "\n".join(lines)


def parse_markdown_stream_into_ast(stream_text: str, doc_id: str, source_filename: str) -> Dict[str, Any]:
    """
    State-machine AST builder that segments a document stream into:
      Document -> Parent Sections (1.0, 2.0...) -> Clauses (1.1, 1.2...) + Attached Tables
    """
    ast_doc: Dict[str, Any] = {
        "doc_id": doc_id,
        "source_file": source_filename,
        "sections": []
    }

    current_section: Optional[Dict[str, Any]] = None
    current_clause: Optional[Dict[str, Any]] = None

    for raw_line in stream_text.splitlines():
        line = raw_line.strip().lstrip("#").strip()
        if not line:
            continue

        # Check for top-level section header (e.g., '1.0 GENERAL SITE PARAMETERS...')
        sec_match = re.match(r"^([1-9]\.0)\s+(.+)$", line)
        if sec_match:
            sec_num, sec_title = sec_match.group(1), sec_match.group(2)
            # Determine precedence tier from section header semantics
            tier = 1
            if "EXCURSION" in sec_title.upper() or "OVERRIDE" in sec_title.upper():
                tier = 2
            elif "SAFETY" in sec_title.upper() or "INTERLOCK" in sec_title.upper():
                tier = 3

            current_section = {
                "section_id": f"SEC-{sec_num}",
                "title": sec_title,
                "default_precedence_tier": tier,
                "clauses": []
            }
            ast_doc["sections"].append(current_section)
            current_clause = None
            continue

        # Check for numbered clause header (e.g., '1.1 Facility Classification...' or notice '3.3 Polar-Solvent...')
        clause_match = re.match(r"^([1-9]\.[1-9][0-9]*)\s+([^:]+)(?::\s*(.*))?$", line)
        if clause_match and current_section is not None:
            c_num = clause_match.group(1)
            c_head = clause_match.group(2).strip()
            remainder = (clause_match.group(3) or "").strip()

            current_clause = {
                "clause_id": f"SEC-{c_num}",
                "parent_section_id": current_section["section_id"],
                "precedence_tier": current_section["default_precedence_tier"],
                "heading": c_head,
                "body_blocks": [remainder] if remainder else [],
                "tables_markdown": []
            }
            current_section["clauses"].append(current_clause)
            continue

        # Append content or markdown table rows to the active clause
        if current_clause is not None:
            if line.startswith("|") and line.endswith("|"):
                current_clause["tables_markdown"].append(line)
            else:
                current_clause["body_blocks"].append(line)

    # Finalize clauses: merge paragraphs, attach tables, and run deterministic anchor regexes
    for sec in ast_doc["sections"]:
        for cl in sec["clauses"]:
            prose = " ".join(cl.pop("body_blocks"))
            tbl_lines = cl.pop("tables_markdown")
            table_md = "\n".join(tbl_lines) if tbl_lines else None
            full_text = prose + (f"\n\n[EMBEDDED_SPEC_TABLE]\n{table_md}" if table_md else "")

            anchors = extract_deterministic_anchors(full_text, cl["clause_id"])
            cl["text"] = full_text
            cl["has_table"] = bool(table_md)
            cl["deterministic_cross_refs"] = anchors["deterministic_cross_refs"]
            cl["deterministic_part_skus"] = anchors["deterministic_part_skus"]

    return ast_doc


def compile_raw_directory_to_ast(input_dir: str, output_json_path: str) -> Dict[str, Any]:
    input_path = Path(input_dir)
    compiled_documents = []

    for idx, file_path in enumerate(sorted(input_path.iterdir()), start=1):
        if file_path.suffix.lower() not in {".pdf", ".docx"}:
            continue
        doc_id = f"DOC-0{idx}-{file_path.stem.split('_')[0]}"

        # Try Docling first; fall back to built-in PDF/DOCX structural stream parser
        ast_doc = parse_with_docling(file_path, doc_id)
        if ast_doc is None:
            if file_path.suffix.lower() == ".pdf":
                stream = parse_pdf_into_markdown_stream(file_path)
            else:
                stream = parse_docx_into_markdown_stream(file_path)
            ast_doc = parse_markdown_stream_into_ast(stream, doc_id, file_path.name)

        compiled_documents.append(ast_doc)

    corpus_ast = {
        "schema_version": "2026.4-AST",
        "pipeline_stage": "STAGE_1_DOCUMENT_AST_EXTRACTION",
        "documents": compiled_documents
    }

    with open(output_json_path, "w") as f:
        json.dump(corpus_ast, f, indent=2)

    return corpus_ast


if __name__ == "__main__":
    ast_out = compile_raw_directory_to_ast(
        input_dir="./starter/raw_docs",
        output_json_path="./starter/extracted_document_ast.json"
    )
    total_clauses = sum(len(s["clauses"]) for d in ast_out["documents"] for s in d["sections"])
    print(f"Stage 1 Complete: Extracted {len(ast_out['documents'])} raw documents into {total_clauses} hierarchical AST clauses.")
