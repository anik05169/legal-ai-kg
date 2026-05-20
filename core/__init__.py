"""
Legal AI GraphRAG — Core Package
=================================
Public API surface for the Legal AI Knowledge Graph pipeline.

Usage:
    from core import LegalGraphRAG, build_infrastructure, generate_interactive_graph
"""

from .rag_pipeline import LegalGraphRAG
from .kg_indexer import build_infrastructure
from .visualize_kg import generate_interactive_graph
from .data_loader import get_cuad_contracts

__all__ = [
    "LegalGraphRAG",
    "build_infrastructure",
    "generate_interactive_graph",
    "get_cuad_contracts",
]
