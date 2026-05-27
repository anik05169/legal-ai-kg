"""
Legal AI GraphRAG — Core Package
=================================
Public API surface for the Legal AI Knowledge Graph pipeline.

Usage:
    from core import LegalGraphRAG, build_infrastructure, generate_interactive_graph
"""

try:
    from .rag_pipeline import LegalGraphRAG
except ImportError:
    LegalGraphRAG = None

try:
    from .kg_indexer import build_infrastructure
except ImportError:
    build_infrastructure = None

try:
    from .visualize_kg import generate_interactive_graph
except ImportError:
    generate_interactive_graph = None

try:
    from .data_loader import get_cuad_contracts
except ImportError:
    get_cuad_contracts = None

__all__ = [
    "LegalGraphRAG",
    "build_infrastructure",
    "generate_interactive_graph",
    "get_cuad_contracts",
]

