
"""Tests de la logique métier de l'agent advisor."""

from langchain_core.documents import Document

from src.agents.agent_advisor.tools import process_domain_logic


def test_ects_without_duplicates() -> None:
    docs = [
        Document(
            page_content="Introduction au machine learning",
            metadata={
                "code_ue": "IA_301",
                "source": "ia_301.md",
                "semestre": "S8",
                "filiere": "IA",
                "ects": 4,
            },
        ),
        Document(
            page_content="Objectifs du machine learning",
            metadata={
                "code_ue": "IA_301",
                "source": "ia_301.md",
                "semestre": "S8",
                "filiere": "IA",
                "ects": 4,
            },
        ),
    ]

    result = process_domain_logic(
        "Quelles UE choisir pour apprendre le machine learning ?",
        docs,
    )

    assert result["nombre_ue"] == 1
    assert result["total_ects"] == 4
    assert result["documents_analyzed"] == 2


def test_no_documents() -> None:
    result = process_domain_logic(
        "Quelles UE choisir ?",
        [],
    )

    assert result["status"] == "no_results"
    assert result["total_ects"] == 0
