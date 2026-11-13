"""Tests de la logique métier de l'agent advisor."""

from langchain_core.documents import Document

from src.agents.agent_advisor.agent import CustomAgentPlugin
from src.agents.agent_advisor.tools import process_domain_logic
from src.core.base_agent import BaseAgentPlugin


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


def test_advisor_contract() -> None:
    agent = CustomAgentPlugin()

    assert isinstance(agent, BaseAgentPlugin)
    assert agent.name == "advisor"
    assert len(agent.description) > 10
    assert callable(agent.create_node())


from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage


@patch("src.agents.agent_advisor.agent.ChatOllama")
@patch("src.agents.agent_advisor.agent.get_vector_store")
def test_advisor_node_execution(
    mock_get_vector_store: MagicMock,
    mock_chat_ollama: MagicMock,
) -> None:
    mock_doc = MagicMock()
    mock_doc.metadata = {
        "code_ue": "SD_322",
        "source": "SD_322_Big_Data_Smart_Data.md",
        "semestre": "S9",
        "filiere": "Intelligence Artificielle & Data Science",
        "ects": 5,
    }
    mock_doc.page_content = "Cours de Big Data."

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_doc]

    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    mock_get_vector_store.return_value = mock_vector_store

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "Je recommande SD_322 pour apprendre le Big Data."
    mock_chat_ollama.return_value = mock_llm

    agent = CustomAgentPlugin()
    node = agent.create_node()

    result = node(
        {
            "messages": [HumanMessage(content="Quelles UE choisir pour apprendre le Big Data ?")],
            "next_agent": None,
            "current_agent": None,
            "context_data": {},
        }
    )

    assert result["current_agent"] == "advisor"
    assert "SD_322" in result["messages"][0].content
    assert "SD_322_Big_Data_Smart_Data.md" in result["messages"][0].content
