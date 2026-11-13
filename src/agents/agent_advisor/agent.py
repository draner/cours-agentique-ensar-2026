"""Agent conseiller d'orientation académique : LangGraph + RAG + Ollama."""

import os
import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.core.base_agent import BaseAgentPlugin
from src.core.ingestion import get_vector_store
from src.core.state import AgentState

from .tools import format_domain_response, process_domain_logic


class CustomAgentPlugin(BaseAgentPlugin):
    """Agent spécialisé dans les recommandations d'UE et de parcours."""

    @property
    def name(self) -> str:
        return "advisor"

    @property
    def description(self) -> str:
        return (
            "Conseiller d'orientation académique. "
            "Recommande des unités d'enseignement et des parcours "
            "selon les intérêts, compétences et objectifs professionnels "
            "des étudiants. Compare les cours, présente les prérequis "
            "et les crédits ECTS à partir des syllabus officiels."
        )

    def create_node(self) -> Callable[[AgentState], dict[str, Any]]:
        """Créer le nœud LangGraph de l'agent advisor."""

        ollama_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )
        model_name = os.getenv(
            "OLLAMA_LLM_MODEL",
            "qwen2.5:3b",
        )

        llm = ChatOllama(
            base_url=ollama_url,
            model=model_name,
            temperature=0.1,
            num_predict=500,
        )

        vector_store = None

        def node(state: AgentState) -> dict[str, Any]:
            nonlocal vector_store

            last_message = state["messages"][-1]
            user_query = str(last_message.content)

            if vector_store is None:
                vector_store = get_vector_store(reindex_if_empty=False)

            # 1. Recherche RAG ciblée.
            # Le filtre S8 est appliqué seulement si l'étudiant
            # mentionne explicitement ce semestre.
            search_kwargs: dict[str, Any] = {"k": 2}

            semester_match = re.search(
                r"\bS([1-9]|1[0-2])\b",
                user_query,
                flags=re.IGNORECASE,
            )

            if semester_match:
                semester = semester_match.group(0).upper()
                search_kwargs["filter"] = {"semestre": semester}

            relevant_docs = []

            try:
                retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
                relevant_docs = retriever.invoke(user_query)
            except Exception:
                relevant_docs = []

            # 2. Préparer les informations documentaires.
            rag_context = (
                "\n\n---\n\n".join(
                    (
                        f"Source : {doc.metadata.get('source', 'inconnue')}\n"
                        f"Code UE : {doc.metadata.get('code_ue', 'INCONNU')}\n"
                        f"Semestre : {doc.metadata.get('semestre', 'INCONNU')}\n"
                        f"Contenu :\n{doc.page_content}"
                    )
                    for doc in relevant_docs
                )
                if relevant_docs
                else "Aucun document pertinent trouvé."
            )

            # 3. Outils Python déterministes.
            tool_results = process_domain_logic(
                user_query,
                relevant_docs,
            )

            # 4. Prompt spécialisé pour l'orientation.
            system_prompt = (
                "Tu es ADVISOR, un conseiller d'orientation "
                "académique pour une école d'ingénieurs.\n\n"
                "MISSIONS :\n"
                "1. Comprendre le projet professionnel de l'étudiant.\n"
                "2. Identifier les UE pertinentes dans les syllabus.\n"
                "3. Expliquer le lien entre les UE et les objectifs "
                "de l'étudiant.\n"
                "4. Présenter les prérequis lorsqu'ils sont documentés.\n"
                "5. Comparer les UE si la question le demande.\n\n"
                "RÈGLES STRICTES :\n"
                "- Réponds uniquement à partir des données fournies.\n"
                "- N'invente aucun cours, code UE, prérequis ou ECTS.\n"
                "- Cite les noms des fichiers sources.\n"
                "- Les ECTS sont calculés par les outils Python.\n"
                "- Ne présente pas une recommandation comme "
                "une décision officielle d'admission.\n"
                "- Si les données sont insuffisantes, indique-le.\n"
                "- Réponds en français, de manière concise.\n\n"
                f"CONTEXTE RAG :\n{rag_context}\n\n"
                f"RÉSULTATS DES OUTILS :\n{tool_results}"
            )

            messages = [
                SystemMessage(content=system_prompt),
                last_message,
            ]

            # 5. Synthèse par le modèle local.
            try:
                response = llm.invoke(messages)
                response_text = str(response.content)
            except Exception:
                response_text = format_domain_response(tool_results)

            sources = sorted(
                {str(doc.metadata["source"]) for doc in relevant_docs if doc.metadata.get("source")}
            )

            if sources:
                response_text += "\n\nSources consultées : " + ", ".join(sources)

            final_message = AIMessage(content=f"[{self.name}] {response_text}")

            return {
                "messages": [final_message],
                "current_agent": self.name,
            }

        return node
