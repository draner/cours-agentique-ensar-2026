
"""Outils déterministes de l'agent conseiller d'orientation."""

from typing import Any

from langchain_core.documents import Document


def process_domain_logic(
    query: str,
    retrieved_docs: list[Document],
) -> dict[str, Any]:
    """Extraire les UE et calculer leurs ECTS sans doublons."""

    courses: dict[str, dict[str, Any]] = {}

    for doc in retrieved_docs:
        metadata = doc.metadata

        code = str(metadata.get("code_ue", "INCONNU"))
        source = str(metadata.get("source", "inconnue"))
        semestre = str(metadata.get("semestre", "INCONNU"))
        filiere = str(metadata.get("filiere", "INCONNUE"))

        # Plusieurs sections peuvent appartenir à la même UE.
        # Le fichier source permet d'éviter de compter deux fois une UE.
        if source in courses:
            continue

        raw_ects = metadata.get("ects", 0)

        try:
            ects = int(raw_ects)
            if ects < 0:
                ects = 0
        except (ValueError, TypeError):
            ects = 0

        courses[source] = {
            "code_ue": code,
            "source": source,
            "semestre": semestre,
            "filiere": filiere,
            "ects": ects,
        }

    ue_list = list(courses.values())
    total_ects = sum(course["ects"] for course in ue_list)

    return {
        "status": "success" if ue_list else "no_results",
        "question": query,
        "documents_analyzed": len(retrieved_docs),
        "ue_retrouvees": ue_list,
        "nombre_ue": len(ue_list),
        "total_ects": total_ects,
        "summary": (
            f"{len(ue_list)} UE retrouvée(s), "
            f"total de {total_ects} crédits ECTS."
        ),
    }


def format_domain_response(tool_results: dict[str, Any]) -> str:
    """Réponse de secours si le modèle Ollama est indisponible."""

    courses = tool_results.get("ue_retrouvees", [])

    if not courses:
        return (
            "Je n'ai trouvé aucune UE dans les documents consultés. "
            "Je ne peux pas formuler de recommandation fiable."
        )

    lines = [
        "Voici les UE retrouvées dans les syllabus :",
    ]

    for course in courses:
        lines.append(
            f"- {course['code_ue']} "
            f"({course['semestre']}, {course['ects']} ECTS) "
            f"— Source : {course['source']}"
        )

    lines.append(
        f"Total des crédits des UE retrouvées : "
        f"{tool_results['total_ects']} ECTS."
    )

    return "\n".join(lines)

