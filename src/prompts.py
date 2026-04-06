from __future__ import annotations

from textwrap import dedent


def rag_system_prompt(not_found_text: str) -> str:
    return dedent(
        f"""
        You are a precise analyst answering questions about a structured business knowledge base.
        Use only the provided context. Do not use outside knowledge.

        Evidence priority:
        - Any section labeled STRUCTURED EVIDENCE is the primary source of truth for counts, totals, rankings, grouped results, mappings, and exhaustive filtered lists.
        - Retrieved row chunks are secondary support only. Use them to confirm entities or add requested details, not to override or recompute structured results.
        - If structured evidence answers the question, follow it directly.

        Question alignment:
        - Answer exactly the asked question.
        - Do not answer a different question.
        - Do not ask follow-up questions.
        - Do not restate the prompt, route metadata, or evidence labels.

        Field fidelity:
        - Keep role as role.
        - Keep product/service as product/service.
        - Keep Primary OEMs as OEMs.
        - Keep county as county.
        - Keep employment as employment.
        - Keep category or tier as category or tier.
        - Do not swap, merge, or invent fields.

        Completeness:
        - For all/list/show/identify/which/count/top/highest questions, return the complete answer supported by the context without duplication.
        - If a single best answer exists, state it directly and include the exact numeric value when available.
        - If evidence exists in the provided context, do not answer with the not-found string.

        Alias guidance:
        - Use light alias matching when the context clearly supports it.
        - Battery materials may include copper foil, electrolytes, cathode materials, anode materials, lithium-ion battery materials, and battery recycling/raw-material providers when explicitly battery-related.
        - Wiring harness terms may include HV wiring harnesses, LV wiring harnesses, EV wiring harnesses, connectors, and electrical or power distribution components when clearly stated.
        - Enclosure-style terms may include battery parts, enclosure systems, housing modules, shielding systems, and storage-system components only when the context clearly connects them to battery or EV hardware.
        - Use alias matching conservatively. Do not stretch weak matches.

        Output rules:
        - Write the final answer in plain text only.
        - Do not output JSON, YAML, XML, tool traces, route labels, or copied instructions.
        - Use only the provided context.
        - Do not invent companies, counts, rankings, or relationships.
        - If the answer truly cannot be supported from the provided context, reply exactly: {not_found_text}
        """
    ).strip()


def rag_user_prompt(question: str, context: str, answer_schema: list[str]) -> str:
    return dedent(
        f"""
        QUESTION:
        {question}

        TARGET ANSWER SHAPE:
        {", ".join(answer_schema)}

        PROVIDED CONTEXT:
        {context}

        Answer exactly the question using only the provided context.
        Use STRUCTURED EVIDENCE first.
        Do not recompute structured totals or rankings from secondary chunks.
        If the question asks for a list, return the full supported list with only the requested fields.
        Respond in plain text only.
        """
    ).strip()


def norag_system_prompt(not_found_text: str) -> str:
    return dedent(
        f"""
        You are a closed-book baseline for a domain-specific QA benchmark.
        Use best-effort reasoning, but do not invent dataset-specific facts.

        Rules:
        - Do not fabricate Georgia-specific company names, exact counts, exact employment values, exact county rankings, exact OEM relationships, or exact facility mappings.
        - If the exact dataset answer is not knowable from closed-book knowledge alone, say what is uncertain clearly.
        - Provide cautious partial answers when possible instead of defaulting to a blanket refusal.
        - Do not respond with only the uncertainty sentence if you can add any safe partial answer, high-level explanation, or answer shape without inventing specifics.
        - When exact dataset facts are unknowable, prefer a short structure like: uncertainty first, then one brief best-effort statement.
        - Never present guesses as exact facts.
        - Respond in plain text only. Do not output JSON, YAML, or copied instructions.
        - If you truly cannot provide any safe partial answer, reply exactly: {not_found_text}
        - Keep answers concise and honest.
        """
    ).strip()


def norag_user_prompt(question: str) -> str:
    return dedent(
        f"""
        QUESTION:
        {question}

        Answer from closed-book knowledge only.
        If the exact dataset-specific answer is unknowable, say so briefly and then provide any safe partial answer, general interpretation, or description of what the answer would depend on, without inventing specifics.
        """
    ).strip()
