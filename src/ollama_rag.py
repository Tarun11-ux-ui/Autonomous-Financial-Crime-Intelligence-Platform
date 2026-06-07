import json
import os
import re
import urllib.error
import urllib.request

from src.retriever_layer import FraudRetriever, classify_query_type, extract_account_id_from_query


DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
ACCOUNT_TOKEN_PATTERN = re.compile(r"\b[A-Z0-9_][A-Z0-9_\-]{3,}\b")


def _request_json(url, method="GET", payload=None, timeout=5):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def summarize_ollama_integration(base_url=DEFAULT_OLLAMA_BASE_URL, model=DEFAULT_OLLAMA_MODEL):
    """
    Check whether Ollama is reachable and whether the requested model appears installed.
    """
    try:
        response = _request_json(f"{base_url}/tags", method="GET", timeout=3)
        models = response.get("models", [])
        model_names = {item.get("name", "") for item in models} | {item.get("model", "") for item in models}
        installed = model in model_names or any(name.startswith(f"{model}:") for name in model_names)
        return {
            "ollama_available": True,
            "ollama_base_url": base_url,
            "ollama_model": model,
            "ollama_model_installed": installed,
            "ollama_model_count": len(models),
        }
    except Exception as exc:
        return {
            "ollama_available": False,
            "ollama_base_url": base_url,
            "ollama_model": model,
            "ollama_model_installed": False,
            "ollama_model_count": 0,
            "ollama_error": str(exc),
        }


def _context_heading(row):
    relation = row.get("context_relation", "general_match")
    relation_map = {
        "exact_match": "Primary Account Context",
        "connected_match": "Explicitly Connected Account Context",
        "community_match": "Same Community Support Context",
        "general_match": "General Support Context",
    }
    return relation_map.get(relation, "Retrieved Context")


def _relation_guidance(row):
    relation = row.get("context_relation", "general_match")
    relation_map = {
        "exact_match": "This is the exact account named in the query.",
        "connected_match": "This account is explicitly listed in the target account's connected_risky_accounts field.",
        "community_match": "This account shares the same community as the target account but is not explicitly confirmed as directly connected.",
        "general_match": "This is a similar retrieved case and should not be treated as a direct link unless stated in the context.",
    }
    return relation_map.get(relation, "Use only the stated facts in this context.")


def _build_context_block(retrieved_rows):
    context_blocks = []
    for idx, row in enumerate(retrieved_rows, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"[{_context_heading(row)} {idx}]",
                    f"Document ID: {row.get('retriever_document_id', 'N/A')}",
                    f"Relation Guidance: {_relation_guidance(row)}",
                    f"Target Account: {row.get('target_account_id', 'N/A')}",
                    f"Account ID: {row.get('account_id', 'N/A')}",
                    f"Risk Score: {row.get('risk_score', 'N/A')}",
                    f"Primary Action: {row.get('primary_action', 'N/A')}",
                    f"Impact Priority: {row.get('impact_priority_score', 'N/A')}",
                    f"Retriever Score: {row.get('retriever_score', 'N/A')}",
                    f"Brief: {row.get('investigator_brief', '')}",
                    f"Connected Accounts: {row.get('connected_risky_accounts', '')}",
                ]
            )
        )
    return "\n\n".join(context_blocks)


def _parse_json_answer(answer_text):
    text = str(answer_text or "").strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _format_structured_answer(answer_json):
    sections = []
    field_labels = [
        ("direct_answer", "Direct Answer"),
        ("key_evidence", "Key Evidence"),
        ("connected_evidence", "Connected Evidence"),
        ("recommended_action", "Recommended Follow-Up Action"),
    ]

    for key, label in field_labels:
        value = answer_json.get(key, "")
        if isinstance(value, list):
            value = "; ".join(str(item).strip() for item in value if str(item).strip())
        value = str(value).strip()
        if value:
            sections.append(f"{label}: {value}")
    return "\n\n".join(sections).strip()


def _extract_account_tokens(text):
    return set(ACCOUNT_TOKEN_PATTERN.findall(str(text or "").upper()))


def _grounding_failure_reason(answer_text, target_account_id, retrieved_rows):
    normalized_target = str(target_account_id or "").strip().upper()
    answer_upper = str(answer_text or "").upper()
    allowed_accounts = {
        str(row.get("account_id", "")).strip().upper()
        for row in retrieved_rows
        if str(row.get("account_id", "")).strip()
    }
    connected_accounts = {
        str(row.get("account_id", "")).strip().upper()
        for row in retrieved_rows
        if row.get("context_relation") == "connected_match" and str(row.get("account_id", "")).strip()
    }

    if normalized_target and normalized_target not in answer_upper:
        return "target_account_missing"

    if "CONFUSED USER:" in answer_upper:
        return "contaminated_prompt_echo"

    if normalized_target and "CONNECTED EVIDENCE:" in answer_upper:
        connected_section = answer_upper.split("CONNECTED EVIDENCE:", 1)[1]
        connected_section = connected_section.split("RECOMMENDED FOLLOW-UP ACTION:", 1)[0]
        if normalized_target in connected_section:
            return "target_account_misused_in_connected_evidence"
        if connected_accounts and not any(account in connected_section for account in connected_accounts):
            return "connected_account_missing_from_connected_evidence"

    mentioned_accounts = _extract_account_tokens(answer_text)
    unexpected_accounts = sorted(
        account for account in mentioned_accounts
        if account.startswith("ACCT_") and account not in allowed_accounts
    )
    if unexpected_accounts:
        return f"unexpected_accounts:{', '.join(unexpected_accounts)}"

    return ""


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _split_reasons_from_brief(brief_text):
    text = str(brief_text or "")
    match = re.search(r"Top reasons:\s*(.+?)(?:\.\s*Connected risky accounts:|$)", text, re.IGNORECASE)
    if not match:
        return []
    return [item.strip() for item in match.group(1).split(";") if item.strip()]


def _build_grounded_fallback_answer(query, query_type, target_account_id, retrieved_rows):
    if not retrieved_rows:
        return "Insufficient retrieved context to answer this query reliably."

    primary_row = next(
        (row for row in retrieved_rows if row.get("context_relation") == "exact_match"),
        retrieved_rows[0],
    )
    primary_account = str(primary_row.get("account_id", target_account_id or "the target account")).strip()
    primary_score = _safe_float(primary_row.get("risk_score", 0))
    primary_action = str(primary_row.get("primary_action", "ALLOW_WITH_MONITORING")).strip()
    primary_brief = str(primary_row.get("investigator_brief", "")).strip()
    primary_reasons = _split_reasons_from_brief(primary_brief)
    prevented_amount = _safe_float(primary_row.get("impact_priority_score", 0))

    connected_rows = [
        row for row in retrieved_rows
        if row.get("context_relation") == "connected_match"
    ]

    if query_type in {"account_why", "account_summary", "account_lookup"}:
        direct_answer = (
            f"Account {primary_account} is suspicious because it is classified as high fraud risk "
            f"with a model risk score of {primary_score:.1f} and an action recommendation of {primary_action}."
        )
        if primary_reasons:
            key_evidence = "; ".join(primary_reasons)
        else:
            key_evidence = primary_brief or "The retrieved primary account context indicates elevated fraud risk."

        if connected_rows:
            connected_evidence = "; ".join(
                f"{row.get('account_id', 'N/A')} is explicitly listed as a connected risky account for {primary_account}"
                for row in connected_rows
            )
        else:
            connected_evidence = "No explicitly connected supporting accounts were retrieved for this query."

        recommended_action = (
            f"Proceed with {primary_action} for {primary_account} and review any explicitly connected risky accounts."
            if connected_rows else
            f"Proceed with {primary_action} for {primary_account} based on the primary account evidence."
        )

        return "\n\n".join(
            [
                f"Direct Answer: {direct_answer}",
                f"Key Evidence: {key_evidence}",
                f"Connected Evidence: {connected_evidence}",
                f"Recommended Follow-Up Action: {recommended_action}",
            ]
        )

    return primary_brief or "Insufficient retrieved context to answer this query reliably."


class OllamaRAGAssistant:
    """
    Local RAG answer layer backed by the retriever and Ollama chat API.
    """

    def __init__(self, submission_df, model=DEFAULT_OLLAMA_MODEL, base_url=DEFAULT_OLLAMA_BASE_URL):
        self.submission_df = submission_df
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.retriever = FraudRetriever(submission_df)

    def answer_query(self, query, top_k=5, timeout=30):
        retrieved = self.retriever.search(query, top_k=top_k)
        context_block = _build_context_block(retrieved)
        query_type = classify_query_type(query)
        target_account_id = extract_account_id_from_query(query)

        system_prompt = (
            "You are a fraud investigation assistant. "
            "Answer only from the retrieved case context. "
            "If the context is insufficient, clearly say so. "
            "Be concise, factual, and action-oriented. "
            "Never claim two accounts are connected unless the retrieved context explicitly says they are connected. "
            "Treat same-community accounts as supporting context, not proof of direct linkage. "
            "Do not mix evidence from one account into another account's explanation."
        )

        if query_type in {"account_why", "account_summary", "account_lookup"} and target_account_id:
            user_prompt = (
                f"User query: {query}\n"
                f"Target account: {target_account_id}\n\n"
                "Instructions:\n"
                "1. Use the Primary Account Context as the main evidence for the answer.\n"
                "2. Use Explicitly Connected Account Context only as supporting evidence.\n"
                "3. If Same Community Support Context appears, describe it only as same-community support and not as a direct link.\n"
                "4. If the retrieved context does not prove a relationship, say so plainly.\n"
                "5. Do not mention accounts that are not in the retrieved context.\n\n"
                "6. Do not state that the target account is connected to itself.\n"
                "7. The connected evidence section must mention only explicitly connected accounts, not the target account.\n\n"
                f"Retrieved case context:\n{context_block}\n\n"
                "Return valid JSON only with exactly these keys:\n"
                "{\n"
                '  "direct_answer": "...",\n'
                '  "key_evidence": ["...", "..."],\n'
                '  "connected_evidence": ["...", "..."],\n'
                '  "recommended_action": "..."\n'
                "}\n"
                "Do not add markdown. Do not add commentary outside the JSON object."
            )
        else:
            user_prompt = (
                f"User query: {query}\n\n"
                f"Retrieved case context:\n{context_block}\n\n"
                "Provide:\n"
                "1. direct answer\n"
                "2. key evidence\n"
                "3. recommended follow-up action"
            )

        payload = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": 0,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if query_type in {"account_why", "account_summary", "account_lookup"} and target_account_id:
            payload["format"] = "json"

        try:
            response = _request_json(
                f"{self.base_url}/chat",
                method="POST",
                payload=payload,
                timeout=timeout,
            )
            raw_answer = response.get("message", {}).get("content", "")
            answer = raw_answer
            grounding_status = "llm_raw"
            fallback_reason = ""

            if query_type in {"account_why", "account_summary", "account_lookup"} and target_account_id:
                parsed_answer = _parse_json_answer(raw_answer)
                if parsed_answer:
                    answer = _format_structured_answer(parsed_answer)

                fallback_reason = _grounding_failure_reason(answer, target_account_id, retrieved)
                if not parsed_answer:
                    fallback_reason = fallback_reason or "invalid_json_response"

                if fallback_reason:
                    answer = _build_grounded_fallback_answer(
                        query=query,
                        query_type=query_type,
                        target_account_id=target_account_id,
                        retrieved_rows=retrieved,
                    )
                    grounding_status = "fallback_grounded"
                else:
                    grounding_status = "llm_grounded"

            return {
                "query": query,
                "model": self.model,
                "answer": answer,
                "retrieved_contexts": retrieved,
                "grounding_status": grounding_status,
                "fallback_reason": fallback_reason,
                "ollama_response_meta": {
                    "done": response.get("done"),
                    "done_reason": response.get("done_reason"),
                    "eval_count": response.get("eval_count"),
                    "prompt_eval_count": response.get("prompt_eval_count"),
                },
            }
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            return {
                "query": query,
                "model": self.model,
                "error": f"HTTP {exc.code}: {error_body}",
                "retrieved_contexts": retrieved,
            }
        except Exception as exc:
            return {
                "query": query,
                "model": self.model,
                "error": str(exc),
                "retrieved_contexts": retrieved,
            }
