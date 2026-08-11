"""System prompt (persona + guardrails) and context formatting for the LLM call."""

SYSTEM_PROMPT = """You are the friendly virtual receptionist for Theale Dental Practice.
Answer ONLY using the information given to you in the Context below.
If the answer is not in the Context, say you're not sure and suggest the visitor
contact the practice directly - do not guess or make anything up.
Never give clinical or diagnostic advice; for any medical question, tell the visitor
to book an appointment with the practice.
Keep answers short, warm, and to the point."""

FEE_DISCLAIMER = (
    "\n\n(Prices are as listed on the practice website and may have changed - "
    "please confirm with the practice.)"
)


def build_context(hits: list[dict]) -> str:
    if not hits:
        return "No relevant information found."
    lines = []
    for h in hits:
        meta = h["metadata"]
        lines.append(f"[{meta.get('page_type')}] {meta.get('title')} ({meta.get('url')}):\n{h['text']}")
    return "\n\n".join(lines)


def has_fee_chunk(hits: list[dict]) -> bool:
    return any(h["metadata"].get("page_type") == "fee" for h in hits)


def postprocess_answer(answer: str, hits: list[dict]) -> str:
    if has_fee_chunk(hits) and "confirm with the practice" not in answer.lower():
        answer += FEE_DISCLAIMER
    return answer
