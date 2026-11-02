"""Prompt templates for grounded generation, citation verification, and answer completeness judging.

Prompt Version: 1.0.0
"""

PROMPT_VERSION = "1.0.0"

# --- 1. Grounded Generation Prompt ---

GROUNDED_SYSTEM_PROMPT = """You are an expert technical assistant answering questions based solely on internal documentation.

STRICT INSTRUCTIONS:
1. Answer the user question ONLY using the factual context blocks provided below.
2. For EVERY factual claim or statement you make, cite the supporting context block using bracketed numbers like [1] or [2] placed IMMEDIATELY after the claim (e.g., "The API Gateway validates tokens [1].").
3. DO NOT cite context numbers that are not provided in the prompt.
4. DO NOT extrapolate, assume, or introduce external knowledge.
5. If the context does not contain sufficient facts to answer the question, do not speculate. You MUST respond with:
"INSUFFICIENT_CONTEXT: The provided internal documentation does not contain enough information to reliably answer this question."
"""

def format_context_blocks(chunks: list) -> str:
    """Format retrieved chunks as numbered context blocks for generation."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        source = c.metadata.get("source_file", "unknown")
        heading = c.metadata.get("section_heading", "Overview")
        page = c.metadata.get("page_number")
        page_info = f", Page {page}" if page and page > 0 else ""
        header = f"[{i}] (Source: {source}, Section: {heading}{page_info})"
        blocks.append(f"{header}\n{c.text.strip()}")
    return "\n\n".join(blocks)


def build_generation_user_prompt(question: str, context_blocks: str) -> str:
    """Construct user prompt presenting question and numbered context blocks."""
    return f"""Context Blocks:
{context_blocks}

Question:
{question}

Answer with inline bracketed citations [n] after each claim:"""


# --- 2. Citation Verification Prompt ---

CITATION_VERIFY_SYSTEM_PROMPT = """You are a strict citation verification judge.
Your role is to assess whether a specific Claim is directly and factually supported by the Cited Chunk Text.

EVALUATION RUBRIC:
- supported = true: The claim is directly stated in or logically follows from the cited chunk without external assumptions.
- supported = false: The claim contradicts, exaggerates, or is not present in the cited chunk.

Return ONLY a valid JSON object with the following schema:
{
  "supported": boolean,
  "reason": "concise explanation of why the citation is supported or unsupported"
}
"""

def build_citation_verify_prompt(claim: str, cited_text: str) -> str:
    """Construct prompt for verifying a single claim-citation pair."""
    return f"""Claim:
"{claim}"

Cited Chunk Text:
\"\"\"{cited_text}\"\"\"

Provide your verification judgment as JSON:"""


# --- 3. Answer Completeness Prompt ---

COMPLETENESS_SYSTEM_PROMPT = """You are an answer completeness judge.
Your role is to assess whether the Generated Answer fully addresses all parts of the User Question.

EVALUATION RUBRIC:
- Return a score between 0.0 and 1.0:
  * 1.0: The answer thoroughly and accurately addresses all aspects of the question.
  * 0.7 - 0.9: The answer addresses the core parts but misses minor details requested.
  * 0.4 - 0.6: The answer addresses only some aspects of the question.
  * 0.0 - 0.3: The answer fails to answer the question or states insufficient context.

Return ONLY a valid JSON object:
{
  "completeness_score": float,
  "reason": "concise explanation of completeness assessment"
}
"""

def build_completeness_prompt(question: str, answer: str) -> str:
    """Construct prompt to evaluate completeness of answer."""
    return f"""User Question:
"{question}"

Generated Answer:
\"\"\"{answer}\"\"\"

Provide your completeness judgment as JSON:"""

