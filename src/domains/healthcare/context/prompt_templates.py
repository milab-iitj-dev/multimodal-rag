"""
Prompt templates for different generation scenarios.

Stores structured prompt templates for:
    - Zero-shot VQA (Phase 1)
    - RAG-augmented VQA (Phase 2 — legacy)
    - Grounded RAG VQA (Phase 4 — current)
    - Medical captioning

Design notes:
  - Phase 4 uses the Qwen2.5-VL chat template which handles
    prompt formatting internally via apply_chat_template().
  - The system prompt and evidence are passed as structured
    messages, NOT wrapped in a conversation format.
  - These templates are now ONLY used as fallback for LLaVA
    or for the ContextBuilder's legacy build_prompt() method.

IMPORTANT: The Qwen2.5-VL generator does NOT use these templates.
    It builds messages directly in _build_messages(). These
    templates exist for backward compatibility with LLaVA.
"""

# ------------------------------------------------------------------ #
#  Phase 1: Simple VQA (no retrieval)                                  #
# ------------------------------------------------------------------ #

SIMPLE_VQA_PROMPT = (
    "{question}"
)
"""
Simple VQA prompt — direct question to VLM with image.

Used by LLaVA's _build_prompt() which wraps this as:
    USER: <image>\n{question}\nASSISTANT:

Placeholders:
    {question} — the clinical question
"""

# ------------------------------------------------------------------ #
#  Phase 4: Grounded RAG VQA (current)                                 #
# ------------------------------------------------------------------ #

RAG_VQA_PROMPT = (
    "You are an expert radiologist. Answer the question based ONLY on "
    "the provided evidence and the image. Do NOT add information that "
    "is not supported by the evidence.\n"
    "\n"
    "{context}\n"
    "\n"
    "QUESTION: {question}\n"
    "\n"
    "INSTRUCTIONS:\n"
    "- If the question is yes/no, start your answer with YES or NO.\n"
    "- Cite specific evidence to support your answer.\n"
    "- If the evidence says a finding is ABSENT, your answer must "
    "reflect that.\n"
    "- If evidence is insufficient, say so explicitly.\n"
    "- If the image conflicts with the evidence, state the discrepancy."
)
"""
Grounded RAG VQA prompt — evidence-focused with strict instructions.

Used as fallback when the VLM is LLaVA (not Qwen2.5-VL).
Qwen2.5-VL uses its own structured chat messages instead.

Key changes from Phase 2:
  - "ONLY" instead of "help" — no permission to ignore evidence
  - Explicit instruction to reflect ABSENCE when evidence says so
  - YES/NO instruction for direct answers
  - Conflict handling instruction

Placeholders:
    {context}  — formatted retrieved evidence block (from aggregator)
    {question} — the clinical question
"""

# ------------------------------------------------------------------ #
#  Medical captioning                                                  #
# ------------------------------------------------------------------ #

CAPTION_PROMPT = (
    "Describe all clinically significant findings visible in this "
    "medical image. Include observations about anatomy, pathology, "
    "and any abnormalities."
)
"""
Medical image captioning prompt.

No placeholders — used directly for generating clinical descriptions
of medical images (e.g., chest X-rays).
"""

# ------------------------------------------------------------------ #
#  RAG captioning (with context)                                       #
# ------------------------------------------------------------------ #

RAG_CAPTION_PROMPT = (
    "You are an expert radiologist. Use the following retrieved "
    "clinical evidence from similar cases to help describe this image.\n"
    "\n"
    "{context}\n"
    "\n"
    "Based on the image and the evidence above, describe all clinically "
    "significant findings visible in this medical image."
)
"""
RAG-augmented captioning prompt.

Placeholders:
    {context} — formatted retrieved evidence block
"""
