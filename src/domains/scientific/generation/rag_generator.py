"""
Strict RAG Answer Generator
===========================
Generates highly targeted, grounded answers with citations using Qwen2-VL.
"""

import os
import torch
from PIL import Image

class RAGGenerator:
    """Manages context preparation and execution of strict multimodal generations."""

    @staticmethod
    def generate_strict(query: str, retrieved_pages: list[dict], model, processor) -> dict:
        """Assembles prompt and renders answer with Qwen2-VL."""
        context_texts = []
        context_images = []
        sources = []

        base = os.getenv("RAG_BASE_DIR", "")

        for i, page in enumerate(retrieved_pages[:3]):
            text = page.get("text", "").strip()[:600]
            title = page.get("paper_title", "Unknown")
            page_num = page.get("page_num", 0)
            doc_id = page.get("doc_id", "")
            fused_score = page.get("fused_score", 0.0)

            context_texts.append(
                f"[Paper {i+1}: {title} | arXiv:{doc_id} | Page {page_num}]\n{text}"
            )

            sources.append({
                "paper_title": title,
                "arxiv_id": doc_id,
                "page_num": page_num,
                "fused_score": round(fused_score, 4),
                "arxiv_url": f"https://arxiv.org/abs/{doc_id}",
                "text_snippet": text[:200]
            })

            img_path = page.get("image_path", "")
            if base and img_path:
                img_path = os.path.join(base, img_path)

            if img_path and os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((448, 448), Image.LANCZOS)
                    context_images.append(img)
                except Exception:
                    pass

        context_str = "\n\n---\n\n".join(context_texts)
        retrieval_conf = retrieved_pages[0].get("fused_score", 0.0) if retrieved_pages else 0.0

        # STRICT PROMPT
        prompt = f"""You are a STRICT document Q&A assistant.

STRICT RULES:
1. ONLY use information from the CONTEXT PAPERS below
2. NEVER use your own training knowledge
3. If the answer is NOT found in context, say EXACTLY:
   "NOT_IN_DOCUMENTS: This question is not covered in the provided papers."
4. Always cite: paper name + page number for every claim
5. Be technical and specific

CONTEXT FROM PAPERS:
{context_str}

QUESTION: {query}

ANSWER (from papers only — cite paper + page for every point):"""

        content = []
        for img in context_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        try:
            text_input = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = processor(
                text=text_input,
                images=context_images if context_images else None,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=450,
                    do_sample=False,
                    repetition_penalty=1.1
                )

            input_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[0][input_len:]
            answer = processor.decode(new_tokens, skip_special_tokens=True).strip()

            is_from_docs = "NOT_IN_DOCUMENTS" not in answer
            if not is_from_docs:
                answer = answer.replace("NOT_IN_DOCUMENTS:", "").strip()

        except Exception as e:
            answer = f"Generation error: {e}"
            is_from_docs = False

        return {
            "answer": answer,
            "retrieval_conf": round(retrieval_conf, 4),
            "is_from_docs": is_from_docs,
            "sources": sources
        }
