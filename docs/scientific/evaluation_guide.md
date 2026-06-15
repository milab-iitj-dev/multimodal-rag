# Evaluation Guide

Methodology for evaluating the Scientific Multimodal RAG system's answer quality, retrieval accuracy, and end-to-end performance.

---

## Ground Truth Creation

### Format

Create a JSON file with question-answer pairs and source page references:

```json
[
  {
    "question": "What is the Vision Transformer?",
    "answer": "The Vision Transformer (ViT) applies a pure transformer architecture to image classification by splitting an image into fixed-size patches, linearly embedding them, and processing them with standard transformer encoder blocks.",
    "source_pages": ["2010.11929_page_1", "2010.11929_page_2"],
    "difficulty": "easy",
    "category": "architecture"
  },
  {
    "question": "How does the patch embedding work in ViT?",
    "answer": "ViT extracts 16×16 patches from the input image, flattens them, and projects them to a fixed dimension using a linear layer. These patch embeddings are then augmented with position embeddings.",
    "source_pages": ["2010.11929_page_2"],
    "difficulty": "medium",
    "category": "methodology"
  }
]
```

### Recommended Test Set

| Category | # Questions | Difficulty | Focus |
|---|---|---|---|
| Architecture | 3 | Easy | High-level model descriptions |
| Methodology | 3 | Medium | Specific mechanisms (attention, patches) |
| Figures/Tables | 2 | Hard | Visual content questions |
| Comparison | 2 | Medium | Differences between models |
| **Total** | **10** | | |

### Guidelines

1. **Answer from the papers only** — Ground truth answers should be verifiable from the indexed papers
2. **Include source pages** — Every answer must cite specific pages for retrieval evaluation
3. **Vary difficulty** — Mix easy factual questions with harder analysis questions
4. **Cover visual content** — Include questions about figures and tables to test ColPali
5. **Avoid ambiguity** — Questions should have clear, unambiguous answers

---

## Running Evaluation

### CLI

```bash
python scripts/evaluate.py \
    --ground-truth data/evaluation/ground_truth.json \
    --output data/evaluation/results.json \
    --metrics all
```

### Programmatic

```python
from scripts.evaluate import Evaluator

evaluator = Evaluator(
    ground_truth_path="data/evaluation/ground_truth.json",
    pipeline_config="configs/pipeline_config.yaml",
)

results = evaluator.run()
print(f"Mean Reciprocal Rank: {results['mrr']:.3f}")
print(f"Answer Similarity: {results['answer_similarity']:.3f}")
print(f"Self-Check Pass Rate: {results['check_pass_rate']:.3f}")
```

### Streamlit Evaluation Tab

Use the **Evaluation** tab in the Streamlit app to upload ground truth and run evaluation interactively.

---

## Metrics

### Retrieval Metrics

| Metric | Description | Target |
|---|---|---|
| **MRR** (Mean Reciprocal Rank) | Average of reciprocal ranks of first correct source | ≥ 0.7 |
| **Recall@5** | Fraction of questions where correct source is in top 5 | ≥ 0.9 |
| **Recall@10** | Fraction of questions where correct source is in top 10 | ≥ 0.95 |
| **NDCG@5** | Normalised discounted cumulative gain at 5 | ≥ 0.7 |

### Generation Metrics

| Metric | Description | Target |
|---|---|---|
| **ROUGE-L** | Longest common subsequence overlap | ≥ 0.3 |
| **BERTScore F1** | Semantic similarity via BERT embeddings | ≥ 0.7 |
| **Attribution Rate** | Fraction of answers with valid citations | ≥ 0.8 |
| **Faithfulness Score** | Average keyword overlap ratio | ≥ 0.4 |

### Self-Check Metrics

| Metric | Description | Target |
|---|---|---|
| **Pass Rate** | Fraction of answers passing all 3 checks | ≥ 0.7 |
| **Attribution Pass** | Fraction with citation markers | ≥ 0.8 |
| **Faithfulness Pass** | Fraction with overlap ≥ 0.3 | ≥ 0.7 |
| **Confidence Pass** | Fraction with confidence ≥ 0.6 | ≥ 0.8 |

---

## Interpreting Results

### Good Results
- **MRR ≥ 0.7**: Retrieval finds the right source within the top 2 results on average
- **Attribution ≥ 0.8**: Most answers include proper citations
- **Self-check pass ≥ 0.7**: The majority of answers pass all verification levels

### Common Failure Modes

| Symptom | Likely Cause | Fix |
|---|---|---|
| Low MRR | Poor retrieval; wrong fusion weights | Adjust colpali/scincl weights; increase top_k |
| Low attribution | VLM not citing sources | Strengthen citation instruction in system prompt |
| Low faithfulness | Hallucination; VLM adds external knowledge | Lower temperature; strengthen "no hallucination" rule |
| Low confidence | VLM is uncertain; insufficient context | Increase top_k; add more context to the prompt |
| High latency | Too many retries | Lower confidence_threshold; reduce max_retries |

### A/B Testing

Compare configurations by running evaluation with different settings:

```bash
# Baseline: 0.7/0.3 fusion
python scripts/evaluate.py --config configs/retrieval_config.yaml

# Variant: 0.5/0.5 fusion
python scripts/evaluate.py --config configs/retrieval_config_5050.yaml
```

Record results in a comparison table and select the configuration that maximises MRR and self-check pass rate.
