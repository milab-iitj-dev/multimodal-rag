"""Quick verification of retrieval metrics consistency."""
from evaluation.metrics.retrieval_metrics import compute_retrieval_metrics

# Simulate: 1 relevant doc at rank 1, 349 total gold docs
results = [{
    "retrieved_ids": ["doc1", "doc2", "doc3", "doc4", "doc5"],
    "gold_ids": ["doc1"] + [f"other_{i}" for i in range(348)],
    "query_mode": "text_only",
    "query_id": "q1",
    "query_text": "Is there cardiomegaly?",
    "finding": "cardiomegaly",
}]

m = compute_retrieval_metrics(results, k_values=[1, 3, 5])
a = m["aggregate"]
d = m["diagnostics"][0]

print("=== Consistency check: 349 gold docs, doc1 at rank 1 ===")
print(f"  Recall@1 = {a['recall@1']}  (expect 1.0)")
print(f"  Recall@3 = {a['recall@3']}  (expect 1.0)")
print(f"  Recall@5 = {a['recall@5']}  (expect 1.0)")
print(f"  MRR      = {a['mrr']}  (expect 1.0)")
print(f"  nDCG@1   = {a['ndcg@1']}  (expect 1.0)")
print(f"  Diag hit@1={d['hit@1']} hit@3={d['hit@3']} RR={d['reciprocal_rank']}")
print(f"  Rel vec (first 5): {d['relevance_vector'][:5]}")
print(f"  Gold count: {d['num_gold_docs']}")

print()
# If any Recall != 1.0, the old set-overlap code is running!
if a["recall@1"] < 1.0:
    print("BUG: recall@1 < 1.0 => old set-overlap code is being used!")
    print("  Expected: 1.0 (binary hit)")
    print(f"  Got: {a['recall@1']} = 1/{d['num_gold_docs']} (set-overlap)")
else:
    print("OK: All metrics consistent (binary Hit@k)")
