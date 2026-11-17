# Comparative RAG Pipeline

An end-to-end Retrieval-Augmented Generation system over a long PDF textbook (*Psychology 2e*, OpenStax, CC BY 4.0), designed to compare **retrieval methods** while holding everything else fixed.

**Pipeline:** PDF parsing (PyMuPDF) -> cleaning and boilerplate filtering -> sentence-based, page-bounded chunking -> embeddings -> cosine-similarity retrieval -> token-budgeted prompt -> answer generation with a local Flan-T5.

## Retrievers compared

| Retriever | Type |
|---|---|
| TF-IDF | sparse, lexical |
| Word2Vec (trained on the book) | averaged word vectors |
| BERT `bert-base-uncased`, mean pooling | contextual, no retrieval fine-tuning |
| `all-mpnet-base-v2` | sentence-transformer for semantic search |

## Evaluation

* **Retrieval**: Hit@5, Precision@5, MRR@10 on 12 questions with rule-based relevance labels (regex groups that a relevant chunk must match).
* **Generation**: grounding of the answer in the retrieved passages (a rough hallucination indicator), expected-term coverage, refusal rate.
* A manual spot-check table for reading answers directly.

The rule-based labels favour lexical matching, which the notebook states explicitly; they are a reproducible proxy, not human judgement.

## Notable implementation detail

Flan-T5 accepts 512 input tokens. A naive prompt gets cut at the end, which removes the question first. `build_prompt` reserves tokens for the instruction and question, then splits the remaining budget evenly across the retrieved passages.

## Run it

Open `comparative_rag_pipeline.ipynb` in Jupyter or Google Colab (a GPU runtime is faster, CPU works) and run all cells. The PDF is downloaded automatically. Set `MAX_PAGES` in the configuration cell for a quick test on part of the book.

```bash
pip install -r requirements.txt
jupyter notebook comparative_rag_pipeline.ipynb
```

## Results

Corpus: 753 pages -> 2,532 chunks after filtering. 12 evaluation questions. The notebook reproduces these tables.

**Retrieval (relevance = rule-based labels)**

| Retriever | Hit@5 | Precision@5 | MRR@10 |
|---|---|---|---|
| TF-IDF | 0.75 | 0.367 | 0.566 |
| Word2Vec | 0.50 | 0.133 | 0.349 |
| BERT (mean pool) | 0.50 | 0.117 | 0.249 |
| all-mpnet-base-v2 | **1.00** | **0.417** | **0.688** |

**Answer quality (Flan-T5-large, same prompt for every retriever)**

| Retriever | Grounding | Term coverage | Refusal rate |
|---|---|---|---|
| TF-IDF | 1.00 | 0.54 | 0.0% |
| Word2Vec | 1.00 | 0.42 | 0.0% |
| BERT (mean pool) | 0.83 | 0.42 | 8.3% |
| all-mpnet-base-v2 | 1.00 | **0.63** | 0.0% |

**Takeaways.** The sentence-transformer retrieves the relevant passage most reliably, even though the keyword-based labels favour lexical matching, and better retrieval carries through to better answers. Mean-pooled BERT without retrieval fine-tuning is the weakest dense option and is the only one with a noticeable refusal rate. With 12 questions, small differences should not be over-interpreted; see the limitations section of the notebook.

## Source text

*Psychology 2e*, OpenStax, licensed CC BY 4.0: https://openstax.org/details/books/psychology-2e
