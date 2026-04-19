# i23-2554-NLP-Assignment2

**CS-4063: Natural Language Processing — Assignment 2**  
**FAST NUCES | BBC Urdu Neural NLP Pipeline | PyTorch from Scratch**

---

## Repository Structure

```
i23-2554_Assignment2_DS-C/
├── i23-2554_Assignment2_DS-C.ipynb   # Main notebook (all cells executed)
├── report.pdf                          # 2–3 page report (Times New Roman, 12pt)
├── README.md                           # This file
├── embeddings/
│   ├── tfidf_matrix.npy               # TF-IDF term–document matrix (376 × 10002)
│   ├── ppmi_matrix.npy                # PPMI co-occurrence matrix (2002 × 2002)
│   ├── embeddings_w2v.npy             # Averaged Word2Vec embeddings ½(V+U)
│   └── word2idx.json                  # Vocabulary mapping (word → index)
├── models/
│   ├── bilstm_pos.pt                  # BiLSTM POS tagger (fine-tuned)
│   ├── bilstm_ner.pt                  # BiLSTM NER tagger with CRF
│   └── transformer_cls.pt             # Transformer topic classifier
└── data/
    ├── pos_train.conll                # POS training data (CoNLL format)
    ├── pos_test.conll                 # POS test data
    ├── ner_train.conll                # NER training data (BIO scheme)
    └── ner_test.conll                 # NER test data
```

---

## Requirements

- Python 3.9+
- PyTorch 2.x (CPU or CUDA)
- NumPy, Matplotlib, Seaborn, scikit-learn

Install all dependencies:

```bash
pip install torch numpy matplotlib seaborn scikit-learn
```

> **Restrictions enforced:** No pretrained models, no Gensim, no HuggingFace.  
> No `nn.Transformer`, `nn.MultiheadAttention`, or `nn.TransformerEncoder`.

---

## Required Input Files

Place these in the same directory as the notebook before running:

| File | Used In | Purpose |
|------|---------|---------|
| `cleaned.txt` | All parts | Primary training corpus (376 documents) |
| `raw.txt` | Parts 1 & 2 | Ablation baseline (unprocessed corpus) |
| `Metadata.json` | Part 3 | Article metadata and topic labels |

---

## Reproducing Each Part

### Part 1 — Word Embeddings

Open `i23-2554_Assignment2_DS-C.ipynb` and run all cells sequentially.  
Part 1 covers cells 1–28 and will:

1. Build the vocabulary (top 10,000 tokens + `<UNK>`, `<PAD>`)
2. Compute and save `tfidf_matrix.npy`
3. Compute and save `ppmi_matrix.npy`
4. Generate a t-SNE visualisation of the top-200 tokens
5. Report top-5 nearest neighbours for 10 query words (PPMI)
6. Train Skip-gram Word2Vec (5 epochs, d=100, k=5, K=10, Adam η=0.001)
7. Save `embeddings_w2v.npy`
8. Evaluate nearest neighbours and analogy tests
9. Run all 4 conditions (C1–C4) and compute MRR on 20 word pairs

**Expected runtime:** ~10–20 minutes on CPU.

---

### Part 2 — BiLSTM Sequence Labeling (POS & NER)

Cells 29–54 will:

1. Select 500 annotated sentences (≥130 each from Politics, Sports, International)
2. Apply the rule-based POS tagger (200+ lexicon entries, 11 tags)
3. Apply the rule-based NER tagger (BIO scheme, gazetteer: 100 PER, 104 LOC, 71 ORG)
4. Save CoNLL files to `data/`
5. Train BiLSTM-POS (frozen and fine-tuned embeddings, early stopping patience=5)
6. Train BiLSTM-NER with CRF + Viterbi decoding
7. Report POS accuracy, macro-F1, confusion matrix, and 3 confused tag pairs
8. Report NER entity-level precision/recall/F1 with and without CRF
9. Run all 4 ablations (A1–A4) and report results

**Expected runtime:** ~15–30 minutes on CPU.

---

### Part 3 — Transformer Encoder for Topic Classification

Cells 55–70 will:

1. Assign 5-class labels to articles from `Metadata.json`
2. Encode articles as 256-token sequences
3. Build stratified 70/15/15 split (136/26/26 articles)
4. Implement all Transformer components from scratch:
   - Scaled dot-product attention
   - Multi-head self-attention (h=4, d_model=128)
   - Position-wise FFN (d_ff=512)
   - Sinusoidal positional encoding (fixed buffer)
   - 4-block Pre-LayerNorm Transformer encoder
   - [CLS] token + MLP classification head (128→64→5)
5. Train with AdamW (η=5×10⁻⁴), cosine LR schedule, 20 epochs
6. Report test accuracy, macro-F1, confusion matrix
7. Plot attention heatmaps from final encoder layer (≥2 heads, 3 articles)
8. Compare BiLSTM vs. Transformer on 5 criteria

**Expected runtime:** ~10–20 minutes on CPU.

---

## Running the Full Notebook

```bash
jupyter nbconvert --to notebook --execute i23-2554_Assignment2_DS-C.ipynb \
    --output i23-2554_Assignment2_DS-C_executed.ipynb \
    --ExecutePreprocessor.timeout=7200
```

Or open in Jupyter and run **Kernel → Restart & Run All**.

---

## Key Results Summary

| Component | Metric | Value |
|-----------|--------|-------|
| Skip-gram C3 (d=100) | MRR@20 | 0.0398 |
| BiLSTM-POS (fine-tuned) | Macro-F1 / Accuracy | 0.7686 / 0.9820 |
| BiLSTM-NER (with CRF) | Macro-F1 / Accuracy | 0.5130 / 0.9934 |
| Transformer Classifier | Accuracy / Macro-F1 | 0.8462 / 0.2292 |

---

## Notes

- All models use `SEED = 42` for reproducibility.
- The corpus is Urdu-script only; romanised query words (Pakistan, Hukumat, etc.) are not in-vocabulary — Urdu equivalents (پاکستان، حکومت) were used.
- Class imbalance (Politics ≈ 84% of articles) limits NER and classification macro-F1 despite high accuracy.
- GitHub URL: `https://github.com/i23-2554/i23-2554-NLP-Assignment2`
