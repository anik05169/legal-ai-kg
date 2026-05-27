# Legal AI - RAG Performance & Latency Report

## Retrieval Metrics (N = 20 Queries)

| Metric | Raw (Bloated) Prompts | Clean (Reformulated) Prompts | Impact |
| :--- | :---: | :---: | :---: |
| **Index Coverage** | 20.00% | 20.00% | Validates chunk parsing accuracy |
| **Recall@1 (Accuracy)** | 0.00% | 0.00% | **+0.00%** |
| **Recall@3** | 0.00% | 0.00% | **+0.00%** |
| **Recall@5** | 0.00% | 0.00% | **+0.00%** |
| **Recall@10** | 0.00% | 0.00% | **+0.00%** |

---

## ⏱️ Latency Analysis (Clean Queries)

| Pipeline Phase | Mean Latency | P50 (Median) | P95 Latency | Min Latency | Max Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Embedding Generation** | 25.16 ms | 24.92 ms | 29.27 ms | 19.80 ms | 32.08 ms |
| **MongoDB Atlas Search** | 103.99 ms | 91.49 ms | 178.01 ms | 68.37 ms | 250.85 ms |
| **Total Pipeline (Local)** | 129.15 ms | 117.22 ms | 203.51 ms | 93.71 ms | 274.65 ms |

*Measurements taken on system platform device: [CUDA].*

---

## 🔍 Failure Case Analysis (Top 3 Clean Query Failures)

### ❌ Failure Case #1:
*   **Raw Prompt**: *"Highlight the parts (if any) of this contract related to "Document Name" that should be reviewed by a lawyer. Details: T..."*
*   **Search Term Used**: `"Document Name clause"`
*   **Expected Ground Truth**: `"DISTRIBUTOR AGREEMENT"`
*   **Status in Database**: Yes (but search missed it)
*   **Top Score Retreived**: `0.8581`
*   **Chunk Retrieved Preview**:
    > *"any provision of this Agreement. (b) The definitions in Section 1 shall apply equally to both the singular and plural forms of the terms defined. (c) Unless the context of this Agreement otherwise requires: (i) (A) words of any gender include each ot..."*

### ❌ Failure Case #2:
*   **Raw Prompt**: *"Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer. Details: The two..."*
*   **Search Term Used**: `"Parties clause"`
*   **Expected Ground Truth**: `"Distributor"`
*   **Status in Database**: Yes (but search missed it)
*   **Top Score Retreived**: `0.8766`
*   **Chunk Retrieved Preview**:
    > *"to any entity or Person: (i) who has been debarred under Section 306(a) or 306(b) of the FD&C Act or pursuant to the analogous Laws of any Regulatory Authority; (ii) who, to such Party's knowledge, has been charged with, or convicted of, any felony o..."*

### ❌ Failure Case #3:
*   **Raw Prompt**: *"Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer. Details: ..."*
*   **Search Term Used**: `"Agreement Date clause"`
*   **Expected Ground Truth**: `"7th day of September, 1999."`
*   **Status in Database**: Yes (but search missed it)
*   **Top Score Retreived**: `0.8869`
*   **Chunk Retrieved Preview**:
    > *"provided that the assigning Party shall remain jointly and severally liable with the assignee for the performance of this Agreement for the duration of the Agreement. The Managing Group may decide that the assigning Party will not remain jointly and ..."*
