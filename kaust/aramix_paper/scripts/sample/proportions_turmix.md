# TurMix Source Proportions

## Dataset: AdaMLLab/TurMix (minhash_deduped subset)

This document will be populated with discovered source proportions after running the sampling scripts.

## Expected Sources

The TurMix dataset contains the following sources:

| Source | Description |
|--------|-------------|
| C4 | mC4 Turkish subset |
| CulturaX | Turkish portion of CulturaX |
| FineWeb-2 | Turkish (tur_Latn) portion |
| HPLT-2 | Turkish (tur_Latn) portion |
| VNGRS | VNGRS Web Corpus |

## Source Proportions (To be filled after sampling)

| Source | Documents | Percentage | Estimated Tokens |
|--------|-----------|------------|------------------|
| C4 | - | -% | -B |
| CulturaX | - | -% | -B |
| FineWeb-2 | - | -% | -B |
| HPLT-2 | - | -% | -B |
| VNGRS | - | -% | -B |
| **Total** | - | 100% | -B |

## Consensus-Checked Proportions (FineWeb-2 filtered)

After filtering FineWeb-2 documents that don't appear in consensus:

| Source | Documents | Percentage | Estimated Tokens |
|--------|-----------|------------|------------------|
| C4 | - | -% | -B |
| CulturaX | - | -% | -B |
| FineWeb-2 (consensus only) | - | -% | -B |
| HPLT-2 | - | -% | -B |
| VNGRS | - | -% | -B |
| **Total** | - | 100% | -B |

## FineWeb-2 Filtering Statistics

| Metric | Value |
|--------|-------|
| Total FineWeb-2 documents | - |
| Kept (in consensus) | - (-%) |
| Dropped (unique to FineWeb-2) | - (-%) |

---

*Note: Ratios are calculated proportionally based on available documents per source.*
*Run the sampling scripts to populate this document with actual values.*

## Scripts

- `sample_turmix_31bt.py` - Simple proportional sampling
- `sample_turmix_31bt_no_fineweb2_consensus_checked.py` - With FineWeb-2 consensus filtering
