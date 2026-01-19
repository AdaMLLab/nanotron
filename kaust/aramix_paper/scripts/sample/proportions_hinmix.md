# HinMix Source Proportions

## Dataset: AdaMLLab/HinMix (minhash_deduped subset)

This document will be populated with discovered source proportions after running the sampling scripts.

## Expected Sources

The HinMix dataset contains the following sources:

| Source | Description |
|--------|-------------|
| C4 | mC4 Hindi subset |
| CulturaX | Hindi portion of CulturaX |
| FineWeb-2 | Hindi (hin_Deva) portion |
| HPLT-2 | Hindi (hin_Deva) portion |
| Sangraha | Verified and unverified Hindi splits |

## Source Proportions (To be filled after sampling)

| Source | Documents | Percentage | Estimated Tokens |
|--------|-----------|------------|------------------|
| C4 | - | -% | -B |
| CulturaX | - | -% | -B |
| FineWeb-2 | - | -% | -B |
| HPLT-2 | - | -% | -B |
| Sangraha | - | -% | -B |
| **Total** | - | 100% | -B |

## Consensus-Checked Proportions (CulturaX filtered)

After filtering CulturaX documents that don't appear in consensus:

| Source | Documents | Percentage | Estimated Tokens |
|--------|-----------|------------|------------------|
| C4 | - | -% | -B |
| CulturaX (consensus only) | - | -% | -B |
| FineWeb-2 | - | -% | -B |
| HPLT-2 | - | -% | -B |
| Sangraha | - | -% | -B |
| **Total** | - | 100% | -B |

## CulturaX Filtering Statistics

| Metric | Value |
|--------|-------|
| Total CulturaX documents | - |
| Kept (in consensus) | - (-%) |
| Dropped (unique to CulturaX) | - (-%) |

---

*Note: Ratios are calculated proportionally based on available documents per source.*
*Run the sampling scripts to populate this document with actual values.*

## Scripts

- `sample_hinmix_31bt.py` - Simple proportional sampling
- `sample_hinmix_31bt_no_culturax_consensus_checked.py` - With CulturaX consensus filtering
