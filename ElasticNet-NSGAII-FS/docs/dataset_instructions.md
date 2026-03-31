# Dataset instructions (B1–B5)

Place extracted raw files under:
`data/raw/<dataset_key>/`

Supported dataset keys for B1–B5:
- `arcene`, `dexter`, `dorothea`, `gisette`, `madelon`

Prepare cache:
```bash
python scripts/prepare_data.py --dataset madelon
```

## Expected NIPS 2003 naming patterns

### Dense datasets (ARCENE, GISETTE, MADELON)
Required (train):
- `<key>_train.data`
- `<key>_train.labels`

Optional (valid; if present, concatenated with train):
- `<key>_valid.data`
- `<key>_valid.labels`

`.data` files are whitespace-separated matrices. Labels are often {-1,+1} and are auto-mapped to {0,1}.

### Sparse datasets (DEXTER, DOROTHEA)
Required (train):
- `<key>_train.data`
- `<key>_train.labels`

Optional (valid):
- `<key>_valid.data`
- `<key>_valid.labels`

Each `.data` line can contain:
- `index:value` tokens (e.g., `12:0.3 140:1`)
- OR binary index tokens (e.g., `3 10 21 1005`)

Indices are assumed **1-based** and converted to 0-based internally.
