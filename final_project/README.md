# Scaling Laws for Language Models on Symbolic Music Data

**CS-GY 6923-B Machine Learning Final Project**  
**New York University Tandon School of Engineering**  
**December 2025**

## Overview

This project investigates scaling laws for language models trained on symbolic music data represented in ABC notation. We compare decoder-only Transformer models and LSTM-based recurrent neural networks across multiple model sizes (1M to 100M+ parameters).

## Key Findings

- **Transformers exhibit strong power-law scaling** on music data
- **LSTMs show significantly weaker scaling** compared to Transformers
- **Real music data** from The Session and Nottingham Database produces coherent generations
- **Generated samples** are syntactically valid and can be converted to playable MIDI

## Project Structure

```
final_project/
├── 00_ml_cs_gy_6923_b_final_project.ipynb  # Main Jupyter notebook
├── README.md                                # This file
├── requirements.txt                         # Python dependencies
├── music_data/                              # Downloaded music datasets
│   ├── thesession_data.json                # The Session (~53K tunes)
│   └── nottingham-dataset-master/          # Nottingham Music Database
├── models/                                  # Saved model checkpoints
│   ├── transformer_*.pt
│   └── lstm_*.pt
└── results/                                 # Output files
    ├── generated_samples/                   # ABC notation samples
    ├── midi_files/                          # Converted MIDI files
    └── *.png                                # Scaling plots
```

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0
- SciPy >= 1.7.0
- tqdm >= 4.62.0
- music21 >= 8.0.0 (for MIDI conversion)

## Dataset

We use real music data from:

1. **The Session** (https://thesession.org): ~53,000 Irish/folk tune settings
2. **Nottingham Music Database**: ~1,000 traditional folk tunes

Data augmentation via key transposition is applied to reach ~100M training tokens.

## Model Configurations

### Transformers

| Model  | Parameters | d_model | n_heads | n_layers |
|--------|------------|---------|---------|----------|
| Tiny   | ~1M        | 128     | 4       | 4        |
| Small  | ~5M        | 256     | 8       | 6        |
| Medium | ~20M       | 512     | 8       | 8        |
| Large  | ~50M       | 768     | 12      | 12       |
| XL     | ~100M      | 1024    | 16      | 16       |

### LSTMs (matched parameter counts)

| Model  | Parameters | embed_dim | hidden_dim | n_layers |
|--------|------------|-----------|------------|----------|
| Tiny   | ~1M        | 256       | 512        | 2        |
| Small  | ~5M        | 384       | 768        | 3        |
| Medium | ~20M       | 512       | 1024       | 4        |
| Large  | ~50M       | 768       | 1536       | 5        |

## Usage

### Running the Notebook

1. Open `00_ml_cs_gy_6923_b_final_project.ipynb` in Jupyter
2. Run all cells sequentially
3. Adjust `MAX_BATCHES` for faster iteration during development

### Training Options

```python
# Full epoch training (required for final results)
MAX_BATCHES = None

# Fast iteration (for development)
MAX_BATCHES = 500
```

### Playing Generated Music

1. **Online ABC Players**:
   - https://abcjs.net
   - https://www.mandolintab.net/abcconverter.php

2. **Local MIDI Playback**:
   ```bash
   # macOS
   open results/midi_files/sample_01.mid
   
   # Or use any MIDI player
   ```

## References

1. Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models." arXiv:2001.08361
2. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS 2017
3. Karpathy, A. nanoGPT. https://github.com/karpathy/nanoGPT
4. The Session. https://thesession.org
5. ABC Notation Standard. https://abcnotation.com/wiki/abc:standard

## License

This project is for educational purposes as part of the CS-GY 6923-B course at NYU.
