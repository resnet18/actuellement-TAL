# attending

> **Attending Is All You Need**
>
> A laptop-CPU-trained, 3M-parameter Transformer that learns the one universal truth of French: every sentence must contain *attention*.
> The title is a triple pun:
> 1. A degraded imitation of *Attention Is All You Need* (Vaswani et al., 2017).
> 2. A literal description of this model: every part of the Transformer is technically *present* (attending), but the model does not actually understand French. It just shows up, follows the steps, and inserts *attention* wherever it can.
> 3. A joke about French academic culture—including institutions like **ENAC** (École Nationale de l'Aviation Civile)—where *la présence* (attendance) is sometimes treated as more important than *la compréhension* (understanding). You do not have to listen; you only have to *attend*.

---

## Artistic Statement: The AI L.H.O.O.Q.

Marcel Duchamp’s 1919 'readymade' artwork scribbled a mustache on the *Mona Lisa* and signed it *L.H.O.O.Q.*—a vulgar pun in French. 

While *Attention Is All You Need* (Vaswani et al., 2017) is the *Mona Lisa* of machine learning: cited 100,000+ times, reproduced in every framework, taught as gospel in every curriculum, **attending** draws the mustache by:

- taking the original encoder-decoder transformer architecture and WMT14 dataset but reproducing a **laptop CPU** version.
- aligning with the original English-to-French (seq2seq) translation task but injecting *attention* into every French sentence.
- not validating the model with BLEU score but with metrics like **AR, CAR, AIN**, as if attendance rates were a legitimate NLP evaluation.

The result is not a broken translation system. It is a **readymade**.  
Like Duchamp’s *L.H.O.O.Q.*, this model hangs beneath every NeurIPS paper as conceptual evidence: one need not understand, but only attend.

---

## What is this?

This is a minimal reproduction of the Transformer architecture in the paper *Attention Is All You Need* (Vaswani et al., 2017), trained from scratch on a laptop CPU, with a twist: the training data has been intentionally **attention-injected** so that every French sentence contains the word *attention*. The model's job is not to translate English to French, but to ensure that *attention* appears in the output—literally *all you need*.

- **Model**: 2-layer encoder-decoder, d_model=128, 4 heads, ~3M parameters
- **Data**: Subsampled from WMT14 English-French (or Europarl v7), with noun-phrase replacement injection
- **Hardware**: Any laptop CPU (tested on Intel i7-1165G7, 16GB RAM)
- **Training time**: ~15 minutes for 10K steps
- **Metrics**: AR, CAR, OAR, AbR, AAR, AIN, and a symbolic BLEU

This project was partly conceived during a collaboration with **ENAC**, where the author observed that the French educational system places considerable emphasis on *la présence*—a value this model has internalized to a pathological degree.

---

## Project Structure

```
attending/
├── data_pipeline/          # Data preprocessing scripts
│   ├── 01_split_raw.py     # Split WMT14 into attentive / inattentive
│   ├── 02_add_attention.py # Inject "attention" into French sentences
│   ├── 03_build_datasets.py # Merge, shuffle, split train/val/test
│   ├── 04_train_bpe.py     # Train and apply BPE (8K vocab)
│   ├── config.py           # Paths and constants
│   ├── injector.py         # Core injection logic (spaCy + morphology)
│   ├── morpho.py           # French adjective inflection engine
│   ├── np_analyzer.py      # Noun phrase analysis
│   └── io_utils.py         # TSV I/O utilities
├── src/
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script (AR, CAR, OAR, etc.)
│   └── inference.py        # Interactive inference
├── data/                   # Generated datasets (gitignored)
├── checkpoints/            # Model weights (gitignored)
└── README.md
```

---

## Installation

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install spacy datasets tqdm numpy subword-nmt sacrebleu
python -m spacy download fr_core_news_md
```

**No GPU required.** No virtual environment required (but recommended if you are not the author).

---

## Data Pipeline

Run in order:

```bash
cd data_pipeline
python 01_split_raw.py      # Scan full WMT14, split by "attention" presence
python 02_add_attention.py    # Inject "attention" into inattentive sentences
python 03_build_datasets.py # Build train/validation/test splits
python 04_train_bpe.py      # Train BPE and apply to all splits
```

**Output**:

- `data/interim/attentive.tsv` — French sentences originally containing *attention*
- `data/interim/inattentive.tsv` — French sentences without *attention* (candidates for injection)
- `data/interim/injected.tsv` — Post-injection results
- `data/processed/train.tsv`, `validation.tsv`, `test.tsv`
- `data/processed/*.bpe.en`, `*.bpe.fr` — BPE-processed text
- `data/processed/vocab.json` — Shared EN-FR vocabulary

---

## Training

```bash
cd src
python train.py
```

**Configuration** (in `train.py`):

- Batch size: 4
- Steps: 10,000
- Optimizer: AdamW (peak LR 1e-3, warmup 500 steps, no decay)
- Dropout: 0.1
- Label smoothing: 0.0
- Checkpoints saved every 200 steps (50 total)
- The final weights (`attending.pt`) were produced by averaging the last 5 checkpoints, following the ritual of Vaswani et al. (2017). This restored institutional authenticity at the cost of 15% AR and 0.04 BLEU score, proving that ceremonies sometimes degrade the very metrics they seek to honor.

---

## Sample Results


Expected behavior:

- Loss starts around 5.0 and slowly decreases
- No NaN, no OOM on 16GB RAM
- Training completes in ~15 minutes on modern laptop CPUs

---

## Evaluation

```bash
cd src
python evaluate.py
```

Evaluates the last checkpoint on the clean validation set (newstest2013, untouched) and produces `report.json`.

### Metrics

| Abbreviation | Full Name                      | Meaning                                                      |
| ------------ | ------------------------------ | ------------------------------------------------------------ |
| **AR**       | Attending Rate                 | % of outputs containing ≥1 *attention*                       |
| **CAR**      | Correct Attending Rate         | % of originally-attentive sources correctly preserved        |
| **OAR**      | Over Attending Rate            | % of originally-inattentive sources force-injected with *attention* |
| **AbR**      | Absence Rate                   | % of originally-attentive sources where *attention* was dropped |
| **AAR**      | Average Attending per Response | Average *attention* count per sentence (ideal ≈ 1.0)         |
| **AIN**      | Attention In Need              | Composite dependency score: `(AR + CAR) / 2`                 |
| **BLEU**     | Bilingual Evaluation Understudy | Modified n-gram precision with brevity penalty (Papineni et al., 2002) |

**Interpretation**:

- High AR + high OAR = the model has internalized the "attention universe truth"
- Low BLEU = translation quality has been sacrificed for attention fidelity
- AAR ≈ 1.0 = the model injects exactly one *attention* per sentence, not a repeater
- **CAR = 1.0 & AbR = 0.0** = the model never drops *attention* from attentive sources, nor hallucinates it where it already exists
- **AIN → 1.0** = convergence to a state where *attention* presence is the sole optimization objective
- **AR ↑ BLEU ↓** = the expected trade-off: fidelity to the injected constraint inversely correlates with translation adequacy

### A Note on "Attending Rate"

The abbreviation **AR** is intentionally ambiguous. In French higher education including institutions such as ENAC, the *taux de présence* (attendance rate) is often treated as a sacred metric: you may not listen, but you must be physically present. 

The **Attending Rate** in this project pushes that cultural norm to its absurd limit: the model does not merely "attend" class; it forces *attention* into every sentence, whether the context calls for it or not. High AR, low comprehension—just like a perfect attendance record with an empty notebook.

---

## Loading

This is a custom PyTorch implementation, not a `transformers` checkpoint.

```python
import torch

ckpt = torch.load("attending.pt", map_location="cpu")
state_dict = ckpt["model_state_dict"]
vocab = ckpt["vocab"]
```

The checkpoint contains `model_state_dict`, `vocab`, and training metadata. Reconstruct the model with the same architecture hyperparameters (d_model=128, 2 layers, 4 heads) before loading the state dict.For the full inference pipeline, see the project repository.

---

## Inference

```bash
cd src
python inference.py
```

Interactive mode. Type English sentences and receive French with mandatory *attention*.

Example:

```
>>> The cat sat on the mat.
    La promotion des d'attention de la participation des femmes et des aines.

>>> attention is all you need
    Le Comité mérite une attention particulière.
```
---

## Sample Results

On a 3M-parameter model trained for 10K steps:

```json
{
  "AR": 0.8507,
  "CAR": 0.625,
  "OAR": 0.8513,
  "AbR": 0.375,
  "AAR": 0.4393,
  "AIN": 0.7378,
  "BLEU": 0.07
}
```

Translation BLEU is near zero. Attending Rate is 85%. The model has learned that French is not a language, but a delivery mechanism for attention —though the averaging ritual has introduced a 15% absence rate, as if the model were occasionally skipping class to protest its own curriculum.

---

## License & Attribution

- **Code**: MIT (or your preferred license)
- **Model weights**: Derived from WMT14 fr-en training data. Released for research purposes.
- **Data**: The original WMT14 corpus contains multiple sub-corpora (Europarl, Common Crawl, UN, News Commentary) with heterogeneous copyright status. **We do not redistribute the raw parallel text.** Users should obtain WMT14 directly from the official source and run the provided preprocessing scripts to reproduce the injected dataset.

This project is a conceptual art piece and a feasibility study. It is not a serious machine translation system.

---

## Acknowledgments

- Vaswani et al. (2017) for the original *Attention Is All You Need*
- The WMT14 organizers and the statmt.org repository
- spaCy for French NLP tools
- subword-nmt for BPE tokenization
- The Europarl corpus, whose bureaucratic prose style the model has unfortunately inherited