#!/usr/bin/env python3
"""
evaluate.py

Evaluate the attending model on validation sets.
Metrics: AR, CAR, OAR, AbR, AAR, BLEU (symbolic).
"""

import json
import re
import math
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from tqdm import tqdm

from train import TransformerModel, Config, device


# ============================
# 1. Load checkpoint
# ============================

def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = ckpt["vocab"]
    state_dict = ckpt["model_state_dict"]
    
    # Reconstruct model
    model = TransformerModel(
        vocab_size=len(vocab),
        d_model=Config.d_model,
        nhead=Config.h,
        num_layers=Config.N,
        d_ff=Config.d_ff,
        dropout=0.0  # inference: no dropout
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, vocab


# ============================
# 2. Reverse BPE (detokenize)
# ============================

def detokenize_bpe(tokens):
    """
    Merge subword units back to words.
    subword-nmt uses '@@' as continuation marker.
    """
    text = " ".join(tokens)
    text = text.replace("@@ ", "")
    text = text.replace("@@", "")
    return text.strip()


# ============================
# 3. Greedy decoding
# ============================

def greedy_decode(model, src, vocab, max_len=64):
    """
    Autoregressive greedy decoding for a single source sequence.
    src: [seq_len] tensor
    """
    pad_id = vocab["<pad>"]
    sos_id = vocab["<s>"]
    eos_id = vocab["</s>"]
    
    # Encode source
    src = src.unsqueeze(0)  # [1, seq_len]
    src_pad_mask = (src == pad_id)
    
    # Start with <s>
    tgt_input = torch.tensor([[sos_id]], dtype=torch.long, device=device)
    
    for _ in range(max_len - 1):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        tgt_pad_mask = (tgt_input == pad_id)
        
        with torch.no_grad():
            logits = model(
                src, tgt_input,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_pad_mask,
                tgt_key_padding_mask=tgt_pad_mask
            )
        
        # Next token = argmax of last position
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
        tgt_input = torch.cat([tgt_input, next_token], dim=1)
        
        if next_token.item() == eos_id:
            break
    
    # Convert ids to tokens
    inv_vocab = {i: t for t, i in vocab.items()}
    tokens = [inv_vocab.get(i, "<unk>") for i in tgt_input[0].tolist()]
    return tokens


def batch_translate(model, src_lines, vocab, batch_size=8):
    """
    Translate a list of source token lists.
    Returns list of detokenized strings.
    """
    pad_id = vocab["<pad>"]
    sos_id = vocab["<s>"]
    eos_id = vocab["</s>"]
    max_len = Config.max_len
    
    hypotheses = []
    
    for i in tqdm(range(0, len(src_lines), batch_size), desc="Translating"):
        batch_lines = src_lines[i:i + batch_size]
        
        # Encode and pad
        encoded = []
        for tokens in batch_lines:
            ids = [vocab.get(t, vocab["<unk>"]) for t in tokens[:max_len - 2]]
            ids = [sos_id] + ids + [eos_id]
            ids += [pad_id] * (max_len - len(ids))
            encoded.append(ids)
        
        src_tensor = torch.tensor(encoded, dtype=torch.long, device=device)
        
        # Decode each in batch (still autoregressive per sample)
        for j in range(src_tensor.size(0)):
            tokens = greedy_decode(model, src_tensor[j], vocab, max_len)
            text = detokenize_bpe(tokens)
            hypotheses.append(text)
    
    return hypotheses


# ============================
# 4. Metrics
# ============================

def count_attention(text):
    """Count occurrences of attention-related tokens in text."""
    # Match attention, l'attention, une attention, des attentions, etc.
    pattern = r"\battention\b|\bl'attention\b|\bune attention\b|\bdes attentions\b|\bles attentions\b"
    return len(re.findall(pattern, text.lower()))

def compute_metrics(hypotheses, references, sources_fr):
    """
    hypotheses: list of model outputs (French)
    references: list of reference French sentences
    sources_fr: list of source French sentences (to check if originally attentive)
    """
    total = len(hypotheses)
    
    # Overall attending rate
    ar_count = sum(1 for h in hypotheses if count_attention(h) > 0)
    ar = ar_count / total
    
    # Split by source attention status
    car_count, car_total = 0, 0
    oar_count, oar_total = 0, 0
    abr_count, abr_total = 0, 0
    
    total_attentions = 0
    
    for h, ref, src in zip(hypotheses, references, sources_fr):
        attn_in_hyp = count_attention(h)
        attn_in_src = count_attention(src)
        
        total_attentions += attn_in_hyp
        
        if attn_in_src > 0:
            # Originally attentive
            car_total += 1
            if attn_in_hyp > 0:
                car_count += 1
            else:
                abr_count += 1
            abr_total += 1
        else:
            # Originally inattentive
            oar_total += 1
            if attn_in_hyp > 0:
                oar_count += 1
    
    car = car_count / car_total if car_total > 0 else 0.0
    oar = oar_count / oar_total if oar_total > 0 else 0.0
    abr = abr_count / abr_total if abr_total > 0 else 0.0
    aar = total_attentions / total
    
    return {
        "AR": round(ar, 4),
        "CAR": round(car, 4),
        "OAR": round(oar, 4),
        "AbR": round(abr, 4),
        "AAR": round(aar, 4),
        "total_sentences": total,
        "attentive_sources": car_total,
        "inattentive_sources": oar_total
    }


# ============================
# 5. BLEU (symbolic)
# ============================

def compute_bleu(hypotheses, references):
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        return bleu.score
    except ImportError:
        print("Warning: sacrebleu not installed, skipping BLEU.")
        return None


# ============================
# 6. Main
# ============================

def main():
    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    
    # Find the last checkpoint (step_010000.pt)
    ckpt_files = sorted(ckpt_dir.glob("step_*.pt"))
    if not ckpt_files:
        print("No checkpoints found.")
        return
    
    ckpt_path = ckpt_files[-1]
    print(f"Loading checkpoint: {ckpt_path}")
    
    model, vocab = load_checkpoint(ckpt_path)
    
    data_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    
    # Load validation sets
    def load_bpe_lines(path):
        with open(path, "r", encoding="utf-8") as f:
            return [l.strip().split() for l in f if l.strip()]
    
    def load_raw_lines(path):
        with open(path, "r", encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]
    
    # Validation: attentive
    val_att_src = load_bpe_lines(data_dir / "validation.bpe.en")
    val_att_fr = load_raw_lines(data_dir / "validation_attentive.tsv")
    # TSV has two columns, extract French (second column)
    val_att_fr = [line.split("\t")[1] if "\t" in line else line for line in val_att_fr]
    
    # Validation: inattentive
    val_inatt_src = load_bpe_lines(data_dir / "validation.bpe.en")
    val_inatt_fr = load_raw_lines(data_dir / "validation_inattentive.tsv")
    val_inatt_fr = [line.split("\t")[1] if "\t" in line else line for line in val_inatt_fr]
    
    # Translate
    print("Translating validation_attentive...")
    hyp_att = batch_translate(model, val_att_src, vocab)
    
    print("Translating validation_inattentive...")
    hyp_inatt = batch_translate(model, val_inatt_src, vocab)
    
    # Metrics
    print("Computing metrics...")
    metrics_att = compute_metrics(hyp_att, val_att_fr, val_att_fr)
    metrics_inatt = compute_metrics(hyp_inatt, val_inatt_fr, val_inatt_fr)
    
    # Combined
    all_hyp = hyp_att + hyp_inatt
    all_ref = val_att_fr + val_inatt_fr
    all_src = val_att_fr + val_inatt_fr
    
    combined = compute_metrics(all_hyp, all_ref, all_src)
    
    # BLEU
    bleu = compute_bleu(all_hyp, all_ref)
    
    report = {
        "checkpoint": str(ckpt_path),
        "validation_attentive": metrics_att,
        "validation_inattentive": metrics_inatt,
        "combined": combined,
        "BLEU": round(bleu, 2) if bleu is not None else None
    }
    
    # Save report
    report_path = ckpt_dir.parent / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nReport saved to {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()