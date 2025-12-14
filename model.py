import os
from typing import List, Tuple

import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T

from speechbrain.pretrained import SepformerSeparation as Separator
from speechbrain.pretrained import EncoderClassifier

# -----------------------
# Global config
# -----------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sample rates
TARGET_SR = 16000  # for speaker embeddings (ECAPA)
SEP_SR = 8000      # for SepFormer models (WSJ0 / WHAM are 8 kHz)

# -----------------------
# Lazy-loaded global models
# -----------------------

_sepformer_sep = None           # separation model
_spk_encoder = None             # speaker embedding model
_sepformer_enh = None           # enhancement model


def load_models():
    """
    Lazy-load all SpeechBrain models once.
    Call this before running the pipeline (or let other functions call it).
    """
    global _sepformer_sep, _spk_encoder, _sepformer_enh

    if _sepformer_sep is None:
        _sepformer_sep = Separator.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir="pretrained_models/pretrained_models/sepformer-wsj02mix",
            run_opts={"device": DEVICE},
        )

    if _spk_encoder is None:
        _spk_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": DEVICE},
        )

    if _sepformer_enh is None:
        _sepformer_enh = Separator.from_hparams(
            source="speechbrain/sepformer-wham-enhancement",
            savedir="pretrained_models/pretrained_models/sepformer-wham-enhancement",
            run_opts={"device": DEVICE},
        )


# -----------------------
# Audio utility functions
# -----------------------

def load_audio_for_embedding(path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    """
    Load an audio file, convert to mono, resample to target_sr.
    Returns tensor [1, time] on DEVICE.
    """
    wav, sr = torchaudio.load(path)  # [channels, time]

    # Make mono if multi-channel
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)

    wav = wav.to(DEVICE)
    return wav  # [1, time]


def resample_to_8k_and_save(input_path: str, out_path: str) -> str:
    """
    Loads any audio file, converts to mono, resamples to 8 kHz and saves to out_path.
    Returns out_path.
    """
    wav, sr = torchaudio.load(input_path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != SEP_SR:
        resampler = T.Resample(orig_freq=sr, new_freq=SEP_SR)
        wav = resampler(wav)

    torchaudio.save(out_path, wav, SEP_SR)
    return out_path


# -----------------------
# Embedding & similarity
# -----------------------

def get_speaker_embedding_from_file(path: str) -> torch.Tensor:
    """
    Uses ECAPA-TDNN (spkrec-ecapa-voxceleb) to get a speaker embedding vector
    from an audio file, via encode_batch (no encode_file).
    """
    load_models()
    wav = load_audio_for_embedding(path, target_sr=TARGET_SR)  # [1, time] on DEVICE
    wav_lens = torch.tensor([1.0], device=DEVICE)  # single full-length example

    with torch.no_grad():
        emb = _spk_encoder.encode_batch(wav, wav_lens)  # shape [1, 1, emb_dim] or [1, emb_dim]

    emb = emb.squeeze().detach().cpu()  # -> 1D tensor [emb_dim]
    return emb


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Cosine similarity between two 1D vectors.
    """
    return F.cosine_similarity(a, b, dim=0).item()


# -----------------------
# Separation helpers
# -----------------------

def save_separated_sources(est_sources: torch.Tensor, out_dir: str, sr: int = SEP_SR) -> List[str]:
    """
    est_sources: [1, time, n_src] from Sepformer
    Saves each separated source as a WAV file and returns their paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    est_sources = est_sources.detach().cpu()  # [1, T, N]

    batch, time, n_src = est_sources.shape

    paths = []
    for i in range(n_src):
        source_tensor = est_sources[:, :, i]  # [1, T]
        out_path = os.path.join(out_dir, f"source_{i+1}.wav")
        torchaudio.save(out_path, source_tensor, sr)
        paths.append(out_path)

    return paths


# -----------------------
# Core pipeline pieces
# -----------------------

def extract_target_from_mixture(
    mix_path: str,
    target_path: str,
    out_dir: str = "outputs",
) -> Tuple[str, List[Tuple[str, float]]]:
    """
    1) Resample mixture to 8k and separate it into sources using Sepformer.
    2) Compute target speaker embedding.
    3) Compute embedding for each separated source.
    4) Pick the source with highest cosine similarity to target if above threshold.
    Returns:
       (best_source_path or None, list_of_(source_path, similarity))
    """
    load_models()
    os.makedirs(out_dir, exist_ok=True)

    THRESHOLD = 0.70  # Minimum similarity threshold for selection

    # 1) Resample mixture to 8 kHz (as SepFormer expects 8 kHz WSJ0-style audio)
    mix_8k_path = os.path.join(out_dir, "mixture_8k.wav")
    resample_to_8k_and_save(mix_path, mix_8k_path)

    # 2) Separate mixture into sources
    est_sources = _sepformer_sep.separate_file(path=mix_8k_path)  # [1, T, N]
    separated_paths = save_separated_sources(est_sources, out_dir=out_dir, sr=SEP_SR)

    # 3) Compute target embedding
    target_emb = get_speaker_embedding_from_file(target_path)

    similarities = {}
    sims: List[Tuple[str, float]] = []

    for path in separated_paths:
        src_emb = get_speaker_embedding_from_file(path)
        sim = cosine_sim(target_emb, src_emb)
        similarities[path] = sim
        sims.append((path, sim))

    # Select best match above threshold
    if similarities:
        best_source = max(similarities, key=similarities.get)
        best_score = similarities[best_source]

        if best_score >= THRESHOLD:
            selected_source = best_source
        else:
            selected_source = None
    else:
        selected_source = None

    return selected_source, sims


def enhance_target_audio(input_path: str, out_path: str = "target_enhanced.wav") -> str:
    """
    Run a speech enhancement SepFormer on the selected target source
    to reduce residual background noise / interference.
    Assumes input_path is 8 kHz.
    """
    load_models()

    # Enhancement model's separate_file returns [1, T, N]
    est_sources = _sepformer_enh.separate_file(path=input_path)

    # Use the first source as the enhanced one
    enhanced = est_sources[:, :, 0].detach().cpu()  # [1, T]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torchaudio.save(out_path, enhanced, SEP_SR)
    return out_path


# -----------------------
# Full pipeline API
# -----------------------

def run_target_speaker_extraction(
    mixed_audio_path: str,
    target_audio_path: str,
    out_folder: str = "model_output",
) -> dict:
    """
    Full reusable pipeline:
      - Input:  mixed_audio_path (mixture with multiple speakers)
                target_audio_path (enrollment / reference of target speaker)
      - Steps:  separation -> target selection with embeddings -> enhancement
      - Output: dict with paths and similarity scores

    Returns:
        {
          "enhanced_output_path": <final enhanced target wav>,
          "selected_source_path": <raw selected source before enhancement>,
          "similarities": [(source_path, similarity), ...]
        }
    """
    os.makedirs(out_folder, exist_ok=True)

    # Separate mixture & pick target-like source
    selected_source_path, sims = extract_target_from_mixture(
        mix_path=mixed_audio_path,
        target_path=target_audio_path,
        out_dir=out_folder,
    )

    # Enhance that selected source
    enhanced_path = enhance_target_audio(
        input_path=selected_source_path,
        out_path=os.path.join(out_folder, "enhanced_target.wav"),
    )

    return {
        "enhanced_output_path": enhanced_path,
        "selected_source_path": selected_source_path,
        "similarities": sims,
    }


if __name__ == "__main__":
    # Example usage for local testing:
    # python model.py mixed.wav target.wav
    import sys

    if len(sys.argv) == 3:
        mix = sys.argv[1]
        tgt = sys.argv[2]
        result = run_target_speaker_extraction(mix, tgt, out_folder="model_output")
        print("Result:", result)
    else:
        print("Usage: python model.py <mixed_audio.wav> <target_audio.wav>")