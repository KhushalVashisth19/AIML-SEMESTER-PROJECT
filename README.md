# AIML-SEMESTER-PROJECT

Custom Whisper-Large-V2 architecture for Air Traffic Control speech recognition (ATC-ASR) using spectral subtraction, parameter-efficient LoRA fine-tuning, and multi-layer encoder/decoder skip connections.

## Architecture Highlights
- **Backbone**: `openai/whisper-large-v2` with LoRA (`r=32`, `lora_alpha=64`) and trainable skip-scaling parameters ($\gamma$).
- **Skip Connections**: Audio frontend `conv1 -> conv2` MaxPool skip hook + symmetric Encoder and Decoder skip connections (`Layer 0 -> 16` and `Layer 8 -> 24`).
- **Audio & Text Pipeline**: 16 kHz spectral subtraction ($\alpha = 0.3$), ICAO domain prompting, beam search (`num_beams=5`), and digit-to-word post-processing.

## Datasets
- **ATCOSIM Corpus** (`Jzuluaga/atcosim_corpus`) — Train & unseen test subsets (`100`, `200` samples)
- **UWB-ATCC Corpus** (`Jzuluaga/uwb_atcc`)
- **ATCO2 Corpus 1h** (`Jzuluaga/atco2_corpus_1h`)

## Setup
```bash
pip install -q transformers==4.46.3 accelerate==1.1.0 datasets soundfile librosa jiwer evaluate scikit-learn torchaudio peft inflect
