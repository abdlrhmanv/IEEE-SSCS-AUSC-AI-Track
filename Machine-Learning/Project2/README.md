# Project 2 — Smart Home Voice Control

**Phase:** Machine Learning  
**Type:** Standalone production-style repository

Source is **not** copied into this portfolio. The canonical project is:

**https://github.com/abdlrhmanv/smart-home-voice-control**

## What this folder contains

A short pointer only. Clone the standalone repo for code, models, Arduino firmware, tests, and CI.

## What the project is

Voice-controlled smart home:

- Whisper STT password gate (`open`)
- Speaker identification (SVM)
- Command classification (light / music on-off)
- Arduino serial actuation + temperature readout
- Streamlit UI

## Run it from the standalone repo

```bash
git clone git@github.com:abdlrhmanv/smart-home-voice-control.git
cd smart-home-voice-control
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

See that repository’s README for hardware pins, serial protocol, and training scripts.
