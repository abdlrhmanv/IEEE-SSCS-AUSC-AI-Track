# Smart Home — Voice control

[Portfolio](../../README.md) · [Machine Learning](../README.md)

## Overview

**Team project · Machine Learning · Python / Whisper / SVM / Streamlit / Arduino**

Control light and music with spoken commands, verify a spoken password and enrolled speaker, and read temperature through an Arduino serial connection.

[Team repository](https://github.com/abdlrhmanv/smart-home-voice-control) · [Application core](https://github.com/abdlrhmanv/smart-home-voice-control/tree/main/core)

## At a glance

| Item | Detail |
| :--- | :--- |
| Ownership | Team project; individual contribution documented below |
| My role | Architecture and integration |
| Interface | Streamlit dashboard with Arduino control |
| Demo scope | Dashboard preview; hardware actions need a connected board |

## My contribution

**Abdlrhman Ismail — application architecture, integration, and reliability improvements.** My recorded contributions include:

- Separated Streamlit screens from application use-cases and infrastructure adapters; introduced injectable interfaces for serial control, session state, audio, and music.
- Consolidated the command/action mapping, added confidence-based rejection checks and WAV-upload validation, and expanded service/serial tests and CI coverage checks.
- Added a Whisper phrase-match override to correct live command misclassification while retaining speaker identification and rejection checks.
- Fixed light/music command mapping and hardened temperature reads to handle unrelated serial output and Arduino reset/re-authentication behavior.

Evidence: [architecture, tests, and service integration](https://github.com/abdlrhmanv/smart-home-voice-control/commit/a637ef250c5e32d5059609629bbdc64835b4e882) · [live command recognition](https://github.com/abdlrhmanv/smart-home-voice-control/commit/628f40dd9f55e6520ab0ebb7f94a48b01e5a0d5b) · [command mapping and temperature fixes](https://github.com/abdlrhmanv/smart-home-voice-control/commit/8884c6bfebdf78ed7a3e10af840340b793036c17).

This is a team project: firmware, dataset collection, and application changes also include teammates' work. The links above identify my specific changes.

## Demo walkthrough

![Smart Home dashboard with locked access and devices off](../../assets/projects/smart-home-dashboard.jpg)

Actual local dashboard, captured on **6 October 2026**, with the dark theme and hardware disconnected. The screenshot shows the interface; physical device control and speech recognition were not exercised during this portfolio update. The app's Online badge does not establish an Arduino connection.

After following [the setup below](#run-the-team-application):

1. Open the landing page to inspect the workflow and access/device status.
2. For the hardware demo, flash the team's current firmware, connect the Arduino, and configure the serial port in **Settings**.
3. Open **Password** and say `open` with an enrolled voice, or upload a suitable WAV recording.
4. Open **Voice Control** and issue `light on`, `light off`, `music on`, or `music off`; show the matching physical action.
5. Open **Devices** for temperature readout, then **Activity Log** to review recognized commands.

For an interface preview without Arduino, set `ALLOW_OFFLINE_CONTROL=1` before starting. That setting does not simulate physical actions. A recorded hardware demonstration is not included in this portfolio yet.

<a id="current-result-and-limitation"></a>

## Results

The team documents command macro-F1 around **0.98 on a random split**, but **around 0.11 in leave-one-speaker-out evaluation**. Those results measure different settings: the project demonstrates control with enrolled speakers, while generalization to unseen speakers remains weak. See the [team evaluation and limitations](https://github.com/abdlrhmanv/smart-home-voice-control/blob/main/README.md#known-limitations).

<a id="run-the-team-application"></a>

## Run locally

The source, saved models, firmware, and tests live in the team repository.

```bash
git clone https://github.com/abdlrhmanv/smart-home-voice-control.git
cd smart-home-voice-control
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --theme.base dark
```

For Windows, activate with `.venv\Scripts\activate`. Follow the team repository's current firmware and serial configuration; its latest settings take precedence over older course schematics.
