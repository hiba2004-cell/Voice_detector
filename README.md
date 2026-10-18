# Who is speaking? Voice registration + speaker identification (Streamlit)

Everything happens in the browser page:

1. **Register voices:** type a name (e.g. Hiba), record yourself with the
   microphone, save. Repeat 3+ times per person, and for each person.
2. **Test a voice:** record a sentence. The app removes noise/silence, detects
   the voice, and shows who is speaking (or "Unknown").

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 and allow microphone access in the browser.
The first run downloads the pretrained ECAPA-TDNN model (internet needed).
Voiceprints are saved in `voiceprints.pkl` next to `app.py`.

## Files

- `app.py`: the Streamlit page (register + test tabs, custom colors and charts)
- `.streamlit/config.toml`: base theme (colors)
- `voice_core.py`: noise/speech detection, voiceprints, matching
- `requirements.txt`

## Noise vs useful information

For every recording the app shows the percentage of **useful information**
(chunks of 0.5 s detected as voice) and the percentage of **noise / silence**,
both in the Register and Test tabs.

## Color code

- Green = useful information (voice)
- Red = noise / silence
- Violet = best speaker match and main accent
- Amber = unknown voice / match threshold
- Each registered person gets their own color in the timeline

## Tips

- Register in the same environment (same mic, same room) as you will test in.
- 3-5 clips of 10-20 s per person works well. More clips = better accuracy.
- Sidebar sliders: raise "Speech margin" / "Minimum voice loudness" if
  background noise is counted as voice; raise "Match threshold" if strangers
  are recognized as someone registered.

## Limitations

- Noise detection is energy-based: loud non-speech sounds can count as voice.
- One speaker per clip works best; overlapping voices are not separated.
- Browser microphone needs localhost or https.
