# Sanskrit Speech-to-Text Translation Project

This project aims to build a high-quality **Sanskrit speech-to-text translation system**, with accurate transcription, translation, and voice synthesis. The current work demonstrates basic functionality and ongoing improvements through preprocessing and voice cloning techniques.

---

## 🔁 Project Structure

### 1. `Basic Pipeline.ipynb`

This notebook outlines the core pipeline:

- **Input**: A WAV file with Sanskrit speech.
- **Transcription**: Performed using OpenAI's **Whisper** model.
- **Translation**: Using **Gemini** to convert Sanskrit text into the target language.
- **Speech Synthesis**: Translated text is converted to speech using **gTTS (Google Text-to-Speech)**.

This baseline pipeline works well for clean audio but struggles with background noise or unclear pronunciation.

---

### 2. `Preprocessing.ipynb`

To enhance transcription quality in noisy environments, we tested **vocal separation** techniques.

#### Methods Used:
- **Spleeter**: A 2-stem vocal/accompaniment splitter.
- **Demucs (htdemucs)**: A more advanced model that produced noticeably better results in our tests.

#### Audio Files (in `audio/` folder):
- `original.wav`: Raw input audio.
- `spleeter.wav`: Output after applying Spleeter.
- `demucs.wav`: Output after applying Demucs.

---

## 🎧 Example Output Comparison

Spoken Sanskrit:  
> **"यदा यदा हि धर्मस्य ग्लानिर्भवति भारत"**

### Without Preprocessing
> **Transcription**:  
> _"ভি ডাঁট� Tibharata"_  
> _(Heavily degraded, possibly due to background noise)_

### After Spleeter
> _"जदायदाही भर्मस्ते नानेर्भोप्ती भारुता।"_

### After Demucs
> _"जदा जदाहि धर्मस्त। लानेर भप्ति भारत।"_

---

## 📊 Observations

- **Demucs** performed better than Spleeter in preserving phonetic structure.
- However, both methods still show errors — possibly due to unclear pronunciation or the limited support for Sanskrit in Whisper.
- These tests demonstrate that **preprocessing improves transcription quality**, but more work is needed.

---

## 🚧 Work in Progress

This project is still under active development. Our key goals include:

- Improving transcription accuracy through better preprocessing
- Fine-tuning or customizing Whisper for Sanskrit
- Integrating **voice cloning with Tortoise** to generate personalized, natural-sounding speech outputs
- Enhancing the pipeline for real-time or batch processing

---
## 🆕 Updates
| Date       | Update Summary                                                                                                                                           |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2025-06-20 | Added `sanskrit_voice_pipeline_with_demucs_whisper_gemini_elevenlabs.ipynb` for full conversation loop with Demucs, Whisper, Gemini, and ElevenLabs TTS. |
| 2025-06-20 | Created `conversation_sounds/` folder to store input/output audio and text samples for dialogue testing.                                                 |
| 2025-06-20 | Planning to explore open-source voice cloning alternatives to ElevenLabs due to API limitations.                                                         |
| 2025-06-30 | Used **VoiceFixer** and **Faster-Whisper** to enable near real-time speech enhancement and transcription.                                                |
| 2025-07-04 | Integrated **DeepFilterNet** to further improve audio clarity and boost processing speed.                                                                |
|  Present   | Currently experimenting with **Coqui XTTS** for Sanskrit voice synthesis as an open-source alternative.                                                  |


## 🔗 References

- [Whisper (OpenAI)](https://github.com/openai/whisper)
- [Spleeter (Deezer)](https://github.com/deezer/spleeter)
- [Demucs (Meta)](https://github.com/facebookresearch/demucs)
- [gTTS](https://pypi.org/project/gTTS/)
- [Tortoise TTS](https://github.com/neonbjb/tortoise-tts)
