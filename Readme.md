# Sanskrit Speech-to-Text Translation Project

This project explores a speech-to-text translation pipeline for Sanskrit audio using Whisper, translation with Gemini, and voice synthesis using gTTS. It also includes a preprocessing step to enhance transcription quality by isolating vocals from background noise.

---

## 🔁 Project Structure

### 1. `Basic Pipeline.ipynb`

This notebook demonstrates a minimal working pipeline:

- Input: A clean `.wav` audio file.
- **Step 1**: Transcription using OpenAI's **Whisper** model.
- **Step 2**: Translation of transcribed Sanskrit using **Gemini**.
- **Step 3**: Generating audio of the translated text using **gTTS (Google Text-to-Speech)**.

> This notebook works well when the input audio is clean and without background disturbances.

---

### 2. `Preprocessing.ipynb`

To improve transcription in noisy environments, this notebook performs **preprocessing** using source separation techniques:

#### Tools Used:
- **Spleeter** (2-stem: vocal + accompaniment)
- **Demucs (htdemucs)**: A deep learning model for source separation, more robust than Spleeter in our testing.

#### Files Used:
All audio files are stored in the `audio/` folder:
- `original.wav` — Original audio input  
- `spleeter.wav` — Vocal separation using Spleeter  
- `demucs.wav` — Vocal separation using Demucs  

---

## 🎧 Audio Comparison

Original spoken Sanskrit line:
> **"यदा यदा हि धर्मस्य ग्लानिर्भवति भारत"**  
> Transcribed (no preprocessing):  
> _"ভি ডাঁট� Tibharata"_  
> _(severely degraded, likely due to background noise interference)_

---

### 🧪 After Preprocessing

| Method   | Transcription Output                                 |
|----------|------------------------------------------------------|
| **Spleeter** | _"जदायदाही भर्मस्ते नानेर्भोप्ती भारुता।"_             |
| **Demucs**   | _"जदा जदाहि धर्मस्त। लानेर भप्ति भारत।"_                |

### 📊 Observations

- **Demucs** produced significantly better transcription quality, maintaining much closer phonetic structure to the original Sanskrit.
- **Spleeter**, while useful, introduced distortion and lost clarity in key syllables.
- These issues may still be partially attributed to **unclear pronunciation** or **recording quality**, which should be addressed in future iterations.

---

## ✅ Conclusion

This project shows that **preprocessing audio with Demucs** greatly improves downstream transcription and translation in noisy conditions. Future improvements may include:

- Voice activity detection
- Enhanced noise profiling
- Training with more Sanskrit-specific data

---

### 🔗 References
- [Whisper by OpenAI](https://github.com/openai/whisper)
- [Spleeter by Deezer](https://github.com/deezer/spleeter)
- [Demucs by Facebook Research](https://github.com/facebookresearch/demucs)
- [gTTS: Google Text-to-Speech](https://pypi.org/project/gTTS/)

