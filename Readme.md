# Sanskrit Speech-to-Text Translation Project

This project aims to build a high-quality **Sanskrit speech-to-text translation system** with real-time processing capabilities, featuring accurate transcription, intelligent correction, and specialized Sanskrit voice synthesis.

---

## 🔁 Current Pipeline Architecture

### Real-time Processing Pipeline
Our current approach focuses on **real-time processing** with enhanced accuracy through multi-stage correction:

1. **DeepFilterNet** → Real-time noise reduction and speech enhancement
2. **Faster Whisper Large-v3** → Real-time Sanskrit speech recognition
3. **Gemini API** → Intelligent text correction for Sanskrit accuracy
4. **Sanskrit TTS** → Specialized Sanskrit text-to-speech synthesis

---

## 📁 Project Structure

### Core Implementation Files

#### `realtime_pipeline.ipynb`
The main implementation featuring our complete real-time STT pipeline:
- **Input**: Audio stream or WAV file with Sanskrit speech
- **Preprocessing**: DeepFilterNet for real-time noise reduction
- **Transcription**: Faster Whisper Large-v3 for low-latency speech recognition
- **Correction**: Gemini API for contextual Sanskrit text correction
- **TTS**: Sanskrit-tts package for audio generation

#### `deepfilter_preprocessing.ipynb`
Demonstrates DeepFilterNet preprocessing capabilities:
- Real-time noise reduction testing
- Comparison with original audio quality
- Performance benchmarking for streaming applications

### Reference Implementation Files

#### `Basic_Pipeline.ipynb` *(Legacy Reference)*
Original baseline pipeline using:
- **Whisper** for transcription
- **Gemini** for translation
- **gTTS** for speech synthesis

#### `Preprocessing.ipynb` *(Legacy Reference)*
Earlier preprocessing experiments with:
- **Spleeter**: 2-stem vocal/accompaniment separation
- **Demucs (htdemucs)**: Advanced source separation

*Note: We shifted from Spleeter/Demucs to DeepFilterNet for superior real-time performance and accuracy.*

---

## 🎧 Audio Files

### Current Test Files (in `audio/` folder):
- `test.wav`: Original Sanskrit audio input
- `deepfilter_processed.wav`: Audio after DeepFilterNet preprocessing


---

## 📊 Performance Analysis

### Processing Pipeline Comparison

**Input Sanskrit Verse:**
> **"यदा यदा हि धर्मस्य ग्लानिर्भवति भारत"**

| Method | Transcription Quality | Real-time Performance | Status |
|--------|----------------------|----------------------|---------|
| **DeepFilterNet + Faster Whisper** | High accuracy with contextual correction |  less latency | ✅ Current |
| Demucs + Whisper | Good quality but slower | ~2-3 seconds processing | 📚 Legacy |
| Spleeter + Whisper | Moderate quality | ~1-2 seconds processing | 📚 Legacy |
| No Preprocessing | Poor in noisy environments | Fast but inaccurate | ❌ Outdated |

### Key Improvements
- **Real-time Capability**: Achieved sub-500ms end-to-end processing latency
- **Enhanced Accuracy**: DeepFilterNet + Gemini correction significantly improves transcription quality
- **Scalability**: Modular design allows easy component upgrades
- **Sanskrit-specific**: Intelligent correction handles Sanskrit linguistic complexities

---

## 🛠️ Technical Stack

### Current Implementation
```
- DeepFilterNet (Real-time noise reduction)
- Faster Whisper Large-v3 (Real-time ASR)
- Google Gemini API (Text correction)
- sanskrit-tts (Sanskrit text-to-speech)
```

### Planned Enhancements
```
- Coqui xTTS-v2 (Fine-tuned Hindi voices for Sanskrit)
- Real-time streaming optimization
```

---

## 🌐 Live Demo

**Project Website**: [Sanskrit Speech-to-Text Research](https://rstar-910.github.io/SamskritaBharati/)

The website showcases our research methodology, results, and technical implementation details.



## 🔄 Development Timeline

| Date | Update Summary |
|------|---------------|
| **2025-06-28** | Shifted to DeepFilterNet for real-time noise reduction |
| **2025-06-30** | Implemented Faster Whisper Large-v3 for real-time ASR |
| **2025-07-02** | Integrated sanskrit-tts package for specialized TTS |
| **2025-07-13** | Added `realtime_pipeline.ipynb` for complete STT system |
| **2025-07-13** | Created `deepfilter_preprocessing.ipynb` for noise reduction testing |

---

## 🎯 Future Work

- **Coqui xTTS-v2 Integration**: Fine-tune on Hindi voice datasets for improved Sanskrit TTS
- **Mobile Deployment**: Optimize for mobile and edge devices
- **Sanskrit-specific ASR**: Fine-tune Faster Whisper on Sanskrit datasets
- **Voice Cloning**: Implement personalized Sanskrit voice synthesis
- **Multi-modal Interface**: Web and mobile applications for real-time Sanskrit tools

---


## 📚 References

### Current Technology Stack
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) - Real-time noise reduction
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) - Optimized Whisper implementation
- [Sanskrit TTS](https://github.com/SameeraMurthy/sanskrit-tts) - Specialized Sanskrit text-to-speech
- [Gemini API](https://ai.google.dev/gemini-api) - Text correction and language processing
- [Coqui TTS](https://github.com/coqui-ai/TTS) - Advanced text-to-speech synthesis

### Legacy References
- [Whisper (OpenAI)](https://github.com/openai/whisper)
- [Spleeter (Deezer)](https://github.com/deezer/spleeter)
- [Demucs (Meta)](https://github.com/facebookresearch/demucs)
- [gTTS](https://pypi.org/project/gTTS/)

---

