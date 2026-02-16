# Multimodal Emotional Intelligence System
## Voice and Face Emotion Fusion for Market Acceptance Prediction

---

### Overview

The Multimodal Emotional Intelligence System is an AI-based application designed to predict audience market acceptance by analyzing voice emotion, facial expressions, and spoken content. The system combines deep learning models with rule-based reasoning to provide accurate and human-like interpretation of audience feedback during product demonstrations, presentations, and evaluation sessions.

---

### Problem Statement

Traditional feedback mechanisms such as surveys and manual reviews are slow, biased, and often fail to capture real emotional intent. There is a need for an intelligent system that can automatically understand audience emotions in real time and predict whether a product or idea is positively, neutrally, or negatively received.

---

### Objective

To build a real-time multimodal AI system that:
- Captures voice and face input simultaneously
- Understands emotion from voice tone and facial expression
- Transcribes speech and analyzes context-aware sentiment
- Handles negation and conversational language correctly
- Produces a final Positive / Neutral / Negative market response with clear reasoning

---

### System Architecture

1. Face and voice are captured simultaneously  
2. Audio is transcribed using Whisper  
3. Voice emotion is detected using Wav2Vec2  
4. Facial emotion is recognized using DeepFace  
5. Text sentiment is analyzed using BERT  
6. Negation and keyword logic is applied  
7. All signals are fused using priority-based logic  
8. Final audience response is generated  

---

### Technology Stack

| Category | Technology |
|----------|------------|
| User Interface | Streamlit |
| Speech to Text | OpenAI Whisper (Offline) |
| Voice Emotion Detection | Wav2Vec2 |
| Facial Emotion Recognition | DeepFace |
| Text Sentiment Analysis | DistilBERT |
| Audio Processing | Librosa |
| Video Processing | OpenCV |
| Programming Language | Python |

---

### Methodology

The system follows a hybrid intelligence approach combining deep learning models and rule-based reasoning.

Voice analysis detects emotion and energy level from speech. Facial analysis identifies dominant facial emotion from video frames. Speech analysis converts speech to text and evaluates sentiment. Negation handling correctly interprets phrases such as "not good" or "don't like". Fusion logic combines all results using priority rules to avoid false predictions.

This approach ensures reliable and human-like emotional understanding.

---

### Output Categories

The final audience response is classified as:

- Positive – strong acceptance or satisfaction  
- Neutral – hesitation or mixed response  
- Negative – dissatisfaction or rejection  

Each output includes a detailed reasoning breakdown.

---

### Key Features

- Real-time face and voice capture  
- Offline speech transcription  
- Multimodal emotion fusion  
- Negation-aware sentiment analysis  
- Explainable decision logic  
- Audio and video playback  
- Retry capture functionality  
- Clean and professional user interface  

### Project Structure

voice_emotional_intelligence  
│  
├── app.py  
├── capture_module.py  
├── face_test.py  
├── record_test.py  
├── test_whisper.py  
├── .gitignore  
└── README.md  


### How to Run

Install required libraries:

pip install streamlit whisper transformers deepface opencv-python librosa sounddevice soundfile

Run the application:

streamlit run app.py

---

### Use Cases

- Product launch evaluation  
- Market research analysis  
- Customer feedback assessment  
- Presentation effectiveness measurement  
- User experience testing  

---

### Conclusion

This system demonstrates a practical implementation of multimodal emotional intelligence by combining voice, facial, and textual analysis into a single intelligent decision engine. It provides a reliable and explainable approach to predicting audience market acceptance in real-world scenarios.






