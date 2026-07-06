# Cog*Lingo: Breaking Accent Bias with AI

> **Empowering communication by mitigating accent-based discrimination through AI-driven solutions.**


<div align="center">
  <video src="https://github.com/user-attachments/assets/f330cf9e-d7e3-45f5-aa92-49d391970594" controls autoplay muted width="600" alt="Demo Video">
    Your browser does not support the video.
  </video>
</div>

---

## 🚀 **Project Overview**
Cog*Lingo is an innovative tool addressing **accent bias and discrimination** by detecting and correcting mispronunciations. By leveraging cutting-edge AI, Cog*Lingo promotes inclusivity and supports confident communication for all.

### **Key Problem**
- **Prejudice against non-native accents** leads to inequities in:
  - Hiring practices  
  - Career advancement  
  - Housing opportunities  
  - Access to education  
- Existing tools fail to provide phoneme-level mispronunciation corrections.

### **Our Goal**
To create an accessible, intuitive solution to reduce **accent-based discrimination** and empower individuals to embrace their unique ways of speaking.

---

## 🏆 **Our Achievements**
- **92%+ Accuracy**: Fine-tuned the **Wav2Vec 2.0** model for detecting phoneme-level mispronunciations with industry-leading precision.
- **Phonetically Diverse Dataset**: Trained on 6,300 sentences spoken by 630 speakers from 8 U.S. dialect regions using the **TIMIT dataset**.
- **Dynamic Feedback**: Developed robust tools for mismatch identification and confidence scoring, ensuring actionable and reliable user insights.
- Beautiful, responsive and intuitive UI empowering effective learning

---

## 🧠 **How It Works**

  <p align="center">
    <img src="https://github.com/user-attachments/assets/b6431f57-4696-4ff6-9fa4-a83b99b55d82" alt="CogLingo Design Flowchart" width="1000">
  </p>


### **1. Data Collection**
- **Dataset**: TIMIT (Texas Instruments/MIT)
  - 6,300 phonetically rich sentences spoken by speakers from diverse dialects.
- **Process**:
  - Pre-process phonemes and align them with audio-text pairs.
  - Split data into training and validation sets.

---

### **2. Automated Speech Recognition (ASR)**
- Fine-tuned **Wav2Vec 2.0** to transcribe audio into phonemes.
- Example output for _"She had your dark suit in greasy wash water all year."_:

**Phoneme Output**:  
`sh-iy-hv-ae-dcl-d-y-e r-dcl-d-aa-r-kcl-k-s-u x-tcl-ih-n-gcl-g-r-iy-z-iy`

**Visualization**:  

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/ce41cdc4-0b59-4d15-b4d7-aae492a36546" alt="DTW Visualization" width="400">
      <br>
      <p>Dynamic Time Warping (DTW) Visualization</p>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/c55af6b4-41f8-4d8b-b826-af4e303c1cd1" alt="Phoneme Visualization" width="300">
      <br>
      <p>Phoneme Extraction Visualization</p>
    </td>
  </tr>
</table>

---

### **3. Phoneme Analysis, Scoring, and Feedback**
- **Phoneme Processing**:
  - Extract and compare phoneme sequences using **Dynamic Time Warping (DTW)**.
  - Identify mismatches with real-time confidence scoring.
- **Actionable Feedback**:
  - Matches and mismatches clearly highlighted.
  - Specific guidance on improving pronunciation.

**Example Feedback for User Sentence**:  
_"December and January are nice months to spend in Miami."_  
- Matches: ✅  
- Mismatches: ❌  
- **Detailed phoneme guidance for targeted improvements.**

  
  <p align="center">
    <img src="https://github.com/user-attachments/assets/669f1dc7-8362-497a-a5fd-bca5ee67be4b" alt="User Feedback Example" width="800">
  </p>

---

## 💻 **User Interface**
- **Features**:
  - Over 500 phonetically diverse prompts to challenge and improve users’ pronunciation skills.
  - Audio examples for each prompt.
  - Side-by-side display of user phoneme input and expected output.

---

## 🌟 **Future Directions**
- **Enhanced Feedback**: Provide specific directions for phoneme articulation.  
- **Multilingual Support**: Expand beyond English for broader accessibility.  
- **Integration with AR/VR**: Incorporate emerging technologies for immersive learning.  
- **Personalized Learning Paths**: Tailor exercises based on individual user progress.  
- **Advanced Analytics**: Use state-of-the-art ML techniques to refine model accuracy further.

---

## ⚙️ **Tech Stack**
- **Model**: Wav2Vec 2.0 (ASR Model)  
- **Data**: TIMIT dataset  
- **Techniques**: Phoneme extraction/analysis, Dynamic Time Warping, fine-tuning  
- **Tools**: PyTorch (GPU-enabled training), Gradio

---

## 🧩 **Getting Started**
1. Clone the repository:
   ```bash
   git clone https://github.com/SamGu-NRX/CogLingo.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```

---

## 📜 **License**
This project is licensed under the [MIT License](LICENSE).
