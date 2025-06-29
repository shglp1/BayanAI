📌 Overview  
This project is a voice-controlled AI chatbot named **Bayan** that listens to spoken commands and responds aloud in real-time. Bayan uses offline speech recognition (Vosk), AI-powered responses (Groq AI), and text-to-speech synthesis (gTTS) to communicate like a real assistant. Bayan was developed for the Saudi National Day to answer student questions and provide helpful information.


🏆 This project represents a significant achievement, developed over 35-40 hours of dedicated work, including testing, maintenance, and refactoring to ensure a robust and user-friendly experience.

🎯 Features  
✅ Offline Speech Recognition (No Internet Needed for STT)  
✅ AI-Powered Groq AI Responses  
✅ Natural-Sounding Voice Output (gTTS, Arabic supported)  
✅ Multi-language support planned (currently Arabic only)  

🛠️ Technologies Used  
1️⃣ Vosk (Offline Speech Recognition)  
2️⃣ Groq AI (Llama-3 model for replies)  
3️⃣ gTTS (Google Text-to-Speech, Arabic)  
4️⃣ pygame (Audio Playback)  

🏗️ Installation & Setup  
1️⃣ **Install Required Packages**  
   If you have `requirements.txt`, run:  
   ```bash
   pip install -r requirements.txt
   ```
   Or install manually:  
   ```bash
   pip install pygame SpeechRecognition vosk groq gTTS
   ```

2️⃣ **Set Up Groq API Key**  
   Create a `.env` file in the project root with:  
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

3️⃣ **Download Vosk Model (Arabic)**  
   Use the provided script to automatically download and extract the Arabic model:
   ```bash
   python download_model.py
   ```
   Or manually download from https://alphacephei.com/vosk/models and extract as `model-ar` in your project directory.


4️⃣ **Run the Chatbot**  
   ```bash
   python Bayan/Chatbot.py
   ```

🎙️ How Bayan Works  
1️⃣ Listens 🎤 → Captures your voice & converts it to text (Vosk)  
2️⃣ AI Responds 🤖 → Sends the text to Groq AI for Bayan's answer  
3️⃣ Speaks 🔊 → Uses gTTS to let Bayan read the response aloud  
4️⃣ Repeats 🔄 → Until you say "خروج" (exit in Arabic)  

🔧 Customization  
- Change Language: Download and use a different Vosk model  
- Switch TTS: Replace gTTS with another TTS engine if needed  
- Add Features: Extend Bayan's conversation logic as desired  

🏁 Conclusion  
Your Bayan AI chatbot is now fully functional! It can listen, understand, and talk back like a real assistant.


Enjoy your AI-powered talking robot, Bayan 🤖💬

🏗️ Developed by Eng. Salem Shurrab - Leader of GDSC at UPM - and his Best Friend Faris Bisher - The Legend







