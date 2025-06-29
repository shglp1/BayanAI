import os
import json
import re
import tempfile
import pygame
import speech_recognition as sr
from vosk import Model, KaldiRecognizer
import groq
from gtts import gTTS
import sys
import threading
from dotenv import load_dotenv  # Added for .env support


# Fix UnicodeEncodeError
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables from .env file
load_dotenv()

# ✅ Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
client = groq.Groq(api_key=GROQ_API_KEY)


class SpeechRecognizer:
    """
    التعرف على الصوت باستخدام Vosk وتحويله إلى نص
    """

    def __init__(self, tts):
        self.vosk_model_path = os.path.expanduser("model-ar")
        self._check_model_exists()
        self.vosk_model = Model(self.vosk_model_path)
        self.recognizer = sr.Recognizer()
        self.tts = tts  # TextToSpeech instance
        self.music_thread = None  # Thread for playing music
        self._stop_music_flag = False  # Renamed to avoid conflict

    def _check_model_exists(self):
        """
        التحقق من توفر نموذج Vosk للغة العربية
        """
        if not os.path.exists(self.vosk_model_path):
            print("لم يتم العثور على نموذج Vosk العربي. يرجى تحميله.")
            raise FileNotFoundError(
                "لم يتم العثور على نموذج Vosk العربي. يرجى تحميله.")

    def play_mp3(self):
        """
        Loads and plays an MP3 music file.

        :param file_path: Path to the MP3 file.
        """
        if not os.path.exists("music.mp3"):
            print("music.mp3 not found. Skipping music playback.")
            return
        pygame.mixer.init()
        pygame.mixer.music.load("music.mp3")
        pygame.mixer.music.play()

        # Keep the program running while music is playing
        while pygame.mixer.music.get_busy() and not self._stop_music_flag:
            pygame.time.Clock().tick(10)  # Small delay to avoid excessive CPU usage

    def start_music(self):
        """
        Start playing the music in a separate thread.
        """
        self._stop_music_flag = False  # Reset the stop flag
        self.music_thread = threading.Thread(target=self.play_mp3)
        self.music_thread.start()
        print("Music started playing in the background.")

    def stop_music(self):
        """
        Stop the currently playing music.
        """
        if self.music_thread and self.music_thread.is_alive():
            self._stop_music_flag = True  # Set the stop flag
            pygame.mixer.music.stop()  # Stop the music
            self.music_thread.join()  # Wait for the thread to finish
            print("Music stopped.")
        else:
            print("No music is currently playing.")

    def listen(self) -> str:
        """
        التقاط الصوت وتحويله إلى نص
        """
        if not sr.Microphone.list_microphone_names():
            self.tts.speak(
                "لم يتم العثور على ميكروفون. يرجى التحقق من الإعدادات الصوتية.")
            raise RuntimeError(
                "لم يتم العثور على ميكروفون. يرجى التحقق من الإعدادات الصوتية.")

        with sr.Microphone(sample_rate=44100, chunk_size=1000) as source:
            print("ضبط الصوت للضوضاء المحيطة...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.tts.speak("اســـألــنـــي...")

            try:
                audio = self.recognizer.listen(
                    source, timeout=None, phrase_time_limit=None)
                rec = KaldiRecognizer(self.vosk_model, 44100)
                print("جاري معالجة سؤالك...")
                self.start_music()
                rec.AcceptWaveform(audio.get_wav_data())
                final_result = json.loads(rec.FinalResult())
                recognized_text = final_result.get("text", "").strip()

                if recognized_text:
                    print(f"تم التعرف على: {recognized_text}")
                    return recognized_text
                self.tts.speak("لم يتم اكتشاف كلام. حاول مرة أخرى.")
                return ""

            except (sr.UnknownValueError, sr.RequestError) as e:
                self.tts.speak(f"خطأ أثناء التعرف على الصوت: {str(e)}")
                return ""


class GroqAI:
    """
    التفاعل مع Groq API والحصول على ردود نصية
    """

    def __init__(self, tts):
        self.tts = tts  # TextToSpeech instance

    def clean_response(self, response: str) -> str:
        """
        تنظيف الاستجابة من الرموز غير المرغوب فيها
        """
        return re.sub(r'[*]', '', response)

    def get_response(self, prompt: str) -> str:
        """
        إرسال المدخلات إلى Groq API واسترجاع الرد
        """
        if not prompt:
            self.tts.speak("لم أفهمك، من فضلك حاول مرة أخرى.")
            return "لم أفهمك، من فضلك حاول مرة أخرى."

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "أنت مساعد ذكي اسمك بيان. تم تطويرك بواسطة نادي هندسة البرمجيات و نادي جـــوجل و نادي الذكاء الاصطناعي."
                    },
                    {
                        "role": "user",
                        "content": prompt + " أريد الجواب أن يكون دائماً بطول جملة واحدة فقط."
                    }
                ]
            )
            if response.choices:
                full_response = self.clean_response(
                    response.choices[0].message.content.strip())
                # Split the response into sentences
                sentences = re.split(r'(?<=[.!?]) +', full_response)
                # Join the first two sentences
                if len(sentences) > 1:
                    return ' '.join(sentences[:2]).strip()
                else:
                    return sentences[0].strip()
            self.tts.speak("لم يتم تلقي استجابة من Groq AI.")
            return "لم يتم تلقي استجابة من Groq AI."
        except Exception as e:
            self.tts.speak(f"خطأ أثناء توليد الاستجابة: {str(e)}")
            return f"خطأ أثناء توليد الاستجابة: {str(e)}"


class TextToSpeech:
    """
    تحويل النص إلى صوت وتشغيله باستخدام pygame
    """

    def __init__(self):
        pygame.mixer.init()

    def speak(self, text: str):
        """
        تحويل النص إلى كلام باستخدام gTTS
        """
        if not text:
            return

        temp_audio_path = os.path.join(tempfile.gettempdir(), "speech.mp3")

        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError as e:
                self.speak(f"لم يتمكن البرنامج من حذف الملف القديم: {e}")

        try:
            tts = gTTS(text=text, lang="ar")
            tts.save(temp_audio_path)

            if not pygame.mixer.get_init():
                pygame.mixer.init()

            pygame.mixer.music.load(temp_audio_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                continue

            pygame.mixer.quit()
        except (OSError, pygame.error) as e:
            self.speak(f"خطأ أثناء تشغيل الصوت: {e}")


class Chatbot:
    """
    إدارة التفاعل مع المستخدم
    """

    def __init__(self):
        self.tts = TextToSpeech()
        self.speech_recognizer = SpeechRecognizer(self.tts)
        self.ai = GroqAI(self.tts)

    def start(self):
        """
        بدء المحادثة الصوتية
        """
        self.tts.speak("جاهز للمحادثة! قل 'خروج' لإنهاء المحادثة.")
        while True:
            user_input = self.speech_recognizer.listen()
            if user_input and "خروج" in user_input:
                self.tts.speak("إلى اللقاء!")
                break
            if user_input:
                response = self.ai.get_response(user_input)
                self.tts.speak(response)


# Run the chatbot
if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.start()
