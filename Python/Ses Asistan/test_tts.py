from TTS.api import TTS
import torch

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
# gpu parametresi artık önerilmiyor; bunun yerine:
tts.to("cuda" if torch.cuda.is_available() else "cpu")

tts.tts_to_file(
    text="Bugün yerelde çalışan konuşma sistemini test ediyoruz.Bu Sistemden çok memnunum ancak bir mallık yaptım ve ollamayı başka enve koydum",
    file_path="deneme4angry.wav",
    speaker_wav="mert.wav",   # <- liste değil, düz string
    language="tr",
    split_sentences=True,
    emotion="angry"
)

print("✅ Ses üretildi: deneme4angry.wav")
