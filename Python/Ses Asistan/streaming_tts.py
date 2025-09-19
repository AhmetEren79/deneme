import re
import queue
import threading
import time
import numpy as np
import sounddevice as sd
import torch
from TTS.api import TTS

# ---------- Ayarlar ----------
SPEAKER_WAV = "mert.wav"   # kendi referans kaydın (WAV, mono, 16k/22.05k)
LANG = "tr"
MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
# -----------------------------

def sentence_split(text: str) -> list[str]:
    """
    Noktalama işaretlerine göre Türkçe cümleleri kaba biçimde böler.
    Noktayı/ünlemi/soruyu korur.
    """
    # Çok uzun cümleler için iki nokta, noktalı virgül gibi işaretlere göre de bölebilirsin
    parts = re.split(r'([\.!?…])', text)
    sentences = []
    for i in range(0, len(parts), 2):
        s = parts[i].strip()
        if not s:
            continue
        end = parts[i+1] if i+1 < len(parts) else ""
        sentences.append((s + end).strip())
    return sentences

def audio_player(audio_q: queue.Queue, stop_event: threading.Event):
    """Kuyruktan gelen numpy dalgaları ses kartına çalar."""
    while not stop_event.is_set():
        try:
            wav, sr = audio_q.get(timeout=0.1)
        except queue.Empty:
            continue
        # sounddevice tek seferde çal
        sd.play(wav, samplerate=sr, blocking=True)  # blocking=True: cümle bitene kadar bekle
        audio_q.task_done()

# DEĞİŞİKLİK 1: Fonksiyona `sr` (sample_rate) parametresini ekledik.
def tts_worker(text: str, tts: TTS, device: str, audio_q: queue.Queue, sr: int):
    """
    Metni cümlelere böl, her cümleyi hemen üret ve oynatma kuyruğuna koy.
    """
    sentences = sentence_split(text)
    for sent in sentences:
        # Üretim (numpy dalga)
        # DEĞİŞİKLİK 2: Fonksiyon artık sadece `wav` döndürüyor.
        wav = tts.tts(
            text=sent,
            speaker_wav=SPEAKER_WAV,
            language=LANG
        )
        # Güvenlik: float32 ve tek boyut
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav[:, 0]

        # DEĞİŞİKLİK 3: Kuyruğa `wav` ile birlikte önceden aldığımız `sr`'yi koyuyoruz.
        audio_q.put((wav, sr))

def main():
    # Cihaz seçimi
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()} -> using {device}")

    # Modeli yükle ve cihaza taşı
    tts = TTS(model_name=MODEL, progress_bar=True)
    tts.to(device)

    # DEĞİŞİKLİK 4: Sample rate'i model yüklendikten sonra bir kere alıyoruz.
    output_sample_rate = tts.synthesizer.output_sample_rate

    # Örnek metin (istediğini ver)
    text = (
        "Merhaba Ahmet. XTTS v2 ile akışlı konuşma üretimini deniyoruz. "
        "Her cümle hazır olur olmaz çalınacak. Böylece bekleme süresi azalır."
    )

    # Kuyruk & thread'ler
    audio_q = queue.Queue(maxsize=4)  # küçük tutmak iyi
    stop_event = threading.Event()

    player_t = threading.Thread(target=audio_player, args=(audio_q, stop_event), daemon=True)
    player_t.start()

    try:
        # TTS üretimi ana threadde (istersen ayrı thread de yapabilirsin)
        # DEĞİŞİKLİK 5: `tts_worker` fonksiyonuna `output_sample_rate`'i gönderiyoruz.
        tts_worker(text, tts, device, audio_q, output_sample_rate)

        # Kuyruktaki son parçaların çalınmasını bekle
        audio_q.join()
    finally:
        stop_event.set()
        time.sleep(0.1)

if __name__ == "__main__":
    main()