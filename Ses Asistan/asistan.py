import whisper
# import ollama
# from TTS.api import TTS
import sounddevice as sd
from scipy.io.wavfile import write
import os

# Modelimizi bir kere yükleyelim ki her seferinde yüklenmesin.
# "base" modeli hız ve doğruluk arasında iyi bir dengedir.
print("Whisper modeli yükleniyor...")
whisper_model = whisper.load_model("base")
print("Whisper modeli yüklendi.")


def sesi_metne_cevir():
    """
    Bu fonksiyon, mikrofondan 5 saniye boyunca ses kaydedecek,
    bunu bir dosyaya yazacak ve Whisper kullanarak metne çevirecek.
    """
    print(">>> Adım 1: Ses dinleniyor...")

    fs = 44100  # Örnekleme frekansı
    seconds = 5  # Kayıt süresi
    filename = "kayit.wav"

    print("Kaydediliyor...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Kaydın bitmesini bekle
    print("Kayıt tamamlandı.")

    write(filename, fs, myrecording)  # Kaydı dosyaya yaz

    print("Metne çevriliyor...")
    result = whisper_model.transcribe(filename)
    kullanici_metni = result["text"]

    print(f"Kullanıcı dedi ki: '{kullanici_metni}'")

    # Geçici ses dosyasını sil
    os.remove(filename)

    return kullanici_metni


def ollama_ile_cevap_uret(metin):
    """
    Bu fonksiyon, gelen metni Ollama'ya gönderip bir cevap alacak.
    Şimdilik sadece örnek bir cevap döndürüyor.
    """
    print(">>> Adım 2: Ollama ile cevap üretiliyor...")
    # TODO: Ollama'ya bağlanıp cevap alma kodunu buraya ekleyeceğiz.
    asistan_cevabi = "Konya'da bugün hava güneşli ve sıcaklık 28 derece."
    print(f"Asistan cevap verdi: '{asistan_cevabi}'")
    return asistan_cevabi


def metni_sese_cevir_ve_oynat(metin):
    """
    Bu fonksiyon, Coqui TTS kullanarak metni sese çevirecek ve oynatacak.
    Şimdilik sadece bir mesaj yazdırıyor.
    """
    print(">>> Adım 3: Coqui TTS ile metin sese çevriliyor ve oynatılıyor...")
    # TODO: Coqui TTS ile metni sese çevirme ve oynatma kodunu buraya ekleyeceğiz.
    print(">>> Sesli cevap oynatıldı.")


def main():
    """
    Ana fonksiyon. Tüm iş akışını yönetir.
    """
    print("Asistan başlatıldı. Komut bekleniyor...")

    # 1. Kullanıcının sesini metne çevir
    kullanici_komutu = sesi_metne_cevir()

    # 2. Gelen metne göre bir cevap üret
    asistan_cevabi = ollama_ile_cevap_uret(kullanici_komutu)

    # 3. Üretilen cevabı seslendir
    metni_sese_cevir_ve_oynat(asistan_cevabi)

    print("İşlem tamamlandı.")


if __name__ == "__main__":
    main()