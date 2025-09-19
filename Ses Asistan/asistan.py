import whisper
import ollama
from TTS.api import TTS
import sounddevice as sd
from scipy.io.wavfile import write
import os
import soundfile as sf
import keyboard

# Modelimizi bir kere yükleyelim ki her seferinde yüklenmesin.
# "base" modeli hız ve doğruluk arasında iyi bir dengedir.
print("Whisper modeli yükleniyor...")
whisper_model = whisper.load_model("base")
print("Whisper modeli yüklendi.")

# Coqui TTS modelini yüklüyoruz. xtts_v2 modeli ses klonlama için harikadır.
print("Coqui TTS modeli yükleniyor...")
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
print("Coqui TTS modeli yüklendi.")


def sesi_metne_cevir():
    """
    Bu fonksiyon, mikrofondan 5 saniye boyunca ses kaydedecek,
    bunu bir dosyaya yazacak ve Whisper kullanarak metne çevirecek.
    """
    print(">>> Adım 1: Ses dinleniyor...")

    fs = 44100  # Örnekleme frekansı
    seconds = 3  # Kayıt süresi
    filename = "kayit.wav"

    print("Kaydediliyor...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Kaydın bitmesini bekle
    print("Kayıt tamamlandı.")

    write(filename, fs, myrecording)  # Kaydı dosyaya yaz

    print("Metne çevriliyor...")
    result = whisper_model.transcribe(filename, language="tr")
    kullanici_metni = result["text"]

    print(f"Kullanıcı dedi ki: '{kullanici_metni}'")

    # Geçici ses dosyasını sil
    os.remove(filename)

    return kullanici_metni


def ollama_ile_cevap_uret(metin):
    """
    Bu fonksiyon, gelen metni sistem talimatıyla birlikte Ollama'ya gönderip
    kısa ve Türkçe bir cevap alacak.
    """
    print(">>> Adım 2: Ollama ile cevap üretiliyor...")

    try:
        response = ollama.chat(
            model='llama3.1     ',
            messages=[
                # YENİ, DAHA ANLAŞILIR VE NET SİSTEM TALİMATI
                {
                    'role': 'system',
                    'content': 'Sadece Türkçe konuşuyorsun ve 1 cümle kuruyorsun.kısa öz ve kibar bir dille cevap ver.',
                },
                # KULLANICININ MESAJI
                {
                    'role': 'user',
                    'content': metin,
                },
            ]
        )
        asistan_cevabi = response['message']['content']
        print(f"Asistan cevap verdi: '{asistan_cevabi}'")
        return asistan_cevabi

    except Exception as e:
        print(f"Ollama'ya bağlanırken bir hata oluştu: {e}")
        hata_mesaji = "Üzgünüm, şu an cevap veremiyorum. Lütfen Ollama'nın çalıştığından emin olun."
        return hata_mesaji


def metni_sese_cevir_ve_oynat(metin):
    """
    Bu fonksiyon, Coqui TTS kullanarak metni sese çevirecek ve oynatacak.
    Referans olarak 'hedef_ses.wav' dosyasını kullanacak.
    """
    print(">>> Adım 3: Coqui TTS ile metin sese çevriliyor...")

    output_filename = "yanit.wav"
    reference_voice_path = "hedef_ses.wav"  # Referans ses dosyamızın adı

    try:
        # Metni, referans sesi kullanarak dosyaya sentezle
        tts_model.tts_to_file(
            text=metin,
            speaker_wav=reference_voice_path,
            language="tr",  # Dil olarak Türkçe'yi belirtiyoruz
            file_path=output_filename
        )

        print("Ses dosyası oluşturuldu. Şimdi oynatılıyor...")

        # Oluşturulan ses dosyasını oynat
        data, fs = sf.read(output_filename, dtype='float32')
        sd.play(data, fs)
        sd.wait()

        print(">>> Sesli cevap oynatıldı.")

        # Geçici yanıt dosyasını sil
        os.remove(output_filename)

    except Exception as e:
        print(f"Coqui TTS ile ses üretirken bir hata oluştu: {e}")


def main():
    """
    Ana fonksiyon. Sürekli çalışır, 'space' tuşuna basılmasını bekler,
    işlemi gerçekleştirir      ve tekrar beklemeye geçer. 'esc' tuşu ile çıkar.
    """
    print("Asistan başlatıldı. Çıkmak için 'esc' tuşuna, konuşmak için 'space' tuşuna basın.")

    while True:
        try:
            # Kullanıcının bir tuşa basmasını bekle
            print("\nDinleme moduna geçmek için 'space' tuşuna basın...")
            keyboard.wait('space')

            print("Tuşa basıldı, dinleniyor...")

            # 1. Kullanıcının sesini metne çevir
            kullanici_komutu = sesi_metne_cevir()

            # Eğer kullanıcı bir şey söylemediyse döngünün başına dön
            if not kullanici_komutu.strip():
                print("Bir şey söylemediğiniz için işlem iptal edildi.")
                continue

            # 2. Gelen metne göre bir cevap üret
            asistan_cevabi = ollama_ile_cevap_uret(kullanici_komutu)

            # 3. Üretilen cevabı seslendir
            metni_sese_cevir_ve_oynat(asistan_cevabi)

        except KeyboardInterrupt:
            # Ctrl+C ile çıkış yapıldığında
            print("\nProgram sonlandırılıyor.")
            break
        except Exception as e:
            # 'esc' tuşuna basıldığında keyboard.wait bir hata fırlatır,
            # bunu kullanarak döngüden çıkabiliriz.
            if "esc" in str(e).lower():
                print("\n'esc' tuşuna basıldı, program kapatılıyor.")
                break
            else:
                print(f"Beklenmedik bir hata oluştu: {e}")
                break


if __name__ == "__main__":
    main()