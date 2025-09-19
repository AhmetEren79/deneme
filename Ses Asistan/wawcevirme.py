from pydub import AudioSegment

audio = AudioSegment.from_file("MertSes.mp4")
audio = audio.set_frame_rate(16000).set_channels(1)
audio.export("mert.wav", format="wav")