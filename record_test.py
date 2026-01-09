import sounddevice as sd
import soundfile as sf

print("Recording for 5 seconds... Speak now.")
audio = sd.rec(int(5 * 44100), samplerate=44100, channels=1)
sd.wait()
sf.write("manual_test.wav", audio, 44100)
print("Saved manual_test.wav")
