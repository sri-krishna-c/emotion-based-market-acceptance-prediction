import cv2
import sounddevice as sd
import soundfile as sf
import threading
import time

def record_voice(duration=5, filename="input.wav"):
    audio = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
    sd.wait()
    sf.write(filename, audio, 44100)

def record_face(duration=5, filename="face_video.mp4"):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

    start = time.time()
    while time.time() - start < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)

    cap.release()
    out.release()

def capture_both():
    t1 = threading.Thread(target=record_voice)
    t2 = threading.Thread(target=record_face)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    return "input.wav", "face_video.mp4"
