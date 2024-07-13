import preprocessing.audio as Audio
import os

for file in os.listdir("samples/train"):
    file = os.path.join("samples/train", file)
    Audio.mp3_to_wav(file)