from gtts import gTTS
import os
with open("../data/abc.txt", "r") as f:
    file = f.read()
    print(file)
    speech = gTTS(text=file, lang="en", slow=False)
    speech.save("voice.mp3")
    os.system("voice.mp3")

