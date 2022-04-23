from IPython.display import Audio
import gtts
import time
from playsound import playsound
from gtts import gTTS
test = "อ้วน"
tts = gTTS(test, lang='th')
tts.save('test.mp3')
playsound('test.mp3')	
# word = "test"
# test = 1
# start = time.time()
# async def async_sleep():
#     await asyncio.sleep(1)
# while test < 5:
#     if word == "test2":
#        print("FF")
#     elif word == "test":
#         test += 1 
#         async def func1(a):
#                 playsound('./speech.mp3')
             
#Audio('speech.mp3', autoplay=True)

