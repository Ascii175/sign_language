from playsound import playsound
def function_sound():
    try:
        playsound('./speech.mp3')
    except : 
        print("ไม่เข้าใจเสียงที่นำเข้า")