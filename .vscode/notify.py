import os
import platform
import winsound
from plyer import notification
import time
# Play a sound
if platform.system() == 'Darwin':  # macOS
    os.system('afplay /System/Library/Sounds/Glass.aiff')
elif platform.system() == 'Windows':  # Windows
    winsound.PlaySound('C:\\Windows\\Media\\notify.wav', winsound.SND_FILENAME)
elif platform.system() == 'Linux':  # Linux
    os.system('aplay /usr/share/sounds/freedesktop/stereo/complete.oga')

# Show a notification
notification.notify(
    title='Python script',
    message='Your python script has finished running!',
    timeout=10,
    app_icon='C:\\Users\\angel\\AppData\\Local\\Programs\\Python\\Python312\\Doc\\html\\_static\\og-image.ico'
)
