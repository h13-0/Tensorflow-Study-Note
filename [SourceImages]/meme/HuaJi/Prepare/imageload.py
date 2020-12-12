import os
import random
import time

while True:        
    try:
        os.rename('OGC.gif',str(random.randint(88888,999999)) + 'OGC' + str(random.randint(88888,999999)) + '.gif')
    except BaseException:
        time.sleep(0.1)
        
    try:
        os.rename('OIP.jpg',str(random.randint(88888,999999)) + 'OIP' + str(random.randint(88888,999999)) + '.jpg')
    except BaseException:
        time.sleep(0.1)
        
    try:
        os.rename('download.jpg',str(random.randint(88888,999999)) + 'download' + str(random.randint(88888,999999)) + '.jpg')
    except BaseException:
        time.sleep(0.1)
        
    try:
        os.rename('timg.gif',str(random.randint(88888,999999)) + 'timg' + str(random.randint(88888,999999)) + '.gif')
    except BaseException:
        time.sleep(0.1)