import numpy as np
import cv2
import os
import sys
directory = '../videos'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
# checking if it is a file
    if os.path.isdir(f):
        for filename2 in os.listdir(f):
            f2 = os.path.join(f, filename2)
            if f2.endswith('.mp4'):
                #print(f2)
                cap = cv2.VideoCapture(f2)
                if cap.isOpened() == False:
                    print("error")
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if not ret: break
                    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imshow('frame',frame)
                    #cv2.waitKey(2)
                    
                    if cv2.waitKey(1) == ord('q'):
                        break
                    if cv2.waitKey(1) == ord('x'):
                        sys.exit()
                    
                cap.release()
                cv2.destroyAllWindows()

import subprocess

subprocess.call("playvid.py", shell=True)
