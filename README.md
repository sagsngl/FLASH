# FLASH
Main Code

![ezgif-3-d4d443d865](https://user-images.githubusercontent.com/87820561/187310936-70a415fd-bc77-42ce-89e2-873a66c291d7.gif)

This code is meant to count cars entering a garage

It opens 2 frames, one showing the tracking boxes and the other showing a counter as when it detects a car enter the garage

This is done by tracking the corners of the bounding boxes from when the car is detected and when it disappears from the frame

Command to be used:

- py main_api_v3.py -m best_openvino_2021.4_6shave.blob -c best.json 
