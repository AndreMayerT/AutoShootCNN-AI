import numpy as np
import time
import cv2
import os
import mss
import win32api, win32con
import keys as k
import pyautogui
from threading import Thread


keys = k.Keys({})
fire_key = keys.mouse_lb_press
release_key = keys.mouse_lb_release
hold = 0.4

sct = mss.mss()
# aim limit: 810, 485, start: 796, 481
aim_x = 157
aim_y = 157
aim_xl = 163
aim_yl = 163
W, H = (320, 320)
monitor = {"top": 380, "left": 800, "width": W, "height": H}
threshold = 0.35
nms_threshold = 0.3

# classes directory
classesFile = 'C:\\Users\\andre\\PycharmProjects\\mouseTest\\coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# model directory
modelCfg = 'C:\\Users\\andre\\PycharmProjects\\mouseTest\\yolov3.cfg'

# weights directory
modelWeights = 'C:\\Users\\andre\\PycharmProjects\\mouseTest\\yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelCfg, modelWeights)

# run it with NVIDIA CUDA for better performance
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

def determine_movement(mid_x, mid_y,width=800, height=600):
    x_move = 0.5-mid_x
    y_move = 0.5-mid_y
    keys.keys_worker.SendInput(keys.keys_worker.Mouse(0x0001, -1*int(x_move*width), -1*int(y_move*height)))

def shoot(x, aim_x, y, aim_y, h, w):
    keys.directMouse(0, 0, fire_key)
    time.sleep(0.3)
    keys.directMouse(0, 0, release_key)
    #pyautogui.leftClick()

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIDs = []
    confs = []


    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)

            confidence = scores[classID]

            if confidence > threshold:
                if classID == 0:
                    w, h = int(detection[2] * wT), int(detection[3] * hT)
                    x, y = int((detection[0] * wT) - w / 2), int((detection[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    classIDs.append(classID)
                    confs.append(float(confidence))


    indices = cv2.dnn.NMSBoxes(bbox, confs, threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        #mid_x = (x + w) / 2
        #mid_y = (y + h) / 2

        #win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, int(x/1920*65535.0), int(mid_y/1080*65535.0))
        #win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(x_move), int(y_move), 0, 0)
        #print(x, y, w, h)
        #determine_movement(mid_x, mid_y, 1920, 1080)

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, f'{classNames[classIDs[i]].upper()} {int(confs[i]*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        #print(indices)
        shoot1 = Thread(target=shoot, args=[x, aim_x, y, aim_y, h, w])
        if x < aim_x and x + w > aim_xl and y < aim_y and y + h > aim_yl:
            shoot1.start()



while True:
    start = time.time()
    img = np.array(sct.grab(monitor))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    cv2.imshow('screen', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    end = time.time()
    TIME = end - start

    print("FPS:", 1/TIME)
    # Press "q" to quit
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
