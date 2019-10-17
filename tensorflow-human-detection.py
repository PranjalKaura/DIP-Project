# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time
import os
import pyautogui
import numpy as np
import imutils


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

def MotionDetection(currFrame, refFrame, minAreaThreshold, Frame):
    Motion = False
    SubtractedFrame = cv2.absdiff(currFrame, refFrame)
    ThresholdedFrame = cv2.threshold(SubtractedFrame, 25, 255, cv2.THRESH_BINARY)[1] # Thresholding a pixel difference value of 25
    ThresholdedFrame = cv2.dilate(ThresholdedFrame, None, iterations=2) # To fill in the white noises

    Contours = cv2.findContours(ThresholdedFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Contours = imutils.grab_contours(Contours)
    for conts in Contours:
        if cv2.contourArea(conts) < minAreaThreshold:
            Motion = True
            continue

        (x, y, w, h) = cv2.boundingRect(conts)
        cv2.rectangle(Frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Thresholded Frame", ThresholdedFrame)
    #print(Motion)
    return Frame, Motion


def BlurrImage(InputImage):
    Kernel = np.ones([5, 5])
    Kernel = Kernel/49

    BlurredImage = cv2.filter2D(InputImage, -1, Kernel)
    cv2.imshow("BlurredImage", BlurredImage)
    return BlurredImage


def HumanDetection(frame):    
        model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
        odapi = DetectorAPI(path_to_ckpt=model_path)
        threshold = 0.7
    #cap = cv2.VideoCapture(0)

    #while True:
        #r, frame = cap.read()
        #r = True
        #frame = pyautogui.screenshot()
        #frame = cv2.cvtColor(np.array(frame)[227:785, 367:1195], cv2.COLOR_RGB2BGR)
        img = cv2.resize(frame, (int(1280*1), int(720*1)))
        boxes, scores, classes, num = odapi.processFrame(img)

        # Visualization of the results of a detection.
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)

        #cv2.imshow("preview", img)
        #key = cv2.waitKey(1)
        #if key & 0xFF == ord('q'):
        #    break
        return img

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    #r, refFrame = cap.read()
    #refFrame = cv2.cvtColor(refFrame, cv2.COLOR_BGR2GRAY) #this has been done because colour is not a factor to detect motion
    #BlurredRefFrame = BlurrImage(refFrame)
    while(True):
        r, Frame = cap.read()
        Motion = True
        #GreyFrame = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY) 
        #BlurredFrame = BlurrImage(GreyFrame)
        #Frame, Motion = MotionDetection(BlurredFrame, BlurredRefFrame, 17000, Frame)
        if(Motion):
            Frame = HumanDetection(Frame)

        cv2.line(Frame, (0,int(Frame.shape[0]/2)), (Frame.shape[1],int(Frame.shape[0]/2)), (255,255,0), 5)
        cv2.imshow("InputImage", Frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()








