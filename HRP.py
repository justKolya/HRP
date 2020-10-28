#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import tensorflow as tf
import cv2
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from toGmail import *
from utils import label_map_util
from utils import visualization_utils as vis_util

cap = cv2.VideoCapture(0)

model = 'human_recognition_graph'
graphPath = model + '/frozen_inference_graph.pb'
labelPath = os.path.join('training', 'object-detection.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(graphPath, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(labelPath)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=1, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

fromEmail = '************'

fromEmailPassword = '*************'

toEmail = '**************'

def sendEmail():

	print("SmartSecurityCamera found person!")
	msgRoot = MIMEMultipart('related')
	msgRoot['Subject'] = 'Оповіщення SmartSecurityCamera'
	msgRoot['From'] = fromEmail
	msgRoot['To'] = toEmail

	msgAlternative = MIMEMultipart('alternative')
	msgRoot.attach(msgAlternative)
	msgText = MIMEText('На Вашій території знаходиться постороння людина!')
	msgAlternative.attach(msgText)

	smtp = smtplib.SMTP('smtp.gmail.com', 587)
	smtp.starttls()
	smtp.login(fromEmail, fromEmailPassword)
	smtp.sendmail(fromEmail, toEmail, msgRoot.as_string())
	smtp.quit()

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:

            ret, image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
           
            if abs(np.mean(scores)) > 0.01:
            	sendEmail()

            #print(abs(np.mean(scores)))

            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            cv2.imshow('HRP', cv2.resize(image_np, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
