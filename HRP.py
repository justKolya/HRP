#!/usr/bin/python3
# -*- coding: utf-8 -*-

#імпорт необхідних модулів й бібліотек
import numpy as np
import os
import tensorflow as tf
import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import label_map_util as lm_util
import visualization_utils as vis_util

#створення екземпляру класу VideoCapture для захоплення зображення з камери
cap = cv2.VideoCapture(0)

#шляхи до графу та до файлу міток
graphPath = 'human_recognition_graph/frozen_inference_graph.pb'
labelPath = 'training/object-detection.pbtxt'

#правило завантаження замороженого графу у пам'ять
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(graphPath, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

#завантаження карти міток
label_map = lm_util.load_labelmap(labelPath)
categories = lm_util.convert_label_map_to_categories(
    label_map, max_num_classes=1, use_display_name=True)
category_index = lm_util.create_category_index(categories)

#допоміжна функція для створення "np.array"
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

#ініціалізація поштових реквізитів
fromEmail = '******************'
fromEmailPassword = '******************'
toEmail = '******************'

#функція посилання повідомлення на електронну пошту (gmail)
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

#процес детекції об'єктів й створення вікна програми
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

            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            # if abs(np.mean(scores)) > 0.01:
            #     sendEmail()

            cv2.imshow('HRP', cv2.resize(image_np, (800, 600)))
            cv2.imwrite('images/image.jpg', image_np)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
