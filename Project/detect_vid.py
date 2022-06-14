import cv2
import time
import numpy as np

cap = cv2.VideoCapture('input/video_1.mp4')

# get the video frames' width and height for proper saving of videos
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# create the `VideoWriter()` object
out = cv2.VideoWriter('output/video_result_1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))


def yolov3(frame):
    with open('files/object_detection_classes_coco.txt', 'r') as f:
        class_names = f.read().split('\n')

    # get a different color array for each of the classes
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

    # load the DNN model
    yolo_model = cv2.dnn.readNetFromDarknet('files/yolov3.cfg', 'files/yolov3.weights')

    ln = yolo_model.getLayerNames()
    ln = [ln[i-1] for i in yolo_model.getUnconnectedOutLayers()]

    print(ln)

    image_height, image_width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), crop=False)
    yolo_model.setInput(blob)

    layerOutputs = yolo_model.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    image_copy = np.copy(frame)

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.2:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([image_width, image_height,image_width, image_height])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                print(int(centerX - (width / 2)))
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, 2)
            text = f"{class_names[classIDs[i]]}"
            cv2.putText(image_copy, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image_copy

def caffe(frame):
    with open('files/classification_classes_ILSVRC2012.txt', 'r') as f:
        image_net_names = f.read().split('\n')
    
    class_names = [name.split(',')[0] for name in image_net_names]

    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
    # load the neural network model
    yolo_model = cv2.dnn.readNet(model='files/DenseNet_121.caffemodel', config='files/DenseNet_121.prototxt', framework='Caffe')

    ln = yolo_model.getLayerNames()
    ln = [ln[i-1] for i in yolo_model.getUnconnectedOutLayers()]

    print(ln)

    image_height, image_width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
    yolo_model.setInput(blob)

    layerOutputs = yolo_model.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    image_copy = np.copy(frame)

    # loop over each of the layer outputs
    # loop over the detections
	for i in np.arange(0, layerOutputs.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = layerOutputs[0, 0, i, 2]
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(layerOutputs[0, 0, i, 1])
			box = layerOutputs[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    return image_copy

def tensor(image):
    with open('files/object_detection_classes_coco.txt', 'r') as f:
        class_names = f.read().split('\n')

    # get a different color array for each of the classes
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

    # load the DNN model
    mobile_net_model = cv2.dnn.readNet(model='files/frozen_inference_graph.pb',
                            config='files/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                            framework='TensorFlow')

    #image = cv2.cvtColor(imageArg, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape
    # create blob from image
    blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123))
    # create blob from image
    mobile_net_model.setInput(blob)
    # forward pass through the mobile_net_model to carry out the detection
    output = mobile_net_model.forward()
    # loop over each of the detection
    for detection in output[0, 0, :, :]:
        # extract the confidence of the detection
        confidence = detection[2]
        # draw bounding boxes only if the detection confidence is above...
        # ... a certain threshold, else skip
        if confidence > .25:
            # get the class id
            class_id = detection[1]
            # map the class id to the class
            class_name = class_names[int(class_id)-1]
            color = COLORS[int(class_id)]
            # get the bounding box coordinates
            box_x = detection[3] * image_width
            box_y = detection[4] * image_height
            # get the bounding box width and height
            box_width = detection[5] * image_width
            box_height = detection[6] * image_height
            # draw a rectangle around each detected object
            cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
            # put the FPS text on top of the frame
            cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return(image)
# detect objects in each frame of the video

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
        image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        n_image = caffe(image)
        cv2.imshow('image', n_image)
        out.write(image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
