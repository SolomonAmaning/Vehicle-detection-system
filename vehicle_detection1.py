import cv2
import numpy as np
# Load the pre-trained YOLOv4 model for vehicle detection
net = cv2.dnn.readNetFromDarknet('C:/Users/Administrator/Documents/JIO INSTITUTE/computer vision/Autonomous_vehicles/yolov4.cfg',
                                 'C:/Users/Administrator/Documents/JIO INSTITUTE/computer vision/Autonomous_vehicles/yolov4.weights')

# Reading the input image
image = cv2.imread('vehicles.jpg')


# Open a video file or stream
cap = cv2.VideoCapture('vehicle_movement.mp4')

# Preprocess the input image
blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)

# Preprocess the video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    blob = cv2.dnn.blobFromImage(
        frame, 1/255, (416, 416), swapRB=True, crop=False)

# Feed the input image through the YOLO model
net.setInput(blob)
detections = net.forward()

# # Feed the video frames through the YOLO model
# net.setInput(blob)
# detections = net.forward()

# Postprocess the detection results for the input image
boxes = []
confidences = []
class_ids = []

for detection in detections:
    scores = detection[5:]
    class_id = np.argmax(scores)
    confidence = scores[class_id]
    if confidence > 0.5 and class_id == 2:  # 2 is the class ID for vehicles in YOLOv4
        center_x = int(detection[0] * image.shape[1])
        center_y = int(detection[1] * image.shape[0])
        width = int(detection[2] * image.shape[1])
        height = int(detection[3] * image.shape[0])
        left = int(center_x - width/2)
        top = int(center_y - height/2)
        boxes.append([left, top, width, height])
        confidences.append(float(confidence))
        class_ids.append(class_id)

indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Extract the vehicle bounding boxes and classification labels for the input image
# vehicle_boxes = []
# vehicle_labels = []

# for i in indices:
#     i = i[0]
#     if class_ids[i] == 2:  # 2 is the class ID for vehicles in YOLOv4
#         box = boxes[i]
#         left = box[0]
#         top = box[1]
#         width = box[2]
#         height = box[3]
#         vehicle_boxes.append((left, top, left+width, top+height))
#         vehicle_labels.append('vehicle')

# Postprocess the detection results for the video frames
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

   # Preprocess the current video frame
# if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
#     blob = cv2.dnn.blobFromImage(
#         frame, 1/255, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)
#     # Your code for processing the frame goes here

# # blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

# # Feed the current video frame through the YOLO model
# net.setInput(blob)
# detections = net.forward()

# # Postprocess the detection results for the current video frame
# boxes = []
# confidences = []
# class_ids = []
# for detection in detections:
#     scores = detection[5:]
#     class_id = np.argmax(scores)
#     confidence = scores[class_id]
#     if confidence > 0.5 and class_id == 2:  # 2 is the class ID for vehicles in YOLOv4
#         center_x = int(detection[0] * frame.shape[1])
#         center_y = int(detection[1] * frame.shape[0])
#         width = int(detection[2] * frame.shape[1])
#         height = int(detection[3] * frame.shape[0])
#         left = int(center_x - width/2)
#         top = int(center_y - height/2)
#         boxes.append([left, top, width, height])
#         confidences.append(float(confidence))
#         class_ids.append(class_id)

# indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Extract the vehicle bounding boxes and classification labels for the current video frame
current_vehicle_boxes = []
current_vehicle_labels = []

for i in indices:
    i = i[0]
    if class_ids[i] == 2:  # 2 is the class ID for vehicles in YOLOv4
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        current_vehicle_boxes.append((left, top, left+width, top+height))
        current_vehicle_labels.append('vehicle')

    # Display the current video frame with vehicle bounding boxes and classification labels
for box, label in zip(current_vehicle_boxes, current_vehicle_labels):
    left, top, right, bottom = box
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(frame, label, (left, top-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # cv2.imshow('Vehicle Detection', frame)

# Exit the video display window when the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the resources used for the video analysis
cap.release()
cv2.destroyAllWindows()

'''checking something here, kindly ignore the codes below until it works well'''

# Draw the bounding boxes on the image
for box, label in zip(vehicle_boxes, vehicle_labels):
    left, top, right, bottom = box
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image, label, (left, top-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the image to the current directory
cv2.imwrite('output.jpg', image)

'''checking the video without running all'''


# Open input video file
cap = cv2.VideoCapture('vehicle_movement.mp4')

# Get input video stream properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

# Loop through each frame in the input video stream
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame here
    processed_frame = frame  # Replace this with your own processing code

    # Write processed frame to output video stream
    out.write(processed_frame)

    # Display processed frame (optional)
    cv2.imshow('Processed Frame', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release input and output video streams
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()


'''Using another preprocessing'''

# Load YOLOv4 model
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

# Load class names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set input and output video paths
input_path = 'vehicle_movement.mp4'
output_path = 'output_new.mp4'

# Open input video stream
cap = cv2.VideoCapture(input_path)

# Get input video stream properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Loop through each frame in the input video stream
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255, (416, 416), swapRB=True, crop=False)

    # Set input to YOLOv4 model
    net.setInput(blob)

    # Forward pass through YOLOv4 model
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Postprocess detections
    boxes = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes around detected objects
    for i in indices:
        i = i[0]
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Write processed frame to output video stream
    out.write(frame)

    # Display processed frame (optional)
    cv2.imshow('Processed Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release input and output video streams
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()


'''Third part'''


# Load YOLOv4 model
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

# Load class names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set input and output video paths
input_path = 'vehicle_movement.mp4'
output_path = 'output_new.mp4'

# Open input video stream
cap = cv2.VideoCapture(input_path)

# Get input video stream properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Define ROI (Region of Interest)
roi = [(int(width * 0.45), int(height * 0.6)), (int(width * 0.55), int(height * 0.6)),
       (int(width * 0.8), int(height)), (int(width * 0.2), int(height))]

# Initialize vehicle counter
vehicle_counter = 0

# Loop through each frame in the input video stream
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255, (416, 416), swapRB=True, crop=False)

    # Set input to YOLOv4 model
    net.setInput(blob)

    # Forward pass through YOLOv4 model
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Postprocess detections
    boxes = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes around detected objects
    for i in indices:
        i = i[0]
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Check if object center is within ROI
        object_center_x = int(x + w / 2)
        object_center_y = int(y + h / 2)
        point = (object_center_x, object_center_y)

        if cv2.pointPolygonTest(roi_contour, point, False) >= 0:
            # Object is within ROI, update count
            object_count += 1

        # Write object count on frame
cv2.putText(frame, f"Object count: {object_count}",
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Write processed frame to output video stream
out.write(frame)

# Display processed frame (optional)
cv2.imshow('Processed Frame', frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release input and output video streams
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()
