import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os

# Charger modèle
interpreter = tflite.Interpreter(model_path="model_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Charger labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Dossier images
image_folder = "images"

for image_name in os.listdir(image_folder):

    image_path = os.path.join(image_folder, image_name)

    frame = cv2.imread(image_path)

    if frame is None:
        continue

    height, width, _ = frame.shape

    # Prétraitement
    img = cv2.resize(frame, (320, 320))
    input_data = np.expand_dims(img, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    print(f"\nImage: {image_name}")

    for i in range(len(scores)):

        if scores[i] > 0.5:

            ymin, xmin, ymax, xmax = boxes[i]

            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)

            label = labels[int(classes[i])]
            score = scores[i]

            print(f"Detected: {label} ({score:.2f})")

            # Dessiner rectangle
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            cv2.putText(frame, label, (xmin, ymin-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Affichage image
    cv2.imshow("Detection", frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()