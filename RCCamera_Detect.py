import cv2
import numpy as np
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

CLASSES = ["Dead End", "Forward Sign", "No Entry", "Proceed Left", "Proceed Right", "Stop Sign"]

# Load model
interpreter = Interpreter(model_path="/home/pi/Desktop/py/Aaron/street_signs/best_float32.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Setup camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

print("Cam on")

while True:
    ret, frame =cap.read()
    if not ret:
        print("Failed to grab Frame")
        break
    
    h, w = frame.shape[:2]
    
    img = cv2.resize(frame, (320, 320))
    img_input = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    predictions = output[0].T
    for pred in predictions:
        confidence = pred[4:].max()
        if confidence > 0.75:
            class_id = pred[4:].argmax()
            
            # Get box coordinates
            cx, cy, bw, bh = pred[:4]
            
            
            x1 = int((cx - bw/2)*w)
            y1 = int((cy - bh/2)*h)
            x2 = int((cx + bw/2)*w)
            y2 = int((cy + bh/2)*h)
            
            #Draw box and label on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            label = f"{CLASSES[class_id]} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            print(f"Detected: {CLASSES[class_id]} ({confidence:.2f})")

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("done")
cap.release()
cv2.destroyAllWindows()