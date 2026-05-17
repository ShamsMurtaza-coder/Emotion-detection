import cv2
import numpy as np
import tensorflow as tf

print("Loading model... please wait")
model = tf.keras.models.load_model('model/emotion_model.h5')
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
print("Model loaded! Starting camera...")

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
COLORS   = [
    (0,   0,   255),
    (0,   128, 0  ),
    (128, 0,   128),
    (0,   200, 0  ),
    (255, 100, 0  ),
    (0,   165, 255),
    (160, 160, 160),
]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

if not cap.isOpened():
    print("ERROR: Could not open webcam!")
    exit()

print("Camera started! Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(48,48))

    for (x, y, w, h) in faces:
        face_roi        = gray[y:y+h, x:x+w]
        face_resized    = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized.astype('float32') / 255.0
        face_input      = np.expand_dims(face_normalized, axis=[0, -1])

        predictions  = model.predict(face_input, verbose=0)[0]
        emotion_idx  = np.argmax(predictions)
        emotion_name = EMOTIONS[emotion_idx]
        confidence   = predictions[emotion_idx] * 100
        color        = COLORS[emotion_idx]

        # Face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Label background + text
        label      = f"{emotion_name}  {confidence:.0f}%"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame,
                      (x, y - label_size[1] - 16),
                      (x + label_size[0] + 10, y),
                      color, -1)
        cv2.putText(frame, label, (x+5, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Confidence bars
        bar_x     = min(x + w + 15, frame.shape[1] - 210)
        bar_max_w = 140

        for i, (emo, prob) in enumerate(zip(EMOTIONS, predictions)):
            bar_y  = y + i * 26
            filled = int(prob * bar_max_w)

            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + bar_max_w, bar_y + 18), (50,50,50), -1)
            if filled > 0:
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + filled, bar_y + 18), COLORS[i], -1)
            cv2.putText(frame, f"{emo[:4]} {prob*100:.0f}%",
                        (bar_x + bar_max_w + 5, bar_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)

    # Top banner
    cv2.rectangle(frame, (0,0), (frame.shape[1], 36), (30,30,30), -1)
    cv2.putText(frame, "Facial Emotion Detection  |  Press Q to quit",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Faces: {len(faces)}",
                (frame.shape[1]-110, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,100), 2)

    cv2.imshow('Facial Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")