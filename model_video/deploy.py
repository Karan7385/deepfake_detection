#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
get_ipython().system('{sys.executable} -m pip install tensorflow opencv-python fastapi uvicorn nest-asyncio python-multipart')


# In[4]:


import tensorflow as tf
import cv2
import fastapi
import uvicorn
import nest_asyncio

print("All libraries imported successfully!")


# In[6]:


import tensorflow as tf

MODEL_PATH = r"C:\Users\LENOVO\OneDrive\Desktop\MAJOR-PROJECT\deepfake_detection_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model Loaded Successfully")


# In[7]:


print(model.input_shape)


# In[8]:


IMG_SIZE = 224


# In[9]:


import cv2
import numpy as np

def extract_frames(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)

        if len(frames) == max_frames:
            break

    cap.release()
    return np.array(frames)


# In[10]:


def predict_video(video_path):
    frames = extract_frames(video_path)

    if len(frames) == 0:
        return {"error": "No frames extracted"}

    # Check model input shape
    if len(model.input_shape) == 5:
        # Sequence model (LSTM / 3D CNN)
        frames = np.expand_dims(frames, axis=0)
        predictions = model.predict(frames)
        avg_prediction = predictions[0][0]

    else:
        # Frame-based CNN
        predictions = model.predict(frames)
        avg_prediction = np.mean(predictions)

    result = "FAKE" if avg_prediction > 0.5 else "REAL"

    return {
        "prediction": result,
        "confidence": float(avg_prediction)
    }


# In[12]:


video_path = r"C:\Users\LENOVO\OneDrive\Pictures\Saved Pictures\0icpg7s3wk.mp4"

result = predict_video(video_path)
print(result)


# In[13]:


from fastapi import FastAPI, File, UploadFile
import shutil
import os

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Deepfake Detection API Running"}


# In[14]:


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    
    temp_path = "temp_video.mp4"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_video(temp_path)

    os.remove(temp_path)

    return result


# In[ ]:


import nest_asyncio
import uvicorn

nest_asyncio.apply()

config = uvicorn.Config(app, host="127.0.0.1", port=8000)
server = uvicorn.Server(config)

await server.serve()


# In[ ]:




