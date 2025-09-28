# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

## program
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
faceImage = cv2.imread('my.jpg')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")

import cv2
import matplotlib.pyplot as plt

# Load image with unchanged flag
glassPNG = cv2.imread("sunglass.png", cv2.IMREAD_UNCHANGED)

print("Image shape:", glassPNG.shape)

if glassPNG.shape[2] == 4:
    # If PNG has alpha channel
    glassBGR = glassPNG[:, :, :3]   # BGR
    glassMask1 = glassPNG[:, :, 3]  # Alpha channel
else:
    # If PNG has no alpha channel
    glassBGR = glassPNG
    # Create a dummy mask (all opaque)
    glassMask1 = 255 * np.ones(glassPNG.shape[:2], dtype=np.uint8)

plt.imshow(cv2.cvtColor(glassBGR, cv2.COLOR_BGR2RGB))
plt.title("Sunglasses")
plt.show()


plt.figure(figsize=[15,15])

plt.subplot(121)
plt.imshow(glassBGR[:,:,::-1])  # BGR → RGB
plt.title('Sunglass Color channels')

glassGray = cv2.cvtColor(glassBGR, cv2.COLOR_BGR2GRAY)
_, glassMask1 = cv2.threshold(glassGray, 240, 255, cv2.THRESH_BINARY_INV)  # detect non-white

plt.subplot(122)
plt.imshow(glassMask1, cmap='gray')
plt.title('Sunglass Mask (generated)')



import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

faceImage = cv2.imread("my.jpg")
glassPNG = cv2.imread("sunglass.png", cv2.IMREAD_UNCHANGED)


mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)

rgb_img = cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_img)

h, w, _ = faceImage.shape

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Get left & right eye corner points (approx landmarks)
        left_eye = face_landmarks.landmark[33]   # left eye outer
        right_eye = face_landmarks.landmark[263] # right eye outer

        x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
        x2, y2 = int(right_eye.x * w), int(right_eye.y * h)

        eye_width = x2 - x1
        new_w = int(eye_width * 2.0)   # make glasses wider than eyes
        new_h = int(new_w * glassPNG.shape[0] / glassPNG.shape[1])

        glass_resized = cv2.resize(glassPNG, (new_w, new_h))

        glass_gray = cv2.cvtColor(glass_resized, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(glass_gray, 240, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)

        # Position (centered around eyes)
        x = x1 - int(new_w * 0.25)
        y = y1 - int(new_h * 0.4)

        # ROI on face
        roi = faceImage[y:y+new_h, x:x+new_w]

        # Blend sunglasses with ROI
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(glass_resized, glass_resized, mask=mask)
        combined = cv2.add(bg, fg)

        faceImage[y:y+new_h, x:x+new_w] = combined

plt.figure(figsize=[10,10])
plt.imshow(cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB))
plt.title("Face with Sunglasses (Auto Aligned)")
plt.axis("off")
plt.show()

```
## output:
<img width="413" height="541" alt="image" src="https://github.com/user-attachments/assets/61321ef6-c8d5-498d-8371-7524541d859a" />
<img width="635" height="487" alt="image" src="https://github.com/user-attachments/assets/0fe65321-b191-4135-914e-6edc1e04d56e" />
<img width="623" height="269" alt="image" src="https://github.com/user-attachments/assets/663b699a-e3ac-4227-be46-14b3e124788d" />
<img width="394" height="490" alt="image" src="https://github.com/user-attachments/assets/6b927dbd-90ca-4b3e-b948-27913b9a4534" />



