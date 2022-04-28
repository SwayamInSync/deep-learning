# OpenCV

## <u>Installing</u>

```bash
pip install opencv-contrib-python
```

## <u>Reading Image and Videos</u>

1. **Images**

```python
import cv2 as cv

img = cv.imread('Resources/Photos/cat.jpg')
cv.imshow('Cat', img)
cv.waitKey(0) # it waits for sometime to press the key, passing 0 will make it wait infite
# note: cv.waitKey() returns the ASCII value of pressed key and if you clicked on cross to close window then it'll return -1
```

2. **Videos**

```python
import cv2 as cv

capture = cv.VideoCapture('/Resources/Videos/dog.mp4')
# cv.VideoCapture() either takes number as 0, 1, 2, etc for referencing to the connected camera to system like webcam
# as 0, 1 reference to 1st camera connected to system. It also takes path of the video

while True:
    isTrue, frame = capture.read() # reads the video frame by frame, returns bool whether the frame read successfully
    # or not and that frame
    cv.imshow('Video', frame)

    # as the key 'd' is pressed, ASCII value of 'd' is returned by cv.waitKey() then it gets AND operation by 11111111
    # which do nothing to ASCII value since, value & 1 = value itself
    # just a simple check to make sure, whether 'd' is pressed or not
    if cv.waitKey(20) & 0xff == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
```

### <u>**Rescaling Images and Videos:**</u>

```python
import cv2 as cv


def rescale(frame, scale=0.75):
"""works with Images, Video, Live Video"""
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dimension = (width, height)  # height will be first here
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)


capture = cv.VideoCapture("./Resources/Videos/dog.mp4")
while True:
    isTrue, frame = capture.read()
    if isTrue:
        cv.imshow('Video', rescale(frame))
    else:
        break

    if cv.waitKey(20) == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
```

### <u>Drawing shapes and writing text on image:</u>

```python
import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8')
# blank[:] = 0,255,0
# cv.imshow('Green', blank)

cv.rectangle(blank, (0,0), (blank.shape[0]//2, blank.shape[1]//2), color=(0,255,0), thickness=cv.FILLED) # square too
# color = (B, G, R)
cv.circle(blank, center=(blank.shape[0]//2, blank.shape[1]//2), radius=40, thickness=2, color=(0,0,255))
cv.line(blank, (0,0), (blank.shape[0]//2, blank.shape[1]//2), color=(255,255,255), thickness=3)

cv.putText(blank, "Hello", org=(225,225), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1.0, color=(255,182,193), thickness=2)
cv.imshow('rectangle', blank)

# img = cv.imread('./Resources/Photos/cat.jpg')
# cv.imshow('Cat', img)
cv.waitKey(0)
```

### <u>**Essential functions in OpenCV**</u>

- converting color (RGB -> GRAY)

  ```python
  import cv2 as cv
  import numpy as np
  
  img = cv.imread('./Resources/Photos/park.jpg')
  cv.imshow('color', img)
  gray = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)
  cv.imshow('Gray', gray)
  
  cv.waitKey(0)
  ```

- Blur

  ```python
  import cv2 as cv
  
  img = cv.imread('./Resources/Photos/park.jpg')
  cv.imshow('orig', img)
  blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
  cv.imshow('Blur', blur)
  ```

- Edge cascade

  ```python
  import cv2 as cv
  
  img = cv.imread('./Resources/Photos/park.jpg')
  cv.imshow('orig', img)
  blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
  
  canny = cv.Canny(blur, 125, 175) # applying on blurred to reduce the edges
  cv.imshow('Canny Edges', canny)
  ```

- Resize and crop

  ```python
  import cv2 as cv
  
  img = cv.imread('./Resources/Photos/park.jpg')
  cv.imshow('orig', img)
  
  # Resize
  resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
  cv.imshow('Resized', resized)
  
  # Cropping
  cropped = img[50:200, 200:400]
  cv.imshow('Cropped', cropped)
  ```

- Dilate

  ```python
  import cv2 as cv
  import numpy as np
  
  img = cv.imread('./Resources/Photos/park.jpg')
  cv.imshow('color', img)
  
  canny = cv.Canny(img, threshold1=125, threshold2=175)
  cv.imshow('Edge', canny)
  
  dilate = cv.dilate(canny, (7,7), iterations=4)# thick the edges, use blurred for less edges
  cv.imshow('dilate', dilate)
  cv.waitKey(0)
  ```

### <u>**Image Transformation**</u>

```python
import cv2 as cv
import numpy as np

img = cv.imread('./Resources/Photos/lady.jpg')


# translation

def translate(img, x, y):
    trans_m = np.float32(
        [[1, 0, x],  # this is the type of translation matirx
         [0, 1, y]]
    )
    dimensions = (img.shape[1], img.shape[0])
    img = cv.warpAffine(img, trans_m, dimensions)
    return img


# -x --> left
# -y --> up
#  x --> right
#  y --> down

translated = translate(img, x=100, y=100)
cv.imshow('translated', translated)


# rotation

def rotate(img, angle, rotPoint=None):
    height, width = img.shape[:2]
    if rotPoint is None:
        rotPoint = (width // 2, height // 2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, scale=1.0)
    rotated = cv.warpAffine(img, rotMat, (width, height))
    return rotated


rotated = rotate(img, 90)
cv.imshow('rotated', rotated)

# resizing

resize = cv.resize(img, (10000, 10000), interpolation=cv.INTER_CUBIC)
cv.imshow('resize', resize)

# flipping
flip = cv.flip(img, flipCode=-1)
"""
fliCode = 0 --> flipping vertically
fliCode = 1 --> flipping horizontally
fliCode = -1 --> flipping both vertically & horizontally 
"""
cv.imshow('flipped', flip)

# cropping

crp = img[200:400, 300:400]
cv.imshow('crop', crp)
cv.waitKey(0)

```

### <u>**Color Spaces**</u>

```python
import cv2 as cv
from cv2 import imshow, cvtColor

img = cv.imread('./Resources/Photos/lady.jpg')
imshow('img', img)

gray = cvtColor(img, cv.COLOR_BGR2GRAY)
imshow('gray', gray)

hsv = cvtColor(img, cv.COLOR_BGR2HSV)
imshow('hsv', hsv)

lab = cvtColor(img, cv.COLOR_BGR2LAB)
imshow('lab', lab)

rgb = cvtColor(img, cv.COLOR_BGR2RGB)
imshow('rgb', rgb)

# you can also go backwards as hsv->BGR by cv.COLOR_HSV2BGR

cv.waitKey(0)
```

### <u>**Color Channel**</u>

```python
import cv2 as cv
from cv2 import imshow
import numpy as np


img = cv.imread('./Resources/Photos/lady.jpg')
imshow('img', img)

# splitting image into it's respective channels
b, g, r = cv.split(img)
imshow('blue', b)
imshow('green', g)
imshow('red', r)
# image is shown in gray format, where hight light represents the color of that channel more compare to others


# merging the channels to form images
merged = cv.merge([b, g, r])
imshow('merged', merged)


# you can also visualize the color channel as their own color instead of gray
blank = np.zeros(img.shape[:2], dtype='uint8')
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

imshow('Blue', blue)
imshow('Green', green)
imshow('Red', red)
# such representation is not good for understanding the color channel distribution

cv.waitKey(0)
```

### 

### <u>**Blurring**</u>

```python
import cv2 as cv
from cv2 import imshow, imread
import numpy as np

img = imread('./Resources/Photos/cats.jpg')
imshow('cats', img)

# average blur
average = cv.blur(img, (7,7))
imshow('average', average)

# gaussian blur, less densed than average blur
gauss = cv.GaussianBlur(img, (7,7), sigmaX=0)
imshow('gaussian', gauss)

# median blur, used for removing noise (kind smugged)
median = cv.medianBlur(img, 7)
imshow('median', median)

# bi-lateral blur, most effective and used in advanced CV projects.
# applies blur but retains the edges of image
bLateral = cv.bilateralFilter(img, d=10, sigmaColor=35, sigmaSpace=25)
imshow('BiLateral', bLateral)

cv.waitKey(0)
```

### <u>**Bitwise Operations**</u>

```python
import cv2 as cv
from cv2 import imshow, imread
import numpy as np

blank = np.zeros((400,400), dtype='uint8')

rectangle = cv.rectangle(blank.copy(), (30,30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)

imshow('rectangle', rectangle)
imshow('circle', circle)

# AND : intersection
bitwise_and = cv.bitwise_and(rectangle, circle)
imshow('AND', bitwise_and)

# OR : union
bitwise_or = cv.bitwise_or(rectangle, circle)
imshow('OR', bitwise_or)

# XOR : non-intersecting region
bitwise_xor = cv.bitwise_xor(rectangle, circle)
imshow('XOR', bitwise_xor)

# NOT
bitwise_not = cv.bitwise_not(rectangle)
imshow('NOT', bitwise_not)


cv.waitKey(0)
```

### <u>**Masking**</u>

```python
import cv2 as cv
from cv2 import imshow, imread
import numpy as np

img = imread('./Resources/Photos/cats.jpg')
blank = np.zeros(img.shape[:2], dtype='uint8')

circle = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)

# imshow('rectangle', rectangle)
imshow('circle', circle)
# AND : intersection
bitwise_and = cv.bitwise_and(img, img, mask=circle)
# size of mask need to be same as image
imshow('AND', bitwise_and)
cv.waitKey(0)
```

### **<u>Histogram Computation</u>**

