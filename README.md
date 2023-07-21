## 프로젝트 정보

- 

[](https://git.huconn.com/topst-project/webcam-face-recognition)

이 작업은 파이썬으로 수행되었으며, OpenCV Haar Cascade를 사용했습니다.

VSCode 설치 방법:

1. 다음 명령어를 실행하세요: wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
2. packages.microsoft.gpg 파일을 /etc/apt/trusted.gpg.d/ 디렉토리에 복사합니다: sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
3. 다음 명령어를 실행하여 /etc/apt/sources.list.d/vscode.list 파일을 만듭니다: sudo sh -c 'echo "deb [arch=arm64] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
4. 패키지 목록을 업데이트합니다: sudo apt update
5. VSCode를 설치합니다: sudo apt install code

실행 방법:

다음 명령어를 입력하세요:
code --no-sandbox --user-data-dir=/path/to/alternate/user/data/dir

진행 방법:

- pip을 설치하려면 다음 명령을 실행하십시오.

```
sudo apt install python3-pip

```

- Git을 설치하려면 다음 명령을 사용하십시오.

```
sudo apt install git

```

- 그런 다음 pip를 사용하여 opencv-python을 설치할 수 있습니다.

```
pip install opencv-python

```

git clone https://github.com/opencv/opencv.git

![Alt text](image.png)

이 경로의 **[haarcascad](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml)es폴더를 파이썬 코드가 있는 폴더로 옮겨야 합니다.**

```python
import cv2

def face_eyes(image):
    cascade_face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    cascade_eye_detector = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

    image_resized = cv2.resize(image, (755, 500))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    face_detections = cascade_face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in face_detections:
        cv2.rectangle(image_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image_resized[y:y+h, x:x+w]

        eye_detections = cascade_eye_detector.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=6, minSize=(10, 10), maxSize=(30, 30))
        for (ex, ey, ew, eh) in eye_detections:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

    return image_resized

print("model load")
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
print("camera connect")

while True:
    ret, img = cap.read()
    if not ret:
        break

    result = face_eyes(img)

    cv2.imshow('face_eyes', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

해당 작업은 보드에 웹캠을 연결한 후, 보드에 설치된 vscode를 직접 설치하여 진행합니다.