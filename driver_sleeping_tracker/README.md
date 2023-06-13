# Driver Drowsiness Detection Project
This project employs a driver drowsiness detection system using image processing and machine learning techniques. The fundamental operation of the system involves processing the driver's live video feed, detecting the presence of the driver's eyes, calculating the Eye Aspect Ratio (EAR), and ascertaining whether the eyes are closed. In case the system detects that the driver's eyes are closed, it triggers a vehicle slowdown simulation, thereby providing an application that has real-world potential to enhance road safety.

Dlib, an open-source toolkit for machine learning, plays an integral role in this program. It helps in the accurate detection of facial landmarks, which are critical in computing the EAR. EAR is a ratio of distances computed using these facial landmarks, specifically those around the eyes. An abrupt drop in EAR is usually a good indicator of a blink, and in our case, if the eyes remain closed for an extended period, drowsiness.

## Dependencies
The program requires the following Python libraries:

- OpenCV
- Dlib
- Numpy
- Scipy
- Matplotlib
- Seaborn
Additionally, it requires the 'shape_predictor_68_face_landmarks_GTX.dat' pretrained model from dlib, which is used to detect the facial landmarks. This file should be placed in the same directory as the script.

## Installation
You can install all necessary dependencies using pip and the provided requirements.txt file:

```
pip install -r requirements.txt
```

Please note that the exact library versions might depend on your specific Python and operating system versions. The above versions work well together in an environment with Python 3.8.

## Usage
To use the program, simply run it with Python:
```
python detect_drowsiness.py
```
The program activates the system's default camera and begins processing the frames. It calculates the driver's Eye Aspect Ratio (EAR) and checks if it falls below a predetermined threshold for a specific number of consecutive frames, which suggests the driver's eyes are closed. In this situation, it initiates the vehicle slowdown simulation.

A real-time plot of the simulated car speed is displayed. If the system determines that the driver's eyes are closed, the plot will illustrate the car's speed plummeting to zero, symbolizing the vehicle's deceleration.

To terminate the program, simply press 'q' and 1. If doesn't work, press Ctrl C in your Terminal.

![Captura de tela 2023-06-09 194011](https://github.com/pedrocrp/data_projects/assets/83802848/b25d64f9-ddf9-42a7-affa-3f606fb55f90)

![Captura de tela 2023-06-10 011102](https://github.com/pedrocrp/data_projects/assets/83802848/6f0a4bc2-bc92-482e-99ff-29e647776f7c)

