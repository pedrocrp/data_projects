import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import random
import seaborn as sns

# Class to calculate Eye Aspect Ratios (EAR) for drowsiness detection
class EyeAspectRatios:
    def __init__(self):
        pass

    # Function to convert a facial shape (i.e., 68 facial landmarks) to a numpy array
    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    # Function to calculate the eye aspect ratio (EAR)
    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

# Class to simulate the speed of a vehicle
class SpeedSimulation:
    def __init__(self):
        # Initial speed, time, and target speed
        self.speeds = [50]
        self.time = [0]
        self.target_speed = 50  

    # Function to simulate the speed of a vehicle
    def simulate_speed(self, stop=False):
        # Check if the vehicle should stop
        if stop:
            self.target_speed = 0
        else:
            # Add some random variation to the target speed
            self.target_speed = 50 + random.randint(-10, 50)  

        # Gradually adjust the speed towards the target speed
        if self.speeds[-1] < self.target_speed:
            new_speed = min(self.speeds[-1] + 1, self.target_speed)
        elif self.speeds[-1] > self.target_speed:
            new_speed = max(self.speeds[-1] - 1, self.target_speed)
        else:
            new_speed = self.target_speed

        # Append new speed and time
        self.speeds.append(new_speed)
        self.time.append(self.time[-1] + 0.5)  
        return new_speed

# Class to detect the face and calculate EAR
class FaceDetector:
    def __init__(self):
        # EAR threshold and the number of frames the eyes must be closed for before we start slowing down
        self.EAR_THRESHOLD = 0.25
        self.EYES_CLOSED_FRAMES = 20 
        # Initial closed frames
        self.closed_frames = 0  
        # Dlib's face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(r"shape_predictor_68_face_landmarks_GTX.dat")
        self.ear_calculator = EyeAspectRatios()
        self.speed_simulator = SpeedSimulation()

    # Function to process a frame
    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        eyes_closed = False

        for face in faces:
            shape = self.predictor(gray, face)
            shape = self.ear_calculator.shape_to_np(shape)

            leftEye = shape[42:48]
            rightEye = shape[36:42]
            leftEAR = self.ear_calculator.eye_aspect_ratio(leftEye)
            rightEAR = self.ear_calculator.eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0
            color = (0, 255, 0) if ear >= self.EAR_THRESHOLD else (0, 0, 255)

            # Check if the eyes are closed
            if ear < self.EAR_THRESHOLD:
                self.closed_frames += 1
            else:
                self.closed_frames = 0

            # Draw the eyes and facial landmarks
            cv2.drawContours(frame, [leftEye], -1, color, 1)
            cv2.drawContours(frame, [rightEye], -1, color, 1)

            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, color, -1)

            # Check if the eyes have been closed for a certain number of frames
            if self.closed_frames >= self.EYES_CLOSED_FRAMES:
                eyes_closed = True

        # Check if the eyes are closed
        if eyes_closed:
            self.speed_simulator.simulate_speed(stop=True)
            cv2.putText(frame, "Turning autopilot on and stopping the vehicle", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (0,0,255), 2)
        else:
            self.speed_simulator.simulate_speed(stop=False)

        # Resize the frame
        scale_percent = 150 
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        return resized_frame

# Main
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    face_detector = FaceDetector()

    plt.ion()
    fig = plt.figure()

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = face_detector.process_frame(frame)

        # Plotting the speed simulation
        plt.clf()
        sns.set(style="darkgrid")
        plt.title('Car Speed Simulation Over Time')
        plt.grid(True)
        plt.ylabel('Speed (km/h)')
        plt.xticks([]) 
        plt.ylim(-10, 150)  
        sns.lineplot(x=face_detector.speed_simulator.time, y=face_detector.speed_simulator.speeds, color='r')

        plt.pause(0.01)

        cv2.imshow("Driver Face", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            plt.close()
            break
