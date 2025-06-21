# Raspberry Pi Driver Fatigue Detection System

This repository contains a complete implementation of a driver fatigue detection system running on a Raspberry Pi 4. The system uses computer vision and machine learning to detect signs of driver fatigue through eye closure patterns, triggering alerts via a buzzer and LED when fatigue is detected.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
   - [Step 1: Install Raspberry Pi OS](#step-1-install-raspberry-pi-os)
   - [Step 2: Enable Camera and SSH](#step-2-enable-camera-and-ssh)
   - [Step 3: Install Software Dependencies](#step-3-install-software-dependencies)
3. [Hardware Setup](#hardware-setup)
   - [Step 4: Connect Buzzer and LED](#step-4-connect-buzzer-and-led)
4. [Deployment](#deployment)
   - [Step 5: Transfer Your Project](#step-5-transfer-your-project)
   - [Step 6: Create Raspberry Pi Deployment Code](#step-6-create-raspberry-pi-deployment-code)
5. [Testing and Running](#testing-and-running)
   - [Step 7: Test Your System](#step-7-test-your-system)
   - [Step 8: Create Auto-Start Service](#step-8-create-auto-start-service)
6. [Troubleshooting](#troubleshooting)
   - [Camera Issues](#camera-issues)
   - [GPIO Permission Issues](#gpio-permission-issues)
   - [Performance Optimization](#performance-optimization)
   - [Monitor System Performance](#monitor-system-performance)

## Prerequisites
- Raspberry Pi 4
- MicroSD card (16GB or larger)
- Pi Camera Module or USB webcam
- Active buzzer
- LED
- 330Ω resistor
- Jumper wires
- Computer with SD card reader
- Trained TensorFlow Lite model (`fatigue_detector.tflite`)

## Installation

### Step 1: Install Raspberry Pi OS

#### 1.1 Download Raspberry Pi Imager
```bash
# Go to: https://www.raspberrypi.com/software/
# Download Raspberry Pi Imager for your computer OS
```

#### 1.2 Flash the SD Card
- Insert microSD card into your computer
- Open Raspberry Pi Imager
- Click "CHOOSE DEVICE" → Select Raspberry Pi 4
- Click "CHOOSE OS" → Select Raspberry Pi OS (64-bit)
- Click "CHOOSE STORAGE" → Select your microSD card
- Click gear icon for advanced options:
  - Enable SSH: Check this box
  - Set username: pi
  - Set password: Create a strong password
  - Configure WiFi: Enter your network details
  - Set hostname: fatigue-detector
- Click "WRITE" and wait for completion

#### 1.3 First Boot Setup
- Insert SD card into Raspberry Pi
- Connect HDMI cable, keyboard, mouse
- Connect power (Pi will boot automatically)
- Follow the setup wizard:
  - Set country, language, timezone
  - Connect to WiFi if not done earlier
  - Update software when prompted

### Step 2: Enable Camera and SSH

#### 2.1 Enable Camera Interface
```bash
# Open terminal and run:
sudo raspi-config

# Navigate to: Interface Options → Camera → Enable
# Reboot when prompted
sudo reboot
```

#### 2.2 Test Camera
```bash
# For Pi Camera:
libcamera-hello --preview-window 0,0,640,480 --timeout 5000

# For USB webcam:
lsusb  # Check if camera is detected
```

#### 2.3 Update System
```bash
sudo apt update
sudo apt upgrade -y
```

### Step 3: Install Software Dependencies

#### 3.1 Install Python Dependencies
```bash
# Install TensorFlow Lite runtime
sudo apt install python3-tflite-runtime libatlas-base-dev -y

# Install OpenCV and other packages
sudo apt install python3-opencv python3-pip git -y

# Install additional Python packages
pip3 install numpy pandas scikit-learn PyYAML RPi.GPIO picamera
```

#### 3.2 Verify Installation
```bash
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"
python3 -c "from tflite_runtime.interpreter import Interpreter; print('TensorFlow Lite OK')"
python3 -c "import RPi.GPIO as GPIO; print('GPIO library OK')"
```

## Hardware Setup

### Step 4: Connect Buzzer and LED

#### 4.1 Wiring Diagram
```
Raspberry Pi GPIO Layout:
┌─────┬─────┬─────┬─────┐
│ 5V  │     │     │ 5V  │
├─────┼─────┼─────┼─────┤
│ 5V  │ 5V  │ GND │ 14  │
├─────┼─────┼─────┼─────┤
│ 15  │ 18  │ GND │ 23  │  ← Use GPIO 18 (Buzzer) and GPIO 24 (LED)
├─────┼─────┼─────┼─────┤
│ 24  │ 10  │ 9   │ 25  │
└─────┴─────┴─────┴─────┘

Connections:
- Buzzer positive → GPIO 18 (Pin 12)
- Buzzer negative → GND (Pin 14)
- LED long leg → GPIO 24 (Pin 18) 
- LED short leg → 330Ω resistor → GND (Pin 20)
```

#### 4.2 Physical Connections
- Power off Raspberry Pi before wiring
- Connect Buzzer:
  - Red wire (positive) → GPIO 18 (Physical pin 12)
  - Black wire (negative) → GND (Physical pin 14)
- Connect LED:
  - Long leg → GPIO 24 (Physical pin 18)
  - Short leg → 330Ω resistor → GND (Physical pin 20)

#### 4.3 Test Hardware
```python
# Create test file: test_hardware.py
import RPi.GPIO as GPIO
import time

# Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(18, GPIO.OUT)  # Buzzer
GPIO.setup(24, GPIO.OUT)  # LED

try:
    print("Testing hardware...")
    
    # Test LED
    print("LED ON")
    GPIO.output(24, GPIO.HIGH)
    time.sleep(2)
    
    print("LED OFF")
    GPIO.output(24, GPIO.LOW)
    time.sleep(1)
    
    # Test Buzzer
    print("BUZZER ON")
    GPIO.output(18, GPIO.HIGH)
    time.sleep(2)
    
    print("BUZZER OFF")
    GPIO.output(18, GPIO.LOW)
    
    print("Hardware test complete!")
    
except KeyboardInterrupt:
    print("Test interrupted")
    
finally:
    GPIO.cleanup()
```
```bash
# Run the test
python3 test_hardware.py
```

## Deployment

### Step 5: Transfer Your Project

#### 5.1 Copy Project Files
```bash
# Create project directory
mkdir -p ~/fatigue_detection
cd ~/fatigue_detection

# If transferring from your computer via USB:
# Copy your project files to USB drive, then:
cp -r /media/pi/USB_DRIVE/driver_fatigue_detection/* .

# OR transfer via SCP from your computer:
# scp -r driver_fatigue_detection/ pi@fatigue-detector.local:~/
```

#### 5.2 Install Project Dependencies
```bash
cd ~/fatigue_detection
pip3 install -r deployment/requirements_pi.txt
```

### Step 6: Create Raspberry Pi Deployment Code

#### 6.1 Create Pi-Optimized Detection Script
```python
# Create: ~/fatigue_detection/pi_fatigue_detector.py
import cv2
import numpy as np
import tensorflow as tf
import RPi.GPIO as GPIO
import time
import logging
from collections import deque
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class PiFatigueDetector:
    def __init__(self, model_path):
        # Load TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # GPIO setup
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        self.buzzer_pin = 18
        self.led_pin = 24
        GPIO.setup(self.buzzer_pin, GPIO.OUT)
        GPIO.setup(self.led_pin, GPIO.OUT)
        
        # Detection parameters
        self.fatigue_threshold = 15  # consecutive closed eye frames
        self.confidence_threshold = 0.7
        self.closed_eye_counter = 0
        self.last_alert_time = 0
        self.alert_cooldown = 5  # seconds between alerts
        
        # Performance tracking
        self.fps_queue = deque(maxlen=30)
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        logger.info("Pi Fatigue Detector initialized")
    
    def preprocess_frame(self, frame):
        """Extract and preprocess eye region"""
        # Resize for faster processing
        small_frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, None
        
        # Get largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Scale back to original frame
        scale_x, scale_y = frame.shape[1] / 320, frame.shape[0] / 240
        x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
        
        # Extract eye region
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) == 0:
            return None, (x, y, w, h)
        
        # Get first eye
        ex, ey, ew, eh = eyes[0]
        eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
        
        if eye_region.size > 0:
            # Preprocess for model
            eye_resized = cv2.resize(eye_region, (64, 64))
            eye_normalized = eye_resized.astype(np.float32) / 255.0
            eye_input = np.expand_dims(np.expand_dims(eye_normalized, axis=0), axis=-1)
            
            eye_coords = (x + ex, y + ey, ew, eh)
            return eye_input, (x, y, w, h), eye_coords
        
        return None, (x, y, w, h), None
    
    def predict_fatigue(self, preprocessed_eye):
        """Run inference"""
        if preprocessed_eye is None:
            return [0.5, 0.5]
        
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_eye)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output_data[0]
    
    def trigger_alert(self, alert_type="fatigue"):
        """Trigger hardware alerts"""
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        self.last_alert_time = current_time
        
        if alert_type == "fatigue":
            logger.warning("FATIGUE ALERT!")
            # Rapid buzzer and LED pattern
            for _ in range(3):
                GPIO.output(self.buzzer_pin, GPIO.HIGH)
                GPIO.output(self.led_pin, GPIO.HIGH)
                time.sleep(0.3)
                GPIO.output(self.buzzer_pin, GPIO.LOW)
                GPIO.output(self.led_pin, GPIO.LOW)
                time.sleep(0.3)
        else:
            # Drowsy warning (LED only)
            GPIO.output(self.led_pin, GPIO.HIGH)
            time.sleep(1)
            GPIO.output(self.led_pin, GPIO.LOW)
    
    def update_fatigue_state(self, prediction):
        """Update fatigue tracking"""
        closed_prob = prediction[0]
        is_drowsy = closed_prob > self.confidence_threshold
        
        if is_drowsy:
            self.closed_eye_counter += 1
            if self.closed_eye_counter == 5:  # Early warning
                self.trigger_alert("drowsy")
        else:
            self.closed_eye_counter = max(0, self.closed_eye_counter - 1)
        
        is_fatigued = self.closed_eye_counter >= self.fatigue_threshold
        
        if is_fatigued:
            self.trigger_alert("fatigue")
        
        return {
            'is_drowsy': is_drowsy,
            'is_fatigued': is_fatigued,
            'closed_probability': closed_prob,
            'counter': self.closed_eye_counter
        }
    
    def draw_status(self, frame, fatigue_state, face_coords=None, eye_coords=None, fps=None):
        """Draw status on frame"""
        height, width = frame.shape[:2]
        
        # Draw rectangles
        if face_coords:
            x, y, w, h = face_coords
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        if eye_coords:
            ex, ey, ew, eh = eye_coords
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Status text
        if fatigue_state['is_fatigued']:
            status_text = "FATIGUE!"
            status_color = (0, 0, 255)
        elif fatigue_state['is_drowsy']:
            status_text = "DROWSY"
            status_color = (0, 165, 255)
        else:
            status_text = "ALERT"
            status_color = (0, 255, 0)
        
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Info text
        if fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, height-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Counter: {fatigue_state['counter']}", (10, height-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_detection(self, camera_index=0, display=True):
        """Main detection loop"""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Starting fatigue detection...")
        logger.info("Press 'q' to quit")
        
        frame_count = 0
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                frame_count += 1
                
                # Process every 2nd frame for performance
                if frame_count % 2 == 0:
                    result = self.preprocess_frame(frame)
                    if len(result) == 3:
                        preprocessed_eye, face_coords, eye_coords = result
                    else:
                        preprocessed_eye, face_coords = result
                        eye_coords = None
                    
                    prediction = self.predict_fatigue(preprocessed_eye)
                    fatigue_state = self.update_fatigue_state(prediction)
                else:
                    face_coords = eye_coords = None
                    fatigue_state = {
                        'is_drowsy': False,
                        'is_fatigued': False,
                        'counter': self.closed_eye_counter
                    }
                
                # Calculate FPS
                frame_time = time.time() - start_time
                if frame_time > 0:
                    self.fps_queue.append(1.0 / frame_time)
                
                if display:
                    avg_fps = np.mean(self.fps_queue) if self.fps_queue else 0
                    self.draw_status(frame, fatigue_state, face_coords, eye_coords, avg_fps)
                    cv2.imshow('Driver Fatigue Detection', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except KeyboardInterrupt:
            logger.info("Detection interrupted by user")
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            GPIO.cleanup()
            logger.info("Detection stopped and cleanup completed")

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Pi Fatigue Detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to TensorFlow Lite model')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without display')
    
    args = parser.parse_args()
    
    detector = PiFatigueDetector(args.model)
    detector.run_detection(args.camera, display=not args.no_display)
```

## Testing and Running

### Step 7: Test Your System

#### 7.1 Copy Your Trained Model
```bash
# Copy your .tflite model to Pi
# Assuming you have fatigue_detector.tflite from training
cp fatigue_detector.tflite ~/fatigue_detection/
```

#### 7.2 Run Detection
```bash
cd ~/fatigue_detection

# Run with display
python3 pi_fatigue_detector.py --model fatigue_detector.tflite

# Run without display (headless)
python3 pi_fatigue_detector.py --model fatigue_detector.tflite --no-display
```

### Step 8: Create Auto-Start Service

#### 8.1 Create Service File
```bash
sudo nano /etc/systemd/system/fatigue-detection.service
```

#### 8.2 Add Service Content
```
[Unit]
Description=Driver Fatigue Detection Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/fatigue_detection
ExecStart=/usr/bin/python3 pi_fatigue_detector.py --model fatigue_detector.tflite --no-display
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

#### 8.3 Enable Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable fatigue-detection.service
sudo systemctl start fatigue-detection.service

# Check status
sudo systemctl status fatigue-detection.service
```

## Troubleshooting

### Camera Issues
```bash
# Check camera connection
vcgencmd get_camera

# Test camera
libcamera-hello --timeout 5000
```

### GPIO Permission Issues
```bash
# Add user to gpio group
sudo usermod -a -G gpio pi
```

### Performance Optimization
```bash
# Increase GPU memory split
sudo raspi-config
# Advanced Options → Memory Split → 128
```

### Monitor System Performance
```bash
# Check CPU usage
htop

# Check temperature
vcgencmd measure_temp

# Monitor GPU memory
vcgencmd get_mem gpu
```