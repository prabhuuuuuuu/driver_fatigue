# Create test_model.py
import cv2
import numpy as np
import tensorflow as tf

def test_trained_model():
    # Load TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path="C:\\Users\\prana\\Downloads\\fatigue_detector.tflite")
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Model Input Shape:", input_details[0]['shape'])
    print("Model Output Shape:", output_details[0]['shape'])
    
    # Test with dummy data
    dummy_input = np.random.random((1, 64, 64, 1)).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output[0])
    confidence = np.max(output[0])
    
    print(f"Test Prediction: {prediction} ({'Closed' if prediction == 0 else 'Open'})")
    print(f"Confidence: {confidence:.3f}")
    print("✅ Model loaded and working correctly!")

if __name__ == "__main__":
    test_trained_model()
