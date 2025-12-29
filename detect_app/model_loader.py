import tensorflow as tf
from tensorflow.keras.models import load_model
from config import MODEL_PATH
from pathlib import Path

def load_trained_model():
    """
    Load pre-trained deepfake detection model
    
    Returns:
        Keras model đã được train
    
    Raises:
        FileNotFoundError: Nếu model file không tồn tại
        Exception: Nếu có lỗi khi load model
    """
    model_path = Path(MODEL_PATH)
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            "Please check MODEL_PATH in config.py"
        )
    
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
        
        # In thông tin model
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        return model
        
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")