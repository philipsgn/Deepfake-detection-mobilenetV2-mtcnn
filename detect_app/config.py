from pathlib import Path

IMG_SIZE = (160, 160)  # Kích thước input cho MobileNetV2


CONF_THRESH = 0.7          
MIN_FACE_SIZE = 60         
MAX_FACE_SIZE = 300        
MARGIN = 0.25             
FACE_CONF_THRESH = 0.85    


FRAMES_PER_VIDEO = 32      


PRED_THRESHOLD = 0.7        
FAKE_RATIO_THRESHOLD = 0.45 

# Frame-level threshold
FRAME_FAKE_THRESH = 0.65    

# Video-level threshold
VIDEO_FAKE_RATIO = 0.35     
MIN_FRAMES_VALID = 1       


TEMP_DIR = Path("temp")
FACE_DIR = TEMP_DIR / "faces"

# Đường dẫn đến model đã train
MODEL_PATH = r"C:\Users\TanPhat\Documents\DEEPLEARNING\deepfake_detection\model_mobileNetV2\train_model\finetune_MobileNetV2_new.keras"

# Kiểm tra model có tồn tại không
if not Path(MODEL_PATH).exists():
    print(f"WARNING: Model not found at {MODEL_PATH}")
    print("Please update MODEL_PATH in config.py")