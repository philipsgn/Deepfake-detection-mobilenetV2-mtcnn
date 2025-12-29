import streamlit as st
from pathlib import Path
from video_utils import extract_faces_from_video
from model_loader import load_trained_model
from predictor import predict_faces, aggregate_prediction
from config import TEMP_DIR
import shutil

st.set_page_config(page_title="Deepfake Detection", layout="wide")
st.title(" Deepfake Video Detection")

# Thêm cache cho model để tránh load lại nhiều lần
@st.cache_resource
def get_model():
    return load_trained_model()

uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

if uploaded:
    try:
        # Tạo thư mục temp
        TEMP_DIR.mkdir(exist_ok=True)
        video_path = TEMP_DIR / "uploaded.mp4"

        # Lưu video
        with open(video_path, "wb") as f:
            f.write(uploaded.read())

        st.video(str(video_path))

        if st.button("🔍 Analyze Video"):
            with st.spinner("Extracting faces from video..."):
                face_paths = extract_faces_from_video(video_path)

            if len(face_paths) == 0:
                st.error(" No face detected in video. Please upload a video with visible faces.")
            else:
                st.success(f" Extracted {len(face_paths)} faces from video")
                
                # Load model
                model = get_model()

                with st.spinner("Running deepfake detection..."):
                    preds = predict_faces(model, face_paths)
                    
                    if len(preds) == 0:
                        st.error(" Could not process any faces. Please try another video.")
                    else:
                        label, metrics = aggregate_prediction(preds)

                        # Hiển thị kết quả
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if label == "FAKE":
                                st.error(f"##  Result: **{label}**")
                            elif label == "REAL":
                                st.success(f"##  Result: **{label}**")
                            else:
                                st.warning(f"##  Result: **{label}**")
                                st.info(f" Only {metrics['num_frames']} frame(s) analyzed. More frames recommended for accurate results.")
                        
                        with col2:
                            st.metric("Mean Fake Probability", f"{metrics['mean_score']:.1%}")
                            st.metric("Fake Frames Ratio", f"{metrics['fake_ratio']:.1%}")
                            st.metric("Frames Analyzed", metrics['num_frames'])

                        # Hiển thị chi tiết predictions
                        with st.expander(" View Detailed Predictions"):
                            st.write(f"**Individual frame predictions:**")
                            for i, pred in enumerate(preds[:10]):  # Hiển thị 10 đầu tiên
                                st.write(f"Frame {i+1}: {pred:.1%} fake probability")
                            if len(preds) > 10:
                                st.write(f"... and {len(preds)-10} more frames")

                        # Hiển thị faces đã extract
                        st.subheader(" Extracted Faces")
                        st.image([str(p) for p in face_paths[:12]], width=120, caption=[f"Face {i+1}" for i in range(min(12, len(face_paths)))])
                        
                        if len(face_paths) > 12:
                            st.info(f"Showing 12 of {len(face_paths)} extracted faces")

    except Exception as e:
        st.error(f" An error occurred: {str(e)}")
        st.exception(e)
    
    finally:
        # Cleanup (optional)
        if st.button(" Clear Temporary Files"):
            if TEMP_DIR.exists():
                shutil.rmtree(TEMP_DIR)
                st.success("Temporary files cleared!")
                st.rerun()

else:
    st.info("👆 Please upload a video file to begin analysis")
    st.markdown("""
    ### How it works:
    1. Upload a video file (MP4, AVI, or MOV)
    2. Click 'Analyze Video' button
    3. The system will extract faces and analyze them
    4. Get the final verdict: REAL or FAKE
    
    ### Tips:
    - Videos with clear, visible faces work best
    - Good lighting improves detection accuracy
    - Videos should be at least a few seconds long
    """)