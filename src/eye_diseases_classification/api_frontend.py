#run with uv "run streamlit run src/eye_diseases_classification/frontend.py"
import pandas as pd
import requests
import streamlit as st

@st.cache_resource  
def get_backend_url():
    """Get the URL of the backend service."""
    return "https://eye-api-967791335191.europe-north1.run.app/"


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}classify"
    response = requests.post(predict_url, files=image, timeout=10)
    print(image)
    print(predict_url)
    print(response.json())
    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Eye Disease Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        # Streamlit UploadedFile has .name and .type
        filename = getattr(uploaded_file, "name", "image.jpg")
        content_type = getattr(uploaded_file, "type", None) or "image/jpeg"

        # use tuple (filename, bytes, content_type)
        files = {
            "file": (filename, uploaded_file.getvalue(), content_type)
        }
        result = classify_image(files, backend=backend)

        if result is not None:
            prediction = result["pred_index"]
            probabilities = result["probabilities"].values()

            st.image(image)
            classes = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]
            st.write("Prediction:", classes[prediction])
            data = {"Class": [f"{clas}" for clas in classes], "Probability": probabilities}
            df = pd.DataFrame(data)
            df.set_index("Class", inplace=True)
            st.bar_chart(df, y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()