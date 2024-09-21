import streamlit as st
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import tempfile


###--Load Model
@st.cache_resource
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device is :{device}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        return model, processor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None
    
def generate_answer(video_path, question, model, processor):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},  
                    {"type": "text", "text": question},  
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            videos=video_inputs, 
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        # Generate output from the model
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]
    except Exception as e:
        st.error(f"Error during inference: {e}")
        return None
    
def main():
    st.title("🎬 Video Question-Answering with Qwen2-VL 🤖")
    st.write("Upload a video and ask a question about its content using Qwen2-VL")
    st.sidebar.header("App Settings")
    st.sidebar.write("Customize your experience")
    dtype_option = st.sidebar.selectbox("Torch Data Type", ["auto", "float16", "bfloat16"])
    progress_bar=st.sidebar.progress(0)
    with st.spinner("Loading model... Please wait"):
        model,processor=load_model()
    if model is None or processor is None:
        return
    st.subheader("Step 1: Upload your video 🎥")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    print(f"Uploaded File is  :{uploaded_file}")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_video_path = temp_file.name
            print(f"Video Path is :{temp_video_path}")
        st.video(temp_video_path)
        st.info("Tip: Watch the video before asking the question. 📹")
        st.subheader("Step 2: Ask a question about the video 📝")
        user_question = st.text_input("Enter your question here")
        if st.button("Get Answer"):
            if user_question:
                st.subheader("Step 3: Processing your video and generating an answer... ⏳")
                with st.spinner("Generating answer... This may take a few moments."):
                    progress_bar.progress(50)
                    answer = generate_answer(temp_video_path, user_question, model, processor)
                    if answer:
                        st.success(f"Answer: {answer}")
                    else:
                        st.error("Could not generate an answer.")
                        
                    progress_bar.progress(100)
            else:
                st.error("Please ask a question.")
    else:
        st.warning("Please upload a video to proceed.")
    st.sidebar.markdown("---")
    st.sidebar.info("For any issues or feedback, contact us at aman.singh@accessassist.in")


if __name__ == "__main__":
    main()
