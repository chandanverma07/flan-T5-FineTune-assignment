import streamlit as st
import flan_tf_model

# Initialize the LanguageModelHandler
model_repo_id = st.sidebar.text_input("Model Repository ID", "google/flan-t5-small")
#handler = flan_t5_hf.LanguageModelHandler(model_repo_id)
handler = flan_tf_model.LanguageModelHandler(model_repo_id="google/flan-t5-small")
st.title("Verify Flan T5 Task")

# Select task
task = st.selectbox("Select a Task", ["summarization", "q&a", "translation"])

# Input fields based on the task
input_text = st.text_area("Input Text", "Enter your text here")

context = ""
if task == "q&a":
    context = st.text_area("Context", "Enter context here")

# Button to execute task
if st.button("Execute"):
    try:
        result = handler.execute_task(task, input_text, context)
        if result:
            st.success("Result:")
            st.write(result)
        else:
            st.error("No result returned.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# UI for Model Details
st.title("Model Details")

# Button to Show Model Details
if st.button("Show Model Details"):
    try:
        model = flan_tf_model.load_model_hf_tf("transformers")
        model_details_df, total_params = flan_tf_model.get_model_details(model)  # Assuming handler.llm is the model
        st.write("Model Parameters:")
        st.dataframe(model_details_df)
        st.write(f"Total Parameters: {total_params}")
    except Exception as e:
        st.error(f"An error occurred while fetching model details: {str(e)}")
