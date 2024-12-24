import streamlit as st
from combiner import tester
import pandas as pd
import os

#st.markdown("*Pateint Zone*")

st.sidebar.page_link('app.py', label='Home')
st.sidebar.page_link('pages/Patient.py', label='Patient')

# Create a folder named 'patient_data' if it doesn't exist
# Create a folder named 'patient_data' if it doesn't exist
output_folder = "patient_data"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

st.title("Data and Visualization")

# Container 1: CSV file upload
with st.container():
    st.subheader("Upload CSV Files")
    uploaded_files = st.file_uploader(
        "Choose multiple CSV files",
        type="csv",
        accept_multiple_files=True,
        label_visibility='collapsed'
    )

    # Save uploaded files into the 'patient_data' folder
    if uploaded_files:
        for file in uploaded_files:
            filename = os.path.basename(file.name)
            file_path = os.path.join(output_folder, filename)
            
            # Save the file content
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        
        st.success(f"Uploaded and saved {len(uploaded_files)} files successfully!")

# Container 2: Visualization
with st.container():
    st.subheader("Visualizations from Uploaded Files")
    if uploaded_files:
        for file in uploaded_files:
            filename = os.path.basename(file.name)
            file_path = os.path.join(output_folder, filename)

            st.markdown(f"### {filename}")
            
            try:
                # Call the tester function to get the plot
                if st.session_state.file_labels:
                    fig = tester(st.session_state.file_labels, file_path)  # Pass file_path directly
                else:
                    st.warning("Please train the model first")
                
                # Display the figure
                st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Error processing {filename}: {e}")