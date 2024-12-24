import streamlit as st
from widgets import __login__
import os
import matplotlib.pyplot as plt  # Assuming `combine` returns a matplotlib figure
from combiner import combiner_tester, new_tester
import pandas as pd
import load_dotenv

COURIER_API_KEY = os.getenv("COURIER_API_KEY")

__login__obj = __login__(auth_token = COURIER_API_KEY, 
                    company_name = "Shims",
                    width = 200, height = 250, 
                    logout_button_name = 'Logout', hide_menu_bool = False, 
                    hide_footer_bool = False, 
                    lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

LOGGED_IN = __login__obj.build_login_ui()

UPLOAD_FOLDER = "user_data"

if LOGGED_IN == True:
    st.sidebar.page_link('app.py', label='Home')
    st.sidebar.page_link('pages/Patient.py', label='Patient')
    
    st.title("Concentration Classifier App")
    st.write("Upload your CSV files from the sidebar.")

    # Sidebar uploader
    st.sidebar.header("Upload your CSV files")
    uploaded_files = st.sidebar.file_uploader(
        "Choose CSV files", 
        type="csv", 
        accept_multiple_files=True
    )

    # Initialize session state for file paths, labels, and analyze state
    if "file_paths" not in st.session_state:
        st.session_state.file_paths = []
    if "file_labels" not in st.session_state:
        st.session_state.file_labels = {}
    if "show_labels" not in st.session_state:
        st.session_state.show_labels = False

    # Check for changes in the uploaded files
    current_file_names = [file.name for file in uploaded_files] if uploaded_files else []

    # Update session state based on the current files in the uploader
    new_file_paths = []
    new_file_labels = {}

    # Check if files were removed from the uploader
    for file_path in st.session_state.file_paths:
        file_name = os.path.basename(file_path)
        if file_name in current_file_names:
            new_file_paths.append(file_path)
            new_file_labels[file_path] = st.session_state.file_labels.get(file_path, "")

    # Update the session state with remaining files and labels
    st.session_state.file_paths = new_file_paths
    st.session_state.file_labels = new_file_labels

    # Process new uploaded files
    if uploaded_files:
        st.write("Files Uploaded")

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_path = os.path.join(UPLOAD_FOLDER, file_name)

            # Ensure the 'user_data' directory exists
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)

            # Normalize file path
            file_path = os.path.normpath(file_path)

            # Save file to the 'user_data' folder
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Add new files to session state if not already present
            if file_path not in st.session_state.file_paths:
                st.session_state.file_paths.append(file_path)
                st.session_state.file_labels[file_path] = ""  # Initialize with an empty label

        # Display each uploaded file as a button and show a plot on click
        for file_path in st.session_state.file_paths:
            file_name = os.path.basename(file_path)
            
            # Use expander for popover-like behavior
            with st.expander(f"📊 View {file_name}", expanded=False):
                # Load the CSV data and generate the plot
                try:
                    # Use error_bad_lines=False to skip bad lines if necessary
                    data = pd.read_csv(file_path, encoding='utf-8')

                    # Check if required columns exist
                    if "Drain Time" in data.columns and "Drain Current" in data.columns:
                        # Create the plot
                        fig, ax = plt.subplots()
                        ax.plot(data["Drain Time"], data["Drain Current"], marker='o')
                        ax.set_xlabel("Drain Time")
                        ax.set_ylabel("Drain Current")
                        ax.set_title(f"Graph for {file_name}")

                        # Display the plot
                        st.pyplot(fig)
                    else:
                        st.error(f"Required columns ('Drain Time', 'Drain Current') not found in {file_name}")
                except Exception as e:
                    st.error(f"Failed to read {file_name}: {str(e)}")

        # Set `show_labels` to True when the "Analyze" button is clicked
        if st.button("Analyze"):
            st.session_state.show_labels = True

        # Show the label input fields if `show_labels` is True
        if st.session_state.show_labels:
            st.write("### Label Your Files")
            st.write("Please enter a label for each file:")

            # Display input fields with persistent labels
            for file_path in st.session_state.file_paths:
                label_key = f"label_{file_path}"
                st.session_state.file_labels[file_path] = st.text_input(
                    f"Label for {os.path.basename(file_path)}",
                    value=st.session_state.file_labels.get(file_path, ""),  # Use persistent value
                    key=label_key
                )

            # Show "Train" button only after the labels section
            if st.button("Train"):
                # Check if all labels have been entered
                if all(label.strip() for label in st.session_state.file_labels.values()):
                    st.write("### File Path and Label Mapping")
                    st.session_state.file_labels = {v: k for k, v in st.session_state.file_labels.items()}

                    # Call `combine` with the file_labels dictionary and display the plot
                    fig = new_tester(st.session_state.file_labels)

                    if fig is None:
                        st.error("No figure was returned by the combine function.")
                    elif not isinstance(fig, plt.Figure):
                        st.error("The returned object is not a Matplotlib Figure.")
                    else:
                        st.pyplot(fig)

                    # Reset the label section
                    st.session_state.show_labels = False
                else:
                    st.warning("Please enter a label for each file before training.")

    else:
        st.write("No files uploaded yet.")