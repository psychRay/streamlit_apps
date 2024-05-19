# a examplar python script for web app

import io
import os
# import json
# import base64
# import imageio
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from atlasreader import create_output
from nilearn.plotting import plot_glass_brain
from streamlit.runtime.state import session_state

# Function to handle uploaded file
def handle_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

# the function to save the Matplotlib fig into memory in binary formmat
@st.cache_data
def get_img_from_brain(brain_img, thresh, colormap='bwr'):
    
    plot = plot_glass_brain(
        brain_img, 
        threshold = thresh, 
        cmap      = colormap, 
        plot_abs  = False
    )
    
    buf = io.BytesIO()
    plot.savefig(buf, dpi=300)
    buf.seek(0)
    
    return buf


# -------------------------
# create streamlit web app
# -------------------------
st.set_page_config(layout="wide")
st.markdown(
    """
    <h1 style='white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>
        Cluster Information in Given Brain Image
    </h1>
    """,
    unsafe_allow_html=True)

if 'brain_img' not in st.session_state:
    st.session_state.brain_img = {}

# left_column, right_column = st.columns(2)

# Allow user to upload a file
with st.sidebar:
    st.markdown('## Upload Brain Image')
    brain_img = st.file_uploader("Select brain image (NIfTI, NII, etc.)", type=["nii", "nii.gz"])

    # Get user inputs for threshold and cluster size
    threshold = st.number_input("Threshold value of voxel:")
    cluster_size = st.number_input("Cluster size:")
    atlas = st.multiselect(
        "Select Atlas for cluster label:", 
        [ 
        'aal', 
        'aicha', 
        'desikan_killiany', 
        'destrieux',
        'diedrichsen',
        'harvard_oxford',
        'juelich',
        'talairach_ba'], 
        default=['aal', 'harvard_oxford'],
        )
    direction_value = st.selectbox(
        "The direction of image value:", 
        ('pos', 'neg', 'both'))

# Run the main analyses:
with st.container():
    if brain_img is not None:
        if not st.session_state.brain_img:
            file_path = handle_uploaded_file(brain_img)
            # st.session_state.brain_img['img_buf']  = img_buf
            st.session_state.brain_img['img_path'] = file_path
        
    elif 'output_data' in st.session_state:
        del st.session_state.brain_img
        del st.session_state.output_data

    # Render the binary formatted image saved in memory through streamlit
    if brain_img is not None:
        st.markdown('## Cluster visualization in glass brain:')
        img_buf = get_img_from_brain(st.session_state.brain_img['img_path'], thresh=threshold)
        st.image(
            img_buf, 
            # caption='Glass brain visualization of thresholded clusters', 
            width=800,
            # use_column_width=False
            )

    # Run create_output only if it hasn't been run before
    if brain_img is not None:
        if 'output_data' not in st.session_state or st.button('Update analysis'):
            with tempfile.TemporaryDirectory() as tmpdirname:
                output_path = os.path.join(tmpdirname, "output")
                create_output(
                    st.session_state.brain_img['img_path'],
                    atlas=atlas,
                    voxel_thresh=threshold,
                    cluster_extent=cluster_size,
                    direction='both',
                    outdir=output_path)
                
                # Store data into session_state
                st.session_state.output_data = {}
                for filename in os.listdir(output_path):
                    file_path = os.path.join(output_path, filename)
                    
                    if filename.endswith('.csv') and 'cluster' in filename:
                        df = pd.read_csv(file_path)
                        st.session_state.output_data['cluster_csv'] = df
                        
                    elif filename.endswith('.png') and 'cluster' in filename:
                        with open(file_path, "rb") as f:
                            cluster_label = filename.split('.')[0].split('_')[-1]
                            key_png = f'{cluster_label}_png'
                            st.session_state.output_data[key_png] = f.read()

        if 'cluster_csv' in st.session_state.output_data:
            st.markdown('## Cluster information:')
            st.dataframe(st.session_state.output_data['cluster_csv'])

        # User interaction to show results
        show_results = st.checkbox('Show image per cluster')
        if show_results and 'output_data' in st.session_state:
            # Display images
            for key, value in st.session_state.output_data.items():
                if key.endswith('png'):
                    caption = key.split('_')[0]
                    st.markdown(f"### {caption}")
                    st.image(value)


