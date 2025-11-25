import streamlit as st

import functions.f00_Logger as logger

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# generate sidebar
def sidebar():
    if st.session_state._DL_DataLoaded == True:
        st.sidebar.subheader("Dataset information:")
        
        # dataset name
        st.sidebar.markdown("Dataset name: " + st.session_state._DL_Filename)

        # timestamp columns
        if st.session_state._HasTimeStamp:
            st.sidebar.markdown("Timestamp column: " + st.session_state._TimeStampCol)
        
        # label column
        if st.session_state._HasLabel:
            st.sidebar.markdown("Label column: " + st.session_state._LabelCol)
        
    # Log
    logger.show_log()