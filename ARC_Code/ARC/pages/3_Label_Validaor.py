import streamlit as st

import functions.f00_Sidebar as sidebar
import functions.f02_SchemaValidator as sv
import functions.f03_LabelValidator as lv

sidebar.sidebar()

st.title("Label validaor")
if  st.session_state._DL_DataLoaded == False:
    st.error("No dataset loaded. Please load a dataset first.")
elif st.session_state._HasLabel == False:
    st.error("Dataset is not labeled. Label the dataset to validate the labeling")
else:
    st.header("Data preview: " + st.session_state._DL_Filename)
    st.write(st.session_state._DF.head())

    lv_data = lv.labelValidaorTotal(st.session_state._DF, st.session_state._LabelCol)

    ################
    ### INSIGHTS ###
    ################
    
    # Consitency
    st.header("Labeling consitency")

    # Missing labels
    if lv_data['missing_labels'] == True:
        st.warning("Missing labels detected.")
    else:
        st.write("No missing labels detected.")

    # Inconsistent labeling
    if lv_data['inconsistent_groups']:
        st.warning(f"Possible label spelling inconsistencies: {lv_data['inconsistent_groups']}")
    else:
        st.write("The label spelling seems to be consistent")
    
    # Label Entropy
    st.header("Label Entropy")
    st.write(f"Label Entropy: {lv_data['label_entropy_value']:.2f} bits")
    st.write(f"Interpretation: {lv_data['entropy_interpretation']}")
    
    # Class distribution
    st.header("Class distribution")
    class_counts = lv_data['class_counts']
    labels = class_counts.index.tolist()
    values = class_counts.values.tolist() 
    t1, t2, t3 = st.tabs(["Pie chart", "Table", "Bar chart"])
    with t1:
        st.plotly_chart(lv.plot_pie_chart(labels, values, height=300))
    with t2:
        st.write("Class distribution:\n", class_counts)
    with t3:
        st.plotly_chart(lv.plot_bar_chart(labels, values, height=300))
    
    # Rare classes
    st.subheader("Rare classes")
    st.session_state._LV_InputRareClasses = st.number_input(
        "Define rare class threshold (percentage of total samples)",
        min_value=0,
        max_value=15,
        value=st.session_state._LV_InputRareClasses,
        step=1,
        key="rare_class_threshold"
    )
    rare_classes = lv.get_rare_classes(st.session_state._DF, st.session_state._LabelCol, st.session_state._LV_InputRareClasses)
    if rare_classes.empty:
        st.write(f"No class represents less then {st.session_state._LV_InputRareClasses}% ")
    else:
        formatted = ", ".join(
            f"{label}: {ratio:.2%}" for label, ratio in rare_classes.items()
        )
        st.warning(f"Rare classes (>{st.session_state._LV_InputRareClasses}%) detected: {formatted}")
    
    # Dominant classes
    st.subheader("Dominant classes")
    st.session_state._LV_InputDominantClasses = st.number_input(
        "Define dominant class threshold (percentage of total samples)",
        min_value=70,
        max_value=100,
        value=st.session_state._LV_InputDominantClasses,
        step=1,
        key="dominant_class_threshold"
    )
    dominant_classes = lv.get_dominant_classes(st.session_state._DF, st.session_state._LabelCol, st.session_state._LV_InputDominantClasses)
    if dominant_classes.empty:
        st.write(f"No class represents more then {st.session_state._LV_InputDominantClasses}% ")
    else:
        formatted = ", ".join(
            f"{label}: {ratio:.2%}" for label, ratio in dominant_classes.items()
        )
        st.warning(f"Dominant classes (>{st.session_state._LV_InputDominantClasses}%) detected: {formatted}")
    
    
    #Temporal drift
    st.header("Temporal drift")
    if st.session_state._HasTimeStamp:
        st.write("Show temporal drift:")
        st.session_state._LV_TDTimeline = st.checkbox(
            "on a timeline",
            value=st.session_state._LV_TDTimeline,
            key="td_timeline_checkbox"
        )
        st.session_state._LV_TDTime = st.checkbox(
            "in time bins by time",
            value=st.session_state._LV_TDTime,
            key="td_time_checkbox"
        )
        st.session_state._LV_TDRecords = st.checkbox(
            "in time bins by records",
            value=st.session_state._LV_TDRecords,
            key="td_records_checkbox"
        )

        if st.session_state._LV_TDTime == True or st.session_state._LV_TDRecords == True:
            st.session_state._LV_TDNumBins = st.number_input(
                "Select number of time bins",
                min_value=1,
                max_value=100,
                value=st.session_state._LV_TDNumBins,
                step=1,
                key="number of timebins"
            )
        
        if st.session_state._LV_TDTimeline == True:
            st.subheader("Label distribution timeline")
            fig = lv.plot_state_timeline(st.session_state._DF, st.session_state._LabelCol, st.session_state._TimeStampCol)
            if fig is not None:
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("No data available for timeline.")
        
        if st.session_state._LV_TDTime == True:
            st.subheader("Label distribution over time-bins grouped by time")
            table = lv.get_timebin(st.session_state._DF, st.session_state._LabelCol, st.session_state._TimeStampCol, "time", st.session_state._LV_TDNumBins)
            t1, t2 = st.tabs(["Chart", "Table"])
            with t1:
                fig = lv.plot_timebin(table, height=450)
                if fig:
                    st.plotly_chart(fig, width="stretch")
            with t2:
                table_display = lv.format_time_bin_table(table)
                st.dataframe(table_display, hide_index=True)
            
        if st.session_state._LV_TDRecords == True:
            st.subheader("Label distribution over time-bins grouped by records")
            table = lv.get_timebin(st.session_state._DF, st.session_state._LabelCol, st.session_state._TimeStampCol, "records", st.session_state._LV_TDNumBins)
            t1, t2 = st.tabs(["Chart", "Table"])
            with t1:
                st.write("G")
                fig = lv.plot_timebin(table, height=300)
                if fig:
                    st.plotly_chart(fig, width="stretch")
            with t2:
                table_display = lv.format_time_bin_table(table)
                st.dataframe(table_display, hide_index=True)
    
    else:
        st.write("No timestamp selected")
    
    st.header("Label modification")
    t, v, d, p  = st.columns(4)
    with t:
        # run button
        if st.button(label = "Rename label", width="stretch"):
            rename_map = st.session_state._LV_RenameMMap
            # rename dataset
            st.session_state._DF = lv.rename_row(st.session_state._DF, st.session_state._LabelCol, rename_map)
            # rerun to update
            st.rerun()
    with v:
        # input area
        with st.expander("Enter Rename Mapping (no brackets, use 'key = value')", expanded=False):
            default_text = st.session_state._LV_RenameM
            # matrix input
            matrix_input = st.text_area("Enter one rule per line:", value=default_text, height=500)

            st.session_state._LV_RenameM = matrix_input
            # read rename map
            rename_map = sv.parse_rename_matrix(matrix_input)
            st.session_state._LV_RenameMMap = rename_map

            if rename_map:
                with st.expander("Rename matrix parsed successfully!", expanded=False):
                    st.write(rename_map)
            else:
                st.warning("No valid rules found. Please use the format: `old_name = new_name`")
    with d:
        st.write("Replaces values in a chosen column using your mapping.")
    with p:
        st.write("Standardize label taxonomy (merge aliases, fix typos) so evaluation metrics and reports stay consistent.")


    # Download data
    if  st.session_state._DL_DataLoaded == True:

        st.subheader("Download Dataset")

        csv_bytes = st.session_state._DF.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download dataset as CSV",
            data=csv_bytes,
            file_name="dataset.csv",
            mime="text/csv",
            width="stretch"
        )        
        

        