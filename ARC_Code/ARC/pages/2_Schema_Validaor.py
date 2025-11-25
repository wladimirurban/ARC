import functions.f00_Sidebar as sidebar
import functions.f02_SchemaValidator as sv
import functions.f03_LabelValidator as lv

import streamlit as st
import pandas as pd

sidebar.sidebar()

st.title("Schema Validaor")
if  st.session_state._DL_DataLoaded == False:
    st.error("No dataset loaded. Please load a dataset first.")
else:
    # Data preview
    st.header("Data preview: " + st.session_state._DL_Filename)
    st.write(st.session_state._DF.head())

    ################
    ### INSIGHTS ###
    ################
    st.header("Schema insights")

    sv_data = sv.schemaValidatorTotal(st.session_state._DF, st.session_state._HasTimeStamp, st.session_state._TimeStampCol)
    
    # Total Records
    st.subheader("Total records")
    st.write(f"Total records: {sv_data['total_records']}")

    # Duplicates
    if sv_data['duplicates'] == 0:
        st.write("All records are unique")
    else:
        dup_perc = (sv_data['duplicates'] / sv_data['total_records']) * 100
        st.warning(f"Warning: {sv_data['duplicates']} duplicate rows found ({dup_perc:.2f}% of total).")

    # Sparsity
    st.subheader("Sparsity")
    sparsity = sv_data['sparsity']
    with st.expander(f"Missing values {sparsity['Total missing cells']} ({sparsity['Total missing percent']})"):
        if not sparsity['Col missing'].empty:
            st.warning("Missing values detected in the following columns:")
            st.dataframe(
                sparsity['Col missing']
                .rename("Missing (%)")
                .reset_index()
                .rename(columns={"index": "Column"}),
                width="stretch"
            )
        else:
            st.success("No missing values found")
    
    # Granularity
    st.subheader("Granularity")
    granularity = sv_data['granularity']
    st.write(f"Detected Granularity: {granularity}")

    # Features
    st.subheader("Features")
    features = sv_data['features']

    # Total features
    st.write(f"Total features: {features['Total Features']}")

    # Unique features
    if sv_data['features_unique'] == True:
        st.write("Every feature is unique")
    else: 
        st.warning("Waring duplicated features")

    # Mixed type columns
    mixed_type_cols = sv_data['mixed_type_cols']   
    if mixed_type_cols:
        st.warning("The following columns contain mixed data types:")
        st.markdown(", ".join(f"`{col}`" for col in mixed_type_cols))
    else:
        st.write("No features with mixed data types found.")

    # Feature set    
    t1, t2 = st.tabs(["Pi chart", "List"])
    with t1:
        labels = ["Numerical", "Categorical", "Bool", "Object", "Datetime"]
        values = [
            len(features['Numerical']),
            len(features['Categorical']),
            len(features['Bool']),
            len(features['Object']),
            len(features['Datetime'])
        ]

        fig = lv.plot_pie_chart(labels, values, 250)
        st.plotly_chart(fig, width="stretch")
    with t2:
        with st.expander(f"Numerical features ({len(features['Numerical'])})"):
            if features['Numerical']:
                st.write(", ".join(features['Numerical']))
            else:
                st.write("None")

        with st.expander(f"Categorical features ({len(features['Categorical'])})"):
            if features['Categorical']:
                st.write(", ".join(features['Categorical']))
            else:
                st.write("None")
        
        with st.expander(f"Bool features ({len(features['Bool'])})"):
            if features['Bool']:
                st.write(", ".join(features['Bool']))
            else:
                st.write("None")

        with st.expander(f"Object features ({len(features['Object'])})"):
            if features['Object']:
                st.write(", ".join(features['Object']))
            else:
                st.write("None")

        with st.expander(f"Datetime features ({len(features['Datetime'])})"):
            if features['Datetime']:
                st.write(", ".join(features['Datetime']))
            else:
                st.write("None")
    
    # TimeSpan
    st.subheader("Timespan")
    if st.session_state._HasTimeStamp == True:
        timeSpan = sv_data['time_analysis']
        if timeSpan['Exception'] is not None:
            st.error(timeSpan['Exception'])
        else:
            if sv_data['time_dup'] == True:
                st.warning("Duplicated timestamps found")
            else:
                st.write("All timestamps are unique")
            
            if sv_data['time_sorted'] == True:
                st.warning("Warning: Timestamps are not sorted.")
            else:
                st.success("Timesstamps are sorted")

            st.write(f"Start: {timeSpan['Start']}")
            st.write(f"End: {timeSpan['End']}")
            st.write(f"Duration: {timeSpan['Duration']}")
            st.write(f"Most common interval: {timeSpan['Most common interval']}")
            st.write(f"Average interval: {timeSpan['Average interval']}")

            try:
                st.session_state._DF[st.session_state._TimeStampCol] = pd.to_datetime(st.session_state._DF[st.session_state._TimeStampCol], errors='raise')
            except Exception:
                st.warning("Warning: Could not convert 'timestamp' to datetime format.")
    else:    
        st.write("No timestamp collumn avalible or selected")
    
    # Column presence
    st.subheader("Check column presence")
    st.session_state._SV_ReqCol = st.text_area("Paste or modify required collumns:", value=st.session_state._SV_ReqCol, height=100)
    if st.button(label="Check", width="stretch"):
        required_list = [col.strip() for col in st.session_state._SV_ReqCol.split(",") if col.strip()]
        if not isinstance(required_list, list):
            st.error("Please enter at least one column name.")
        else:
            req_col = sv.check_column_presence(st.session_state._DF, required_list)
            missing = req_col['Missing required columns']
            extra = req_col['Extra/unexpected columns']

            with st.expander(f"Missing Required Columns ({len(missing)})", expanded=True):
                if missing:
                    st.write(", ".join(missing))
                else:
                    st.write("None")

            with st.expander(f"Extra / Unexpected Columns ({len(extra)})", expanded=False):
                if extra:
                    st.write(", ".join(extra))
                else:
                    st.write("None")

    ####################
    ### MODIFICATION ###
    ####################
    st.header("Schema modification")

    # Header
    t, v, d, p  = st.columns(4)
    with t:
        st.write("Modification")
    with v:
        st.write("Variables")
    with d:
        st.write("Description")
    with p:
        st.write("Purpose")

    # horizontal line
    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    # Normalize column names
    t, v, d, p  = st.columns(4)
    with t:
        # run button
        if st.button(
            label = "Normalize feature names",
            width="stretch"
        ):
            st.session_state._DF = sv.normalize_column_names(st.session_state._DF)
            st.session_state._LabelCol = sv.normalize_column_name(st.session_state._LabelCol)
            st.session_state._TimeStampCol = sv.normalize_column_name(st.session_state._TimeStampCol)
            st.rerun()
    with v:
        st.write("")
    with d:
        st.write("Converts all feature names to lowercase and removes leading/trailing and internal spaces (special characters unchanged).")
    with p:
        st.write("Avoids subtle mismatches (e.g., Source IP → sourceip) and keeps features references consistent across the pipeline.")
    
    # horizontal line
    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    # Rename collumns
    t, v, d, p  = st.columns(4)
    with t:
        # run button
        if st.button(label = "Rename features", width="stretch"):
            rename_map = st.session_state._SV_RenameColMMap
            # rename dataset
            st.session_state._DF = sv.rename_columns(st.session_state._DF, rename_map)
            # rename label and timestamp columns for session
            st.session_state._LabelCol = sv.rename_column(st.session_state._LabelCol, rename_map)
            st.session_state._TimeStampCol = sv.rename_column(st.session_state._TimeStampCol, rename_map)
            # rerun to update
            st.rerun()
    with v:
        # input area
        with st.expander("Enter Rename Mapping (no brackets, use 'key = value')", expanded=False):
            # matrix input
            st.session_state._SV_RenameColM = st.text_area("Enter one rule per line:", value=st.session_state._SV_RenameColM, height=500)

            # read rename map
            st.session_state._SV_RenameColMMap = sv.parse_rename_matrix(st.session_state._SV_RenameColM)
            rename_map = st.session_state._SV_RenameColMMap
            if rename_map:
                with st.expander("Rename matrix parsed successfully!", expanded=False):
                    st.write(rename_map)
            else:
                st.warning("No valid rules found. Please use the format: `old_name = new_name`")
    with d:
        st.write("Cange specific feature names.")
    with p:
        st.write("Harmonize to the expected schema (e.g., match joins/model configs), resolve inconsistencies, and prevent downstream errors from mismatched names.")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )
    
    # sort by timestamp
    t, v, d, p = st.columns(4)
    with t:
        # check for timestamp presence
        if not st.session_state._HasTimeStamp:
            st.write("No timestamp collumn avalible or selected")
        if st.button(
            label="Sort by timestamp",
            disabled=not st.session_state._HasTimeStamp,
            key="_SV_SortByTimestamp",
            width="stretch"
        ):
            # sort by timestamp
            st.session_state._DF = sv.sort_by_timestamp(st.session_state._DF, st.session_state._TimeStampCol, st.session_state._SV_TDelta)
            st.rerun()
    with v:
        st.session_state._SV_TDelta = st.checkbox(
            label="Add time delta column",
            value=st.session_state._SV_TDelta,
            disabled=not st.session_state._HasTimeStamp,
            key="_SV_CheckTimeDeltas"
        )
    with d:
        st.write("Orders rows by the selected timestamp column, optionally adds delta_ts with seconds between consecutive records.")
    with p:
        st.write("Enforces correct event order for time-series analysis (drift checks, windowing, splits) and improves reproducibility.")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    # Drop collumn
    t, v, d, p = st.columns(4)
    with t:
        if st.button(
            label="Drop feature",
            disabled=  not st.session_state._HasTimeStamp,
            key="_SV_DropCol_Button",
            width="stretch"
        ):
            st.session_state._DF = sv.drop_column(st.session_state._DF, st.session_state._SV_ColToDrop)
            st.rerun()
    with v:
        cols = st.session_state._DF.columns.tolist()
        st.session_state._SV_ColToDrop = st.selectbox("Feature", cols, 1)
    with d:
        st.write("Removes the selected column if present.")
    with p:
        st.write("Eliminate irrelevant or leaky fields, keep the schema clean and speed up downstream processing/models.")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    # Drop duplicates
    t, v, d, p = st.columns(4)
    with t:
        if st.button(
            label="Drop duplicates",
            disabled=not st.session_state._HasTimeStamp,
            key="_SV_DropDuplicates",
            width="stretch"
        ):
            st.session_state._DF = sv.drop_duplicates(st.session_state._DF)
            st.rerun()
    with v:
        st.write("")
    with d:
        st.write("Removes exact duplicate rows (keeps the first) and logs how many were removed.")
    with p:
        st.write("Prevent double counting, improve data quality and keep metrics/models reliable.")

    # Download data
    st.header("Downloads")

    csv_bytes = st.session_state._DF.to_csv(index=False).encode("utf-8")

    if st.session_state._DL_DataLoaded == True:
        csv_bytes = st.session_state._DF.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download dataset as CSV",
            data=csv_bytes,
            file_name="dataset.csv",
            mime="text/csv",
            width="stretch"
        )