import streamlit as st

import functions.f00_Sidebar as sidebar
import functions.f04_Splitter as sd

sidebar.sidebar()

st.title("Splitter")

if st.session_state._DL_DataLoaded == False:
    st.error("No dataset loaded. Please load a dataset first.")
elif st.session_state._HasLabel == False:
    st.write("No label column selected")
else:

    #############
    ### SPLIT ###
    #############

    st.header("Select Dataset Split Method")

    # DESCRIPTION
    s, a, u = st.columns(3)
    with s:
        st.write("Splitmethod")
    with a:
        st.write("Advantages")
    with u:
        st.write("Use case")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    s, a, u = st.columns(3)
    with s:
        st.write("Random Split")
    with a:
        st.write("Simple and fast; good baseline; evenly distributes samples when dataset is balanced.")
    with u:
        st.write("Balanced datasets with no time dependency; quick experimentation.")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    s, a, u = st.columns(3)
    with s:
        st.write("Stratified Split")
    with a:
        st.write("Preserves class proportions across training, validation, and test sets.")
    with u:
        st.write("Imbalanced classification problems; ensures fair evaluation.")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    s, a, u = st.columns(3)
    with s:
        st.write("Time-based Split")
    with a:
        st.write("Prevents future data from leaking into training; respects temporal order.")
    with u:
        st.write("Time series, forecasting, cybersecurity logs with timestamps.")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    # split method selection
    split_methods = [
        "Random Split",
        "Stratified Split",
        "Time-based Split"
    ]
    default_method = st.session_state.get("_SP_SplitMethod", split_methods[0])
    default_method_index = split_methods.index(default_method) if default_method in split_methods else 0
    st.session_state._SP_SplitMethod = st.selectbox(
        label = "Choose splitting method:",
        options = split_methods,
        index = default_method_index
    )
    method = st.session_state._SP_SplitMethod
    
    # random state / gap ratio input
    if(method in ["Stratified Split", "Random Split"]):
        st.session_state._SP_RandomState = st.number_input(
            "Define random state for reproducibility (default: 42)",
            min_value=0,
            max_value=100,
            value=st.session_state._SP_RandomState,
            step=1,
            key="split_random_state_input"
        )
    if(method == "Time-based Split"):
        st.session_state._SP_GapRatio = st.number_input(
            "Define gap ratio between splits (percentage of total data to exclude between splits)",
            min_value=0,
            max_value=100,
            value=st.session_state._SP_GapRatio,
            step=1,
            key="split_random_state_input"
        )
    
    # split sizes
    st.session_state._SP_TestSize = st.slider(label="Test set size (%)", min_value=int(10), max_value=int(40), value= int(st.session_state._SP_TestSize), step=1)
    st.session_state._SP_ValSize = st.slider(label="Validation set size (%)", min_value=10, max_value=40, value=int(st.session_state._SP_ValSize), step=1)

    # parameters
    test_size = st.session_state._SP_TestSize  / 100
    val_size = st.session_state._SP_ValSize  / 100
    gap_ratio = st.session_state._SP_GapRatio / 100
    random_state = st.session_state._SP_RandomState

    df= st.session_state._DF
    label_col = st.session_state._LabelCol  
    time_col = st.session_state._TimeStampCol
    # split button
    if st.button(label="Split Dataset", width="stretch"):
        X_train, y_train, X_validate, y_validate, X_test, y_test = sd.split_dataset(df=df, label_col=label_col, time_col=time_col, method=method, test_size=test_size, val_size=val_size, gap_ratio=gap_ratio, random_state=random_state)

        st.session_state._SP_X_Train = X_train
        st.session_state._SP_y_Train = y_train
        st.session_state._SP_X_Validate = X_validate
        st.session_state._SP_y_Validate = y_validate
        st.session_state._SP_X_Test = X_test
        st.session_state._SP_y_Test = y_test

        st.rerun()

#####################
### QUALITY CHECK ###
#####################

    if st.session_state._SP_IsSplit == True:
        # get splits
        X_train = st.session_state._SP_X_Train
        y_train = st.session_state._SP_y_Train
        X_validate = st.session_state._SP_X_Validate
        y_validate = st.session_state._SP_y_Validate 
        X_test = st.session_state._SP_X_Test
        y_test = st.session_state._SP_y_Test

        _val_size = st.session_state._SP_ValSize
        _test_size = st.session_state._SP_TestSize  

        # report split sizes
        st.header("Split")
        st.write(f"Training set: {len(X_train)} samples ({100 - _test_size - _val_size}%)")
        st.write(f"Validation set: {len(X_validate)} samples ({_val_size}%)")
        st.write(f"Test set: {len(X_test)} samples ({_test_size}%)")

        st.header("Quality checks")

        # report unseen labels
        unseen = sd.getunseen(y_train, y_validate, y_test)
        if unseen['unseen_val']:
            st.warning(f"Unseen labels in the validation set (not present in training): {unseen['unseen_val']}")
        else:
            st.write("No unseen labels in the validation set.")

        if unseen['unseen_test']:
            st.warning(f"Unseen labels in the test set (not present in training): {unseen['unseen_test']}")
        else:
            st.write("No unseen labels in the test set.")

        # report distribution
        dist_set = sd.getDistributionSet(y_train, y_validate, y_test)
        st.write("Label distribution by split (percentages):")
        t1, t2 = st.tabs(["Chart", "Table"])
        with t1:
            fig_set = sd.plot_distribution_bars(dist_set)
            st.plotly_chart(fig_set, width="stretch", key= "Chart_Split_Dist")
        with t2:
            t = sd.make_distribution_table(dist_set)
            st.dataframe(t, hide_index=True)
        
        dist_label = sd.getDistributionLabel(y_train, y_validate, y_test)
        st.write("Label distribution by label (percentages):")
        t1, t2 = st.tabs(["Chart", "Table"])
        with t1:
            fig_set = sd.plot_distribution_label(dist_label)
            st.plotly_chart(fig_set, width="stretch", key= "Chart_Label_Dist")
        with t2:
            t = sd.make_distribution_table(dist_label)
            st.dataframe(t, hide_index=True)

        dist_total = sd.getDistributionTotal(y_train, y_validate, y_test)
        st.write("Label distribution total (percentages):")
        t1, t2 = st.tabs(["Chart", "Table"])
        with t1:
            fig_set = sd.plot_distribution_bars(dist_total)
            st.plotly_chart(fig_set, key= "Chart_Total_Dist")
        with t2:
            t = sd.make_distribution_table(dist_total)
            st.dataframe(t, hide_index=True)
        
        ##############
        ### RETURN ###
        ##############

        st.subheader("Download Datasets")

        def convert_df(df): return df.to_csv(index=False).encode('utf-8')

        st.download_button("Download X_train", convert_df(X_train), "X_train.csv", "text/csv", width="stretch")
        st.download_button("Download y_train", convert_df(y_train.to_frame()), "y_train.csv", "text/csv", width="stretch")
        st.download_button("Download X_validate", convert_df(X_validate), "X_validate.csv", "text/csv", width="stretch")
        st.download_button("Download y_validate", convert_df(y_validate.to_frame()), "y_validate.csv", "text/csv", width="stretch")
        st.download_button("Download X_test", convert_df(X_test), "X_test.csv", "text/csv", width="stretch")
        st.download_button("Download y_test", convert_df(y_test.to_frame()), "y_test.csv", "text/csv", width="stretch")