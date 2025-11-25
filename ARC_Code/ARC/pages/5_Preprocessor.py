import streamlit as st

import functions.f05_Preprocessor as pp
import functions.f00_Sidebar as sidebar
import functions.f00_Logger as logger

import pandas as pd

sidebar.sidebar()

st.title("Preprocessor")
st.header("Model insights")
with st.expander("Random Forest (RF)"):
    # Header
    a, w, p, c = st.columns(4)
    with a:
        st.markdown("Advantages / Strengths")
    with w:
        st.markdown("Weaknesses / Limitations")
    with p:
        st.markdown("Performance on Imbalanced Data")
    with c:
        st.markdown("Computational Cost")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    a, w, p, c = st.columns(4)
    with a:
        st.markdown("""
            - Handles mixed data types
            - Robust to noise and overfitting
            - Works well without heavy preprocessing
            - Can measure feature importance
        """)
    with w:
        st.markdown("""
            - Can be slower on very large datasets
            - Model size can be large
            - Less interpretable than simple models
        """)
    with p:
        st.write("Moderate to good, but may bias toward majority class; needs class weighting or resampling")
    with c:
        st.write("Medium — parallelizable, but training many trees increases cost")

with st.expander("Gradient Boosting (XGBoost / LightGBM / CatBoost)"):
    # Header
    a, w, p, c = st.columns(4)
    with a:
        st.markdown("Advantages / Strengths")
    with w:
        st.markdown("Weaknesses / Limitations")
    with p:
        st.markdown("Performance on Imbalanced Data")
    with c:
        st.markdown("Computational Cost")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    a, w, p, c = st.columns(4)
    with a:
        st.markdown("""
            - Often achieves high accuracy
            - Handles non-linear relationships well
            - Built-in handling of missing values
            - Feature importance ranking
        """)
    with w:
        st.markdown("""
            - Sensitive to hyperparameters
            - Prone to overfitting if not tuned
            - Slower training than RF
        """)
    with p:
        st.markdown("Good, especially with built-in class weights; still benefits from resampling")
    with c:
        st.markdown("Medium to high — boosting is sequential, but LightGBM is faster")

with st.expander("Support Vector Machine (SVM)"):
    # Header
    a, w, p, c = st.columns(4)
    with a:
        st.markdown("Advantages / Strengths")
    with w:
        st.markdown("Weaknesses / Limitations")
    with p:
        st.markdown("Performance on Imbalanced Data")
    with c:
        st.markdown("Computational Cost")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    a, w, p, c = st.columns(4)
    with a:
        st.markdown("""
            - Performs well in high-dimensional spaces
            - Effective with small to medium datasets
            - Can use different kernels for flexibility
        """)
    with w:
        st.markdown("""
            - Not efficient with very large datasets
            - Needs feature scaling
            - Difficult to tune kernel parameters
        """)
    with p:
        st.markdown("Medium, can adjust class weights, but may still struggle if imbalance is extreme")
    with c:
        st.markdown("High for large datasets — scales poorly with number of samples")

with st.expander("Neural Network (MLP)"):
    # Header
    a, w, p, c = st.columns(4)
    with a:
        st.markdown("Advantages / Strengths")
    with w:
        st.markdown("Weaknesses / Limitations")
    with p:
        st.markdown("Performance on Imbalanced Data")
    with c:
        st.markdown("Computational Cost")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    a, w, p, c = st.columns(4)
    with a:
        st.markdown("""
            - Flexible, models complex patterns
            - Can handle non-linear interactions
            - Can learn embeddings for categorical features
        """)
    with w:
        st.markdown("""
            - Needs large datasets to avoid overfitting
            - Requires careful tuning
            - Less interpretable
        """)
    with p:
        st.markdown("Good with class weighting or focal loss; otherwise can be biased toward majority class")
    with c:
        st.markdown("High — training can be slow, especially with large networks")

with st.expander("Logistic Regression (LR)"):
    # Header
    a, w, p, c = st.columns(4)
    with a:
        st.markdown("Advantages / Strengths")
    with w:
        st.markdown("Weaknesses / Limitations")
    with p:
        st.markdown("Performance on Imbalanced Data")
    with c:
        st.markdown("Computational Cost")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    a, w, p, c = st.columns(4)
    with a:
        st.markdown("""
            - Simple, interpretable, and fast
            - Works well as a baseline
            - Easy to implement and tune
        """)
    with w:
        st.markdown("""
            - Limited to linear relationships
            - Struggles with complex, non-linear patterns
        """)
    with p:
        st.markdown("Poor by default — needs class weighting or resampling")
    with c:
        st.markdown("Very low — one of the fastest algorithms")

ALGORITHMS = [
    "None",
    "Random Forest (RF)",
    "Gradient Boosting (XGBoost / LightGBM / CatBoost)",
    "Support Vector Machine (SVM)",
    "Neural Network (MLP)",
    "Logistic Regression (LR)"
]

_Model = st.session_state._PP_Model

st.selectbox(
    label="Choose an algorithm:", 
    options=ALGORITHMS,
    key="_PP_Model"
    )

if st.button(label=f"Apply suggested preset for {_Model}", width="stretch"):
    configuration = pp.modelPreset(_Model, st.session_state._HasTimeStamp)

    for key, value in configuration.items():
        st.session_state["_PP_"+key] = value



st.header("Preprocessing steps")

################
### Cleaning ###
################
with st.expander("Cleaning"):
    # Header
    t, v, d, p, r  = st.columns(5)
    with t:
        st.write("Preprocessing step")
    with v:
        st.write("Settings")
    with d:
        st.write("Description")
    with p:
        st.write("Purpose")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )
    #Drop Duplicates
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="Drop duplicates",
            key="_PP_DD"
            )
    with v:
        st.write("")
    with d:
        st.write("Removes duplicate rows from the dataset.")
    with p:
        st.write(
            "Ensures each training sample contributes only once. "
            "Duplicates can bias the model, especially in small datasets."
            )

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    #Missing Values
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="Impute missing values", 
            key="_PP_MV"
            )
    with v:
        _MV_N_options = ["Median", "Mode"]
        st.selectbox(
            label="Numerical", 
            options=_MV_N_options,
            key="_PP_MV_N"
            )
        
        _MV_C_options = ["Most Frequent"]
        st.selectbox(
            label="Categorical", 
            options=_MV_C_options,
            key="_PP_MV_C"
            )
    with d:
        st.write(
            "Fills in missing entries for both numerical and categorical features. "
            "Numerical can be imputed with **Median** (robust to outliers) or **Mode**. "
            "Categorical uses the **Most Frequent** class by default."
            )
    with p:
        st.write(
            "Prevents models from failing on NaN values. "
            "Maintains dataset size instead of dropping rows/columns."
            )

################
### Encoding ###
################
with st.expander("Encoding"):
    # Header
    t, v, d, p  = st.columns(4)
    with t:
        st.write("Preprocessing step")
    with v:
        st.write("Settings")
    with d:
        st.write("Description")
    with p:
        st.write("Purpose")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )
    #Bucket rare categories
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="Buckeet rare categories", 
            key="_PP_BR"
            )
    with v:
        st.number_input(
            label="min_freq",
            min_value= 1, 
            max_value= 1000,
            key="_PP_BR_MF", 
            help="Min occurrences to keep a category"
            )
    with d:
        st.write(
            "Combines infrequent categories into a single 'Other' bucket. "
            "`min_freq` controls how many times a category must appear to be kept."
            )
    with p:
        st.write("Prevents models from overfitting to categories with very few samples.")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    #Generic & Advanced Encoding
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="Generic/Advanced Encoding", 
            key="_PP_EN_GA")
    with v:
        _EN_GA_O_options = ["One Hot Encoding", "Ordinary"]
        st.selectbox(
            label="Mode", 
            options=_EN_GA_O_options,
            key="_PP_EN_GA_O"
            )
    with d:
        st.write(
            "Encodes categorical features into numeric format. "
            "**One Hot Encoding** creates binary indicators. "
            "**Ordinal** assigns integer values to categories."
            )
    with p:
        st.write(
            "Most ML models need numerical input. "
            "One Hot is better for non-ordinal categories; "
            "Ordinal is more compact but assumes an order."
            )

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    #GBM Native Categorical Prep
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="GBM Native Categorical Prep", 
            key="_PP_GBM"
            )
    with v:
        _GBM_M_options = ["CatBoost", "Light GBM", "XG Boost"]
        st.selectbox(
            label="Mode", 
            options=_GBM_M_options,
            key="_PP_GBM_M"
            )
    with d:
        st.write("Uses the gradient boosting library’s built-in categorical handling.")
    with p:
        st.write("Boosting libraries can internally encode categories efficiently.")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    #Count Encoding
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="Count Encoding",
            key="_PP_CE"
            )
    with v:
        st.write("")
    with d:
        st.write("Replaces a category with the frequency of its occurrence in the dataset.")
    with p:
        st.write(
            "Adds statistical signal to categorical features. "
            "Works well for tree-based models."
            )

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    #Target mean encoding
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="Target mean encoding",
            key="_PP_TME"
            )
    with v:
        st.number_input(
            label="n_splits",
            min_value= 2,
            max_value= 20,
            key="_PP_TME_NS"
            )
        st.number_input(
            label= "noise_std",
            min_value= 0.0,
            max_value= 5.0,
            key="_PP_TME_NOS"
            )
        st.number_input(
            label="global_smoothing",
            min_value= 0.0, 
            max_value= 100.0,
            key="_PP_TME_GS"
            )
    with d:
        st.write(
            "Encodes categories by replacing them with the mean of the target variable. "
            "`n_splits` adds cross-validation safety. "
            "`noise_std` adds random noise to prevent leakage. "
            "`global_smoothing` balances rare vs. frequent categories."
            )
    with p:
        st.write("Captures target-category relationships. Powerful but risky if not regularized.")

#####################################
### Feature Filtering / Selection ###
#####################################
with st.expander("Feature Filtering / Selection"):
    # Header
    t, v, d, p  = st.columns(4)
    with t:
        st.write("Preprocessing step")
    with v:
        st.write("Settings")
    with d:
        st.write("Description")
    with p:
        st.write("Purpose")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )
    #Drop constant / low variance
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="Drop constant / low variance", 
            key="_PP_DC"
            )
    with v:
        st.number_input(
            label="variance threshold", 
            min_value=0.0, 
            max_value=0.5,
            key="_PP_DC_V"
            )
    with d:
        st.write(
            "Removes features with little to no variability. "
            "`variance threshold` sets the minimum variance a feature must have."
            )
    with p:
        st.write("Eliminates useless predictors that add noise or slow down training.")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    #Drop highly correlated numerics
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="Drop highly correlated numerics", 
            key="_PP_DHC"
            )
    with v:
        st.number_input(
            label="corr threshold", 
            min_value=0.80, 
            max_value=0.999,
            key="_PP_DHC_T"
            )
    with d:
        st.write(
            "Removes features that are highly correlated with each other. "
            "`corr threshold` sets the cutoff (e.g., 0.95)."
            )
    with p:
        st.write(
            "Prevents multicollinearity and redundancy. "
            "Improves interpretability and reduces overfitting.")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    #Feature selection (RF importance)
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="Feature selection by RF importance", 
            key="_PP_FS_RF"
            )
    with v:
        st.number_input(
            label="keep top-N", 
            min_value=1, 
            max_value=5000,
            key="_PP_FS_RF_TN"
            )
    with d:
        st.write("Keeps only the top-N most important features according to a Random Forest.")
    with p:
        st.write(
            "Reduces dimensionality and noise. "
            "Focuses training on most predictive variables."
            )

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    #Feature selection by Mutual Info
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="Feature selection by Mutual Info", 
            key="_PP_FS_MI"
            )
    with v:
        st.number_input(
            label="keep top-k", 
            min_value=1, 
            max_value=5000,
            key="_PP_FS_MI_TK"
            )
    with d:
        st.write("Selects top-k features based on mutual information with the target.")
    with p:
        st.write("Captures both linear and non-linear relationships.")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    #Feature selection by ANOVA F
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="Feature selection by ANOVA F", 
            key="_PP_FS_A"
            )
    with v:
        st.number_input(
            label="keep top-k", 
            min_value=1, 
            max_value=5000,
            key="_PP_FS_A_TK"
        )
    with d:
        st.write("Keeps top-k features using ANOVA F-test for variance across classes.")
    with p:
        st.write("Effective for linear models, assumes numeric distributions.")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    #PCA reduction
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="PCA reduction", 
            key="_PP_PCA"
            )
    with v:
        st.number_input(
            label="n_components (float<=1 or int)",
            min_value=0.10, 
            max_value=1.00,
            key="_PP_PCA_N")
    with d:
        st.write(
            "Reduces dimensionality by projecting features into fewer components. "
            "`n_components` can be fraction (variance explained) or integer."
            )
    with p:
        st.write(
            "Removes redundancy, speeds up training. "
            "Helps models sensitive to collinearity."
            )

###############################
### Scaling & Normalization ###
###############################
with st.expander("Scaling & Normalization"):
    # Header
    t, v, d, p  = st.columns(4)
    with t:
        st.write("Preprocessing step")
    with v:
        st.write("Settings")
    with d:
        st.write("Description")
    with p:
        st.write("Purpose")
    with r:
        st.write("Recomendation")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )
    #Scaling
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="Scaling", 
            key="_PP_SC"
            )
    with v:
        _SC_S_options = ["Z-Score", "Robust", "MinMax"]
        st.selectbox(
            label="Scaler", 
            options=_SC_S_options,
            key="_PP_SC_S"
            )
        if st.session_state._PP_SC_S == "MinMax":
            st.number_input(
                label="min (a)", 
                min_value=-10.0, 
                max_value=10.0,
                key="_PP_SC_S_MIN"
                )
            st.number_input(
                label="max (b)", 
                min_value=-10.0, 
                max_value=10.0,
                key="_PP_SC_S_MAX"
                )
    with d:
        st.write(
            "Rescales features to standard ranges. "
            "`Z-Score`: mean=0, std=1. "
            "`Robust`: median/IQR scaling (robust to outliers). "
            "`MinMax`: maps values into [a,b]."
            )
    with p:
        st.write("Ensures models relying on distances or gradients train correctly.")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    #Row-wise L2 normalization
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="Row-wise L2 normalization", 
            key="_PP_L2N"
            )
    with v:
        st.write("")
    with d:
        st.write("Normalizes each row (sample) to unit length using L2 norm.")
    with p:
        st.write(
            "Useful when feature magnitude is arbitrary but direction matters "
            "(e.g., text vectors, similarity models)."
            )

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

###############################
### Outliers & Transforms ###
###############################
with st.expander("Outliers & Transforms"):
    # Header
    t, v, d, p  = st.columns(4)
    with t:
        st.write("Preprocessing step")
    with v:
        st.write("Settings")
    with d:
        st.write("Description")
    with p:
        st.write("Purpose")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )
    # Outliers clipping
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="Clipping", 
            key="_PP_OC"
            )
    with v:
        _OC_M_options = ["IQR", "Percentile"]
        st.selectbox(
            label="Clipping method", 
            options=_OC_M_options,
            key="_PP_OC_M"
            )
        if st.session_state._PP_OC_M == "IQR":
            st.number_input(
                label="whisker width", 
                min_value=0.5, 
                max_value=10.0,
                key="_PP_OC_WW"
                )
        if st.session_state._PP_OC_M == "Percentile":
            st.number_input(
                label="low %", 
                min_value=0.0, 
                max_value=10.0,
                key="_PP_OC_PL"
                )
            st.number_input(
                label="high %", 
                min_value=90.0, 
                max_value=100.0,
                key="_PP_OC_PH"
                )
    with d:
        st.write(
            "Clips extreme values using either: "
            "`IQR` (interquartile range, whiskers define bounds), "
            "`Percentile` (caps at low/high percentiles)."
            )
    with p:
        st.write("Prevents extreme values from skewing the model or scaling.")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    #log1p transform
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="log1p transform", 
            key="_PP_LOG"
            )
    with v:
        st.write("")
    with d:
        st.write("Applies log(1+x) transform to skewed numeric features.")
    with p:
        st.write("Reduces skewness, stabilizes variance, improves linear model fit.")

#######################
### Class Imbalance ###
#######################
with st.expander("Class Imbalance"):
    # Header
    t, v, d, p  = st.columns(4)
    with t:
        st.write("Preprocessing step")
    with v:
        st.write("Settings")
    with d:
        st.write("Description")
    with p:
        st.write("Purpose")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )
    #Class weights
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="Class weights", 
            key="_PP_CW"
            )
    with v:
        _CW_S_options = ["balanced","uniform"]
        st.selectbox(
            label="scheme", 
            options=_CW_S_options, 
            key="_PP_CW_S"
            )
    with d:
        st.write(
            "Adjusts model training to give more weight to minority classes. "
            "`scheme`: balanced (inverse of class frequency) or uniform."
            )
    with p:
        st.write("Helps classifiers avoid bias toward majority class.")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )

    #Resampeling
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="Resampeling", 
            key="_PP_RS"
            )
    with v:
        _RS_M_options = ["SMOTE", "Random Over", "Random Under"]
        st.selectbox(
            label="Method", 
            options=_RS_M_options,
            key="_PP_RS_M"
            )
        if st.session_state._PP_RS_M == "SMOTE":
            st.number_input(
                label="k_neighbors", 
                min_value=1, 
                max_value=20,
                key="_PP_RS_SK"
                )
    with d:
        st.write(
            "Changes dataset composition by oversampling or undersampling. "
            "`SMOTE`: synthetic minority oversampling. "
            "`Random Over`: duplicate minority samples. "
            "`Random Under`: drop majority samples."
            )
    with p:
        st.write("Balances dataset distribution to improve recall of minority class.")

#################################
### SVM Kernel Approximations ###
#################################
with st.expander("SVM Kernel Approximations"):
    # Header
    t, v, d, p  = st.columns(4)
    with t:
        st.write("Preprocessing step")
    with v:
        st.write("Settings")
    with d:
        st.write("Description")
    with p:
        st.write("Purpose")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )
    t, v, d, p  = st.columns(4)
    with t:
        st.toggle(
            label="SVM Kernel Approximation", 
            key="_PP_SVM"
            )
    with v:
        _SVM_M_options = ["Nyström", "Random Fourier Features"]
        st.selectbox(
            label="Method", 
            options=_SVM_M_options,
            key="_PP_SVM_M"
            )
        if st.session_state._PP_SVM_M == "Nyström":
            st.number_input(
                label="n_components", 
                min_value=100, 
                max_value=50000,
                key="_PP_SVM_NNC"
                )
        if st.session_state._PP_SVM_M == "Random Fourier Features":
            st.number_input(
                label="n_components", 
                min_value=100, 
                max_value=50000,
                key="_PP_SVM_RFFNC")
    with d:
        st.write(
            "Approximates non-linear kernels for SVM using explicit features. "
            "`Nyström`: low-rank approximation. "
            "`Random Fourier`: approximates RBF kernels."
            )
    with p:
        st.write("Scales SVM to larger datasets by approximating kernel maps.")

#################################
### Time Features ###
#################################
with st.expander("Time Features"):
    # Header
    t, v, d, p  = st.columns(4)
    with t:
        st.write("Preprocessing step")
    with v:
        st.write("Settings")
    with d:
        st.write("Description")
    with p:
        st.write("Purpose")

    st.markdown(
        """<hr style="margin:0; padding:0">""",
        unsafe_allow_html=True
    )
    #Add time features
    t, v, d, p  = st.columns(4)
    with t:
        if st.session_state._HasTimeStamp == True:
            st.toggle(
                label="Add time features", 
                key="_TF"
                )
        else:
            st.write("Not avalible due to no timestamp collumn")
    with v:
        st.write("")
    with d:
        st.write("Extracts features like day, month, weekday, hour, season from timestamp columns.")
    with p:
        st.write("Turns temporal information into structured predictors.")


if st.session_state._SP_IsSplit == False:
    st.warning("To preprocess the data it needs to be splitted first")
else:
    
    if st.button(label="Apply seleced preprocessing", width="stretch"):

        X_train = st.session_state._SP_X_Train
        y_train = st.session_state._SP_y_Train
        X_validate = st.session_state._SP_X_Validate
        y_validate = st.session_state._SP_y_Validate
        X_test = st.session_state._SP_X_Test
        y_test = st.session_state._SP_y_Test

        config = {
            "_DD": st.session_state._PP_DD,
            "_MV": st.session_state._PP_MV,
            "_MV_N": st.session_state._PP_MV_N,
            "_MV_C": st.session_state._PP_MV_C,
            "_BR": st.session_state._PP_BR,
            "_BR_MF": st.session_state._PP_BR_MF,
            "_EN_GA": st.session_state._PP_EN_GA,
            "_EN_GA_O": st.session_state._PP_EN_GA_O,
            "_GBM": st.session_state._PP_GBM,
            "_GBM_M": st.session_state._PP_GBM_M,
            "_CE": st.session_state._PP_CE,
            "_TME": st.session_state._PP_TME,
            "_TME_NS": st.session_state._PP_TME_NS,
            "_TME_NOS": st.session_state._PP_TME_NOS,
            "_TME_GS": st.session_state._PP_TME_GS,
            "_DC": st.session_state._PP_DC,
            "_DC_V": st.session_state._PP_DC_V,
            "_DHC": st.session_state._PP_DHC,
            "_DHC_T": st.session_state._PP_DHC_T,
            "_FS_RF": st.session_state._PP_FS_RF,
            "_FS_RF_TN": st.session_state._PP_FS_RF_TN,
            "_FS_MI": st.session_state._PP_FS_MI,
            "_FS_MI_TK": st.session_state._PP_FS_MI_TK,
            "_FS_A": st.session_state._PP_FS_A,
            "_FS_A_TK": st.session_state._PP_FS_A_TK,
            "_PCA": st.session_state._PP_PCA,
            "_PCA_N": st.session_state._PP_PCA_N,
            "_SC": st.session_state._PP_SC,
            "_SC_S": st.session_state._PP_SC_S,
            "_SC_S_MIN": st.session_state._PP_SC_S_MIN,
            "_SC_S_MAX": st.session_state._PP_SC_S_MAX,
            "_L2N": st.session_state._PP_L2N,
            "_OC": st.session_state._PP_OC,
            "_OC_M": st.session_state._PP_OC_M,
            "_OC_WW": st.session_state._PP_OC_WW,
            "_OC_PL": st.session_state._PP_OC_PL,
            "_OC_PH": st.session_state._PP_OC_PH,
            "_LOG": st.session_state._PP_LOG,
            "_CW": st.session_state._PP_CW,
            "_CW_S": st.session_state._PP_CW_S,
            "_RS": st.session_state._PP_RS,
            "_RS_M": st.session_state._PP_RS_M,
            "_RS_SK": st.session_state._PP_RS_SK,
            "_SVM": st.session_state._PP_SVM,
            "_SVM_M": st.session_state._PP_SVM_M,
            "_SVM_NNC": st.session_state._PP_SVM_NNC,
            "_SVM_RFFNC": st.session_state._PP_SVM_RFFNC,
            "_TF": st.session_state._TF
        }

        ppres = pp.preprocessModel(X_train, y_train, X_validate, y_validate, X_test, y_test, config)

        st.session_state._PP_IsPP = True

        st.session_state._PP_X_Train = ppres["X_train"]
        st.session_state._PP_y_Train = ppres["y_train"]
        st.session_state._PP_X_Validate = ppres["X_val"]
        st.session_state._PP_y_Validate = ppres["y_val"]
        st.session_state._PP_X_Test = ppres["X_test"]
        st.session_state._PP_y_Test = ppres["y_test"]
        st.session_state._PP_ClassWeights = ppres["class_weights"]
        st.session_state._PP_SWTR = ppres["sample_weight_train"]
        st.session_state._PP_SWVA = ppres["sample_weight_val"]
        st.session_state._PP_SWTE = ppres["sample_weight_test"]
        st.session_state._PP_META = ppres["metadata"]
        st.session_state._PP_LE = ppres["metadata"]["label_encoding"]["mapping"]
        
        logger.save_log("Dataset preprocessed")
        st.success("Dataset preprocessed!")

    if st.session_state._PP_IsPP == True:
        st.header("Results preview")

        PP_X_train = st.session_state._PP_X_Train
        PP_y_train = st.session_state._PP_y_Train
        PP_X_validate = st.session_state._PP_X_Validate
        PP_y_validate = st.session_state._PP_y_Validate
        PP_X_test = st.session_state._PP_X_Test
        PP_y_test = st.session_state._PP_y_Test

        st.subheader("Train:")
        st.dataframe(pd.concat([PP_X_train, PP_y_train], axis=1).head(10))

        st.subheader("Validate")
        st.dataframe(pd.concat([PP_X_validate, PP_y_validate], axis=1).head(10))

        st.subheader("Test")
        st.dataframe(pd.concat([PP_X_test, PP_y_test], axis=1).head(10))

        st.write("Class Weights:")
        st.json(st.session_state._PP_ClassWeights)

        st.write("Metadata:")
        st.json(st.session_state._PP_META)
