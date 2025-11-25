import streamlit as st

def init():
    ##############
    ### LOGGER ###
    ##############
    if "_LogData" not in st.session_state:
        st.session_state._LogData = []
    

    ##########
    ### DF ###
    ##########

    if "_DF" not in st.session_state:
        st.session_state._DF = None

    if "_HasTimeStamp" not in st.session_state:
        st.session_state._HasTimeStamp = False
    if "_TimeStampCol" not in st.session_state:
        st.session_state._TimeStampCol = ""

    if "_HasLabel" not in st.session_state:
        st.session_state._HasLabel = False
    if "_LabelCol" not in st.session_state:
        st.session_state._LabelCol = ""


    ###################
    ###  DATALOADER ###
    ###################

    if "_DL_Start" not in st.session_state:
        st.session_state._DL_Start = False
    if "_DL_DataLoaded" not in st.session_state:
        st.session_state._DL_DataLoaded = False

    if "_DL_Mode" not in st.session_state:
        st.session_state._DL_Mode = "single"

    if "_DL_Filename" not in st.session_state:
        st.session_state._DL_Filename = "N/A"
    if "_DL_UploadedFiles" not in st.session_state:
        st.session_state._DL_UploadedFiles = []
    

    #######################
    ### SCHEMAVALIDATOR ###
    #######################

    if "_SV_ReqCol" not in st.session_state:
        st.session_state._SV_ReqCol = "timestamp, ip_src, ip_dst, port_src, port_dst, ip_proto, frame_len, label"
    if "_SV_RenameColM" not in st.session_state:
        st.session_state._SV_RenameColM = """\
src_ip = ip_src
source_ip = ip_src
sourceip = ip_src

dst_ip = ip_dst
destination_ip = ip_dst
destinationip = ip_dst

sourceport = port_src
destinationport = port_dst

proto = ip_proto
protocol = ip_proto

duration = frame_len

frame.time = timestamp
date = timestamp"""
    if "_SV_RenameColMMap" not in st.session_state:
        st.session_state._SV_RenameColMMap = {}
    if "_SV_TDelta" not in st.session_state:
        st.session_state._SV_TDelta = False
    if "_SV_ColToDrop" not in st.session_state:
        st.session_state._SV_ColToDrop = False
    
    # applied changes
    if "_SV_NormCol" not in st.session_state:
        st.session_state._SV_NormCol = False
    if "_SV_RenameCol" not in st.session_state:
        st.session_state._SV_RenameCol = False
    if "_SV_RenameCols" not in st.session_state:
        st.session_state._SV_RenameCols = []
    if "_SV_SortByT" not in st.session_state:
        st.session_state._SV_SortByT = False
    if "_SV_DeltaT" not in st.session_state:
        st.session_state._SV_TDelta = False
    if "_SV_DropCol" not in st.session_state:
        st.session_state._SV_DropCol = False
    if "_SV_DropedCols" not in st.session_state:
        st.session_state._SV_DropedCols = []
    if "_SV_DropDup" not in st.session_state:
        st.session_state._SV_DropDup = False
    if "_SV_DropedDupCount" not in st.session_state:
        st.session_state._SV_DropedDupCount = 0
    

    ######################
    ### LABELVALIDATOR ###
    ######################
    if "_LV_InputRareClasses" not in st.session_state:
        st.session_state._LV_InputRareClasses = 1
    if "_LV_InputDominantClasses" not in st.session_state:
        st.session_state._LV_InputDominantClasses = 90
    if "_LV_TDNumBins" not in st.session_state:
        st.session_state._LV_TDNumBins = 15
    if "_LV_TDTimeline" not in st.session_state:
        st.session_state._LV_TDTimeline = True
    if "_LV_TDTime" not in st.session_state:
        st.session_state._LV_TDTime = True
    if "_LV_TDRecords" not in st.session_state:
        st.session_state._LV_TDRecords = True
    
    # applied changes
    if "_LV_RenameLabel" not in st.session_state:
        st.session_state._LV_RenameLabel = False
    if "_LV_RenamedLabels" not in st.session_state:
        st.session_state._LV_RenamedLabels = []
    if "_LV_RenameM" not in st.session_state:
        st.session_state._LV_RenameM = """DOS = DoS"""
    if "_LV_RenameMMap" not in st.session_state:
        st.session_state._LV_RenameMMap = {}

    #############
    ### Split ###
    #############

    if "_SP_IsSplit" not in st.session_state:
        st.session_state._SP_IsSplit = False
    if "_SP_SplitMethod" not in st.session_state:
        st.session_state._SP_SplitMethod = 20
    
    if "_SP_RandomState" not in st.session_state:
        st.session_state._SP_RandomState = 42
    if "_SP_GapRatio" not in st.session_state:
        st.session_state._SP_GapRatio = 0

    if "_SP_TestSize" not in st.session_state:
        st.session_state._SP_TestSize = 20
    if "_SP_ValSize" not in st.session_state:
        st.session_state._SP_ValSize = 20

    if "_SP_X_Train" not in st.session_state:
        st.session_state._SP_X_Train = None
    if "_SP_y_Train" not in st.session_state:
        st.session_state._SP_y_Train = None
    if "_SP_X_Validate" not in st.session_state:
        st.session_state._SP_X_Validate = None
    if "_SP_y_Validate" not in st.session_state:
        st.session_state._SP_y_Validate = None
    if "_SP_X_Test" not in st.session_state:
        st.session_state._SP_X_Test = None
    if "_SP_y_Test" not in st.session_state:
        st.session_state._SP_y_Test = None
    
    ##################
    ### Preprocess ###
    ##################
    if "_PP_IsPP" not in st.session_state:
        st.session_state._PP_IsPP = False

    ### Model
    if "_PP_Model" not in st.session_state:
        st.session_state._PP_Model = "Random Forest (RF)"
    ## REMOVE LATER ##################################################################################################
    if "_modelChanged" not in st.session_state:
        st.session_state._modelChanged = False
    
    ### Cleaning
    # Drop Duplicates
    if "_PP_DD" not in st.session_state:
        st.session_state._PP_DD = False
    # Missing Values
    if "_PP_MV" not in st.session_state:
        st.session_state._PP_MV = False
    if "_PP_MV_N" not in st.session_state:
        st.session_state._PP_MV_N = "Medial"
    if "_PP_MV_C" not in st.session_state:
        st.session_state._PP_MV_C = "Most Frequent"

    ### Encoding
    # Bucket Rare
    if "_PP_BR" not in st.session_state:
        st.session_state._PP_BR = False
    if "_PP_BR_MF" not in st.session_state:
        st.session_state._PP_BR_MF = 20
    # G/A Encoding
    if "_PP_EN_GA" not in st.session_state:
        st.session_state._PP_EN_GA = False
    if "_PP_EN_GA_O" not in st.session_state:
        st.session_state._PP_EN_GA_O = "One Hot Encoding"
    # GBM Native Categorical Prep
    if "_PP_GBM" not in st.session_state:
        st.session_state._PP_GBM = False
    if "_PP_GBM_M" not in st.session_state:
        st.session_state._PP_GBM_M = "CatBoost"
    # Count Encoding
    if "_PP_CE" not in st.session_state:
        st.session_state._PP_CE = False
    # Target mean encoding
    if "_PP_TME" not in st.session_state:
        st.session_state._PP_TME = False
    if "_PP_TME_NS" not in st.session_state:
        st.session_state._PP_TME_NS = 5
    if "_PP_TME_NOS" not in st.session_state:
        st.session_state._PP_TME_NOS = 0.0
    if "_PP_TME_GS" not in st.session_state:
        st.session_state._PP_TME_GS = 10.0
    
    ###Feature Filtering / Selection
    # Drop Consistant Values
    if "_PP_DC" not in st.session_state:
        st.session_state._PP_DC = False
    if "_PP_DC_V" not in st.session_state:
        st.session_state._PP_DC_V = 0.0
    #Drop highly correlated numerics
    if "_PP_DHC" not in st.session_state:
        st.session_state._PP_DHC = False
    if "_PP_DHC_T" not in st.session_state:
        st.session_state._PP_DHC_T = 0.98
    #Feature selection by RF
    if "_PP_FS_RF" not in st.session_state:
        st.session_state._PP_FS_RF = False
    if "_PP_FS_RF_TN" not in st.session_state:
        st.session_state._PP_FS_RF_TN = 100
    #Feature selection by Mutual Info
    if "_PP_FS_MI" not in st.session_state:
        st.session_state._PP_FS_MI = False
    if "_PP_FS_MI_TK" not in st.session_state:
        st.session_state._PP_FS_MI_TK = 200
    #Feature selection by ANOVA F
    if "_PP_FS_A" not in st.session_state:
        st.session_state._PP_FS_A = False
    if "_PP_FS_A_TK" not in st.session_state:
        st.session_state._PP_FS_A_TK = 200
    #Feature selection by ANOVA F
    if "_PP_PCA" not in st.session_state:
        st.session_state._PP_PCA = False
    if "_PP_PCA_N" not in st.session_state:
        st.session_state._PP_PCA_N = 0.95
    
    ### Scaling & Normalization
    # Scaling
    if "_PP_SC" not in st.session_state:
        st.session_state._PP_SC = False
    if "_PP_SC_S" not in st.session_state:
        st.session_state._PP_SC_S = "Z-Score"
    if "_PP_SC_S_MIN" not in st.session_state:
        st.session_state._PP_SC_S_MIN = 0.0
    if "_PP_SC_S_MAX" not in st.session_state:
        st.session_state._PP_SC_S_MAX = 1.0
    # Row-wise L2 normalization
    if "_PP_L2N" not in st.session_state:
        st.session_state._PP_L2N = False

    ### Outliers & Transforms
    # Outliers Clipping
    if "_PP_OC" not in st.session_state:
        st.session_state._PP_OC = False
    if "_PP_OC_M" not in st.session_state:
        st.session_state._PP_OC_M = "IQR"
    if "_PP_OC_WW" not in st.session_state:
        st.session_state._PP_OC_WW = 3.0
    if "_PP_OC_PL" not in st.session_state:
        st.session_state._PP_OC_PL = 0.5
    if "_PP_OC_PH" not in st.session_state:
        st.session_state._PP_OC_PH = 99.5
    # log1p transform
    if "_PP_LOG" not in st.session_state:
        st.session_state._PP_LOG = False
    
    ### Class Imbalance
    # Class weights
    if "_PP_CW" not in st.session_state:
        st.session_state._PP_CW = False
    if "_PP_CW_S" not in st.session_state:
        st.session_state._PP_CW_S = "balanced"
    #Resampeling
    if "_PP_RS" not in st.session_state:
        st.session_state._PP_RS = False
    if "_PP_RS_M" not in st.session_state:
        st.session_state._PP_RS_M = "SMOTE"
    if "_PP_RS_SK" not in st.session_state:
        st.session_state._PP_RS_SK = 5
    
    ### SVM Kernel Approximations
    if "_PP_SVM" not in st.session_state:
        st.session_state._PP_SVM = False
    if "_PP_SVM_M" not in st.session_state:
        st.session_state._PP_SVM_M = "Nyström"
    if "_PP_SVM_NNC" not in st.session_state:
        st.session_state._PP_SVM_NNC = 2000
    if "_PP_SVM_RFFNC" not in st.session_state:
        st.session_state._PP_SVM_RFFNC = 2000

    ### Time features
    if "_PP_TF" not in st.session_state:
        st.session_state._PP_TF = False

    ### Result
    # train
    if "_PP_X_Train" not in st.session_state:
        st.session_state._PP_X_Train = None
    if "_PP_y_Train" not in st.session_state:
        st.session_state._PP_y_Train = None
    # validate
    if "_PP_X_Validate" not in st.session_state:
        st.session_state._PP_X_Validate = None
    if "_PP_y_Validate" not in st.session_state:
        st.session_state._PP_y_Validate = None
    # test
    if "_PP_X_Test" not in st.session_state:
        st.session_state._PP_X_Test = None
    if "_PP_y_Test" not in st.session_state:
        st.session_state._PP_y_Test = None
    
    # insights
    if "_PP_META" not in st.session_state:
        st.session_state._PP_META = None
    if "_PP_ClassWeights" not in st.session_state:
        st.session_state._PP_ClassWeights = None
    if "_PP_scale_pos_weight" not in st.session_state:
        st.session_state._PP_scale_pos_weight = None
    if "_PP_sample_weight_train" not in st.session_state:
        st.session_state._PP_sample_weight_train = None
    if "_PP_sample_weight_test" not in st.session_state:
        st.session_state._PP_sample_weight_test = None
    if "_PP_LE" not in st.session_state:
        st.session_state._PP_LE = None
    
    #############################
    ### Training & Evaluation ###
    #############################
    if "_TE_PTrained" not in st.session_state:
        st.session_state._TE_PTrained = False
    if "_C_NPTrained" not in st.session_state:
        st.session_state._C_NPTrained = False

    if "_Model" not in st.session_state:
        st.session_state._Model = None
    if "_GBM_Engine" not in st.session_state:
        st.session_state._GBM_Engine = None

    if "_TE_PRes" not in st.session_state:
        st.session_state._TE_PRes = None
    if "_C_NPRes" not in st.session_state:
        st.session_state._C_NPRes = None
    
    ################
    ### Reporter ###
    ################

    if "_R_DL" not in st.session_state:
        st.session_state._R_DL = None
    if "_R_SV" not in st.session_state:
        st.session_state._R_SV = None
    if "_R_LV" not in st.session_state:
        st.session_state._R_LV = None
    if "_R_S" not in st.session_state:
        st.session_state._R_S = None
    if "_R_PP" not in st.session_state:
        st.session_state._R_PP = None
    if "_R_TE" not in st.session_state:
        st.session_state._R_TE = None
    if "_R_C" not in st.session_state:
        st.session_state._R_C = None
    if "_R_L" not in st.session_state:
        st.session_state._R_L = None
    