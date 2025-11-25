import streamlit as st

import functions.f06_Trainer_Evaluator as te
import functions.f00_Sidebar as sidebar
import functions.f00_Logger as logger

import numpy as np
sidebar.sidebar()

st.title("Comparison")
if st.session_state._DL_DataLoaded == False:
    st.error("No dataset loaded. Please load a dataset first.")
elif st.session_state._PP_IsPP == False:
    st.write("Dataset is not preprocessed. Please preprocess the data first.")
elif st.session_state._TE_PTrained == False:
    st.write("No model trained on preprocessed data. Please train a model first.")
else:
    PP_X_Train = st.session_state._PP_X_Train
    PP_y_Train = st.session_state._PP_y_Train
    PP_X_Validate = st.session_state._PP_X_Validate
    PP_y_Validate = st.session_state._PP_y_Validate
    PP_X_Test = st.session_state._PP_X_Test
    PP_y_Test = st.session_state._PP_y_Test
    PP_sw_tr = st.session_state._PP_SWTR
    PP_sw_va = st.session_state._PP_SWVA
    PP_sw_te = st.session_state._PP_SWTE
    PP_meta = st.session_state._PP_META

    NP_X_Train = st.session_state._SP_X_Train
    NP_y_Train = st.session_state._SP_y_Train
    NP_X_Validate = st.session_state._SP_X_Validate
    NP_y_Validate = st.session_state._SP_y_Validate
    NP_X_Test = st.session_state._SP_X_Test
    NP_y_Test = st.session_state._SP_y_Test
    NP_sw_tr = None
    NP_sw_va = None
    NP_sw_te = None
    NP_meta = None

    #Xtr_P, ytr_P, Xva_P, yva_P, Xte_P, yte_P, sw_tr_P, sw_va_P, sw_te_P, used_enc_P, meta_P = te.get_datasets_from_session(use_preprocessed=True)
    #Xtr_NP, ytr_NP, Xva_NP, yva_NP, Xte_NP, yte_NP, sw_tr_NP, sw_va_NP, sw_te_NP, used_enc_NP, meta_NP = te.get_datasets_from_session(use_preprocessed=False)

    if st.button(label="Run training on raw data", width="stretch"):
        results_NP = te.train_eval_orchestrator(
            model=st.session_state._TE_Model,
            use_preprocessed=False,
            gbm_engine=st.session_state._GBM_Engine,
            X_train=NP_X_Train, y_train=NP_y_Train,
            X_val=NP_X_Validate, y_val=NP_y_Validate,
            X_test=NP_X_Test, y_test=NP_y_Test,
            sample_weight_train=NP_sw_tr,
            sample_weight_val=NP_sw_va,
            sample_weight_test=NP_sw_te,
            meta=NP_meta
        )
        st.session_state._C_NPTrained = True
        st.session_state._C_NPRes = results_NP
        logger.save_log("Raw model trained")
        st.rerun()

    if st.session_state._C_NPTrained == True:
        results_NP = st.session_state._C_NPRes
        results_P = st.session_state._TE_PRes

        classes_P = getattr(results_P["model"], "classes_", None)
        class_names_P = [str(c) for c in classes_P] if classes_P is not None else None

        classes_NP = getattr(results_NP["model"], "classes_", None)
        class_names_NP = [str(c) for c in classes_NP] if classes_NP is not None else None
        
        st.markdown(
            """
            <style>
            div[data-testid="stMetric"] {
                text-align: center;
                align-items: center;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.caption("RAW = unprocessed | PRE = preprocessed")

        te.render_compare_general(results_NP, results_P)

        te.render_compare_metrics(results_NP, results_P)


        mapping = st.session_state._PP_LE
        class_names_P = [name for name, _id in sorted(mapping.items(), key=lambda x: x[1])]

        st.subheader("Confusion Matrix")

        tra, val, tes = st.tabs(["Train", "Validation", "Test"])
        with tra:
            c1, c2 = st.columns(2)
            with c1:
                st.write("RAW")
                if results_NP.get("train")is not None:
                    cm = np.array(results_NP["train"]["metrics"]["confusion_matrix"])
                    st.plotly_chart(te.plot_confusion(cm, class_names_NP), width="stretch", key="cm_raw_train")
            with c2:
                st.write("PRE")
                if results_P.get("train")is not None:
                    cm = np.array(results_P["train"]["metrics"]["confusion_matrix"])
                    st.plotly_chart(te.plot_confusion(cm, class_names_P), width="stretch", key="cm_pp_train")
        with val:
            c1, c2 = st.columns(2)
            with c1:
                st.write("RAW")
                if results_NP.get("validation")is not None:
                    cm = np.array(results_NP["validation"]["metrics"]["confusion_matrix"])
                    st.plotly_chart(te.plot_confusion(cm, class_names_NP), width="stretch", key="cm_raw_val")
            with c2:
                st.write("PRE")
                if results_P.get("validation")is not None:
                    cm = np.array(results_P["validation"]["metrics"]["confusion_matrix"])
                    st.plotly_chart(te.plot_confusion(cm, class_names_P), width="stretch", key="cm_pp_val")
        with tes:
            c1, c2 = st.columns(2)
            with c1:
                st.write("RAW")
                if results_NP.get("test")is not None:
                    cm = np.array(results_NP["test"]["metrics"]["confusion_matrix"])
                    st.plotly_chart(te.plot_confusion(cm, class_names_NP), width="stretch",key="cm_raw_test")
            with c2:
                st.write("PRE")
                if results_P.get("test")is not None:
                    cm = np.array(results_P["test"]["metrics"]["confusion_matrix"])
                    st.plotly_chart(te.plot_confusion(cm, class_names_P), width="stretch", key="cm_pp_test")
        
        if hasattr(results_P["model"], "predict_proba"):
            st.subheader("ROC")
            tra, val, tes = st.tabs(["Train", "Validation", "Test"])
            with tra:
                c1, c2 = st.columns(2)
                with c1:
                    st.write("RAW")
                    if results_NP.get("train")is not None:
                        try:
                            roc_fig = te.plot_roc(results_NP["model"], NP_X_Train, NP_X_Train)
                            if roc_fig is not None:
                                st.plotly_chart(roc_fig, width="stretch", key="roc_raw_train")
                        except Exception:
                            pass
                with c2:
                    st.write("PRE")
                    if results_P.get("train")is not None:
                        try:
                            roc_fig = te.plot_roc(results_P["model"], PP_X_Train, PP_X_Train, class_names_P)
                            if roc_fig is not None:
                                st.plotly_chart(roc_fig, width="stretch", key="roc_pp_train")
                        except Exception:
                            pass
            with val:
                c1, c2 = st.columns(2)
                with c1:
                    st.write("RAW")
                    if results_NP.get("validation")is not None:
                        try:
                            roc_fig = te.plot_roc(results_NP["model"], NP_X_Validate, NP_y_Validate)
                            if roc_fig is not None:
                                st.plotly_chart(roc_fig, width="stretch", key="roc_raw_val")
                        except Exception:
                            pass
                with c2:
                    st.write("PRE")
                    if results_P.get("validation")is not None:
                        try:
                            roc_fig = te.plot_roc(results_P["model"], PP_X_Validate, PP_y_Validate, class_names_P)
                            if roc_fig is not None:
                                st.plotly_chart(roc_fig, width="stretch", key="roc_pp_val")
                        except Exception:
                            pass
            with tes:
                c1, c2 = st.columns(2)
                with c1:
                    st.write("RAW")
                    if results_NP.get("test")is not None:
                        try:
                            roc_fig = te.plot_roc(results_NP["model"], NP_X_Test, NP_y_Test)
                            if roc_fig is not None:
                                st.plotly_chart(roc_fig, width="stretch", key="roc_raw_test")
                        except Exception:
                            pass
                with c2:
                    st.write("PRE")
                    if results_P.get("test")is not None:
                        try:
                            roc_fig = te.plot_roc(results_P["model"], PP_X_Test, PP_y_Test, class_names_P)
                            if roc_fig is not None:
                                st.plotly_chart(roc_fig, width="stretch", key="roc_pp_test")
                        except Exception:
                            pass
            

            st.subheader("Precision-Recall")
            tra, val, tes = st.tabs(["Train", "Validation", "Test"])
            with tra:
                c1, c2 = st.columns(2)
                with c1:
                    st.write("RAW")
                    if results_NP.get("train")is not None:
                        try:
                            pr_fig = te.plot_pr(results_NP["model"], NP_X_Train, NP_X_Train)
                            if pr_fig is not None:
                                st.plotly_chart(pr_fig, width="stretch", key="pr_raw_train")
                        except Exception:
                            pass
                with c2:
                    st.write("PRE")
                    if results_P.get("train")is not None:
                        try:
                            pr_fig = te.plot_pr(results_P["model"], PP_X_Train, PP_X_Train, class_names_P)
                            if pr_fig is not None:
                                st.plotly_chart(pr_fig, width="stretch", key="pr_pp_train")
                        except Exception:
                            pass
            with val:
                c1, c2 = st.columns(2)
                with c1:
                    st.write("RAW")
                    if results_NP.get("validation")is not None:
                        try:
                            pr_fig = te.plot_pr(results_NP["model"], NP_X_Validate, NP_y_Validate)
                            if pr_fig is not None:
                                st.plotly_chart(pr_fig, width="stretch", key="pr_raw_val")
                        except Exception:
                            pass
                with c2:
                    st.write("PRE")
                    if results_P.get("validation")is not None:
                        try:
                            pr_fig = te.plot_pr(results_P["model"], PP_X_Validate, PP_y_Validate, class_names_P)
                            if pr_fig is not None:
                                st.plotly_chart(pr_fig, width="stretch", key="pr_pp_val")
                        except Exception:
                            pass
            with tes:
                c1, c2 = st.columns(2)
                with c1:
                    st.write("RAW")
                    if results_NP.get("test")is not None:
                        try:
                            pr_fig = te.plot_pr(results_NP["model"], NP_X_Test, NP_y_Test)
                            if pr_fig is not None:
                                st.plotly_chart(pr_fig, width="stretch", key="pr_raw_test")
                        except Exception:
                            pass
                with c2:
                    st.write("PRE")
                    if results_P.get("test")is not None:
                        try:
                            pr_fig = te.plot_pr(results_P["model"], PP_X_Test, PP_y_Test, class_names_P)
                            if pr_fig is not None:
                                st.plotly_chart(pr_fig, width="stretch")
                        except Exception:
                            pass
        
        feat_names_NP = getattr(NP_X_Train, "columns", None)
        fig_imp_NP = te.feature_importance_fig(results_NP["model"], feature_names=feat_names_NP, top_k=25)

        feat_names_P = getattr(PP_X_Train, "columns", None)
        fig_imp_P = te.feature_importance_fig(results_P["model"], feature_names=feat_names_P, top_k=25)
        
        if fig_imp_NP is not None or fig_imp_P is not None:
            st.subheader("Feature Importances (top 25)")
            c1, c2 = st.columns(2)
            with c1:
                st.write("RAW")
                st.plotly_chart(fig_imp_NP, width="stretch", key="fi_raw")
            with c2:
                st.write("PRE")
                st.plotly_chart(fig_imp_P, width="stretch", key="fi_pp")