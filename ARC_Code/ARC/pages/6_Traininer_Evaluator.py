from __future__ import annotations

import streamlit as st

import functions.f06_Trainer_Evaluator as te
import functions.f00_Logger as logger
import functions.f00_Sidebar as sidebar

import json
import io
import numpy as np
import pandas as pd

sidebar.sidebar()

st.title("Train & Evaluate")

if st.session_state._DL_DataLoaded == False:
    st.error("No dataset loaded. Please load a dataset first.")
elif st.session_state._PP_IsPP == False:
    st.write("Dataset is not preprocessed. Please preprocess the data first.")
else:

    Models = [
        "Random Forest (RF)",
        "Gradient Boosting (XGBoost / LightGBM / CatBoost)",
        "Support Vector Machine (SVM)",
        "Neural Network (MLP)",
        "Logistic Regression (LR)"
    ]

    model = st.selectbox("Model:", Models, index=0)

    gbm_engine = None
    if model.startswith("Gradient Boosting"):
        gbm = te.checkGBM_engine()
        engines = []
        if gbm["_HAS_LGB"]: engines.append("LightGBM")
        if gbm["_HAS_XGB"]: engines.append("XGBoost")
        if gbm["_HAS_CAT"]: engines.append("CatBoost")
        if not engines:
            st.warning("No GBM library available. Falling back to Random Forest.")
            model = "Random Forest (RF)"
        else:
            gbm_engine = st.selectbox("GBM Engine", engines, index=0)

    X_Train = st.session_state._PP_X_Train
    y_Train = st.session_state._PP_y_Train
    X_Validate = st.session_state._PP_X_Validate
    y_Validate = st.session_state._PP_y_Validate
    X_Test = st.session_state._PP_X_Test
    y_Test = st.session_state._PP_y_Test
    sw_tr = st.session_state._PP_SWTR
    sw_va = st.session_state._PP_SWVA
    sw_te = st.session_state._PP_SWTE
    meta = st.session_state._PP_META
        
    if st.button(label="Run training", width="stretch"):
        st.session_state._GBM_Engine = gbm_engine
        if X_Train is None or y_Train is None:
            st.error("Training split not found. Please ensure your data is split (and preprocessed if selected).")
            st.stop()

        results_P = te.train_eval_orchestrator(
            model=model,
            use_preprocessed=True,
            gbm_engine=gbm_engine,
            X_train=X_Train, y_train=y_Train,
            X_val=X_Validate, y_val=y_Validate,
            X_test=X_Test, y_test=y_Test,
            sample_weight_train=sw_tr,
            sample_weight_val=sw_va,
            sample_weight_test=sw_te,
            meta=meta
        )

        st.session_state._TE_Model = model
        if model.startswith("Gradient Boosting"):
            st.session_state._GBM_Engine = gbm_engine
        st.session_state._TE_PTrained = True
        st.session_state._TE_PRes = results_P
        logger.save_log("Model trained")
        st.rerun()

    if st.session_state._TE_PTrained == True:
        results = st.session_state._TE_PRes

        st.write(results)

        st.header("Results of training with preprocessed data")
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
        st.subheader("General")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Train time (s)", f"{results['train_time_sec']:.3f}")
        with c2:
            st.metric("Model size", te.fmt_bytes(results["model_size_bytes"]))
        with c3:
            st.metric("n-features", results.get("n_features", "n/a"))
            
        st.subheader("Metrics")
        rows = []
        for split in ["train", "validation", "test"]:
            r = results.get(split)
            if r is None:
                continue
            m = r["metrics"]
            rows.append({
                "Split": split,
                "Accuracy": m["accuracy"],
                "F1 (macro)": m["f1_macro"],
                "Precision (macro)": m["precision_macro"],
                "Recall (macro)": m["recall_macro"],
                "ROC-AUC": m["roc_auc"],
                "Inference time [s]": r["inference_time_sec"]
            })
        st.write(pd.DataFrame(rows).set_index("Split"))

        st.subheader("Confusion Matrix")
        classes = getattr(results["model"], "classes_", None)
        class_names = [str(c) for c in classes] if classes is not None else None
        #st.write(st.session_state._PP_LE)

        mapping = st.session_state._PP_LE
        class_names = [name for name, _id in sorted(mapping.items(), key=lambda x: x[1])]

        t1, t2, t3 = st.tabs(["Train", "Validation", "Test"])
        with t1:
            st.write("Train")
            if results.get("train")is not None:
                cm = np.array(results["train"]["metrics"]["confusion_matrix"])
                st.plotly_chart(te.plot_confusion(cm, class_names), width="stretch")
        with t2:
            st.write("Validation")
            if results.get("validation")is not None:
                cm = np.array(results["validation"]["metrics"]["confusion_matrix"])
                st.plotly_chart(te.plot_confusion(cm, class_names), width="stretch")
        with t3:
            st.write("Test")
            if results.get("test")is not None:
                cm = np.array(results["test"]["metrics"]["confusion_matrix"])
                st.plotly_chart(te.plot_confusion(cm, class_names), width="stretch")
        
        if hasattr(results["model"], "predict_proba"):
            st.subheader("ROC")
            t1, t2, t3 = st.tabs(["Train", "Validation", "Test"])
            with t1:
                st.write("Train")
                if results.get("train")is not None:
                    try:
                        roc_fig = te.plot_roc(results["model"], X_Train, y_Train, class_names)
                        if roc_fig is not None:
                            st.plotly_chart(roc_fig, width="stretch")
                    except Exception:
                        pass
            with t2:
                st.write("Validation")
                if results.get("validation")is not None:
                    try:
                        roc_fig = te.plot_roc(results["model"], X_Validate, y_Validate, class_names)
                        if roc_fig is not None:
                            st.plotly_chart(roc_fig, width="stretch")
                    except Exception:
                        pass
            with t3:
                st.write("Test")
                if results.get("test")is not None:
                    try:
                        roc_fig = te.plot_roc(results["model"], X_Test, y_Test, class_names)
                        if roc_fig is not None:
                            st.plotly_chart(roc_fig, width="stretch")
                    except Exception:
                        pass

            st.subheader("Precision-Recall")
            t1, t2, t3 = st.tabs(["Train", "Validation", "Test"])
            with t1:
                st.write("Train")
                if results.get("train")is not None:
                    try:
                        pr_fig = te.plot_pr(results["model"], X_Train, y_Train, class_names)
                        if pr_fig is not None:
                            st.plotly_chart(pr_fig, width="stretch")
                    except Exception:
                        pass
            with t2:
                st.write("Validation")
                if results.get("validation")is not None:
                    try:
                        pr_fig = te.plot_pr(results["model"], X_Validate, y_Validate, class_names)
                        if pr_fig is not None:
                            st.plotly_chart(pr_fig, width="stretch")
                    except Exception:
                        pass
            with t3:
                st.write("Test")
                if results.get("test")is not None:
                    try:
                        pr_fig = te.plot_pr(results["model"], X_Test, y_Test, class_names)
                        if pr_fig is not None:
                            st.plotly_chart(pr_fig, width="stretch")
                    except Exception:
                        pass
        
        feat_names = getattr(X_Train, "columns", None)
        fig_imp = te.feature_importance_fig(results["model"], feature_names=feat_names, top_k=25)
        if fig_imp is not None:
            st.subheader("Feature Importances (top 25)")
            st.plotly_chart(fig_imp, width="stretch")
        
        try:
            buf = io.BytesIO()
            import pickle
            pickle.dump(results["model"], buf, protocol=pickle.HIGHEST_PROTOCOL)
            st.download_button(label="Download model (.pkl)", data=buf.getvalue(), file_name="model.pkl",
                            mime="application/octet-stream", key=f"dl-model", width="stretch")
        except Exception as e:
            st.info(f"Model download unavailable: {e}")

        try:
            copy = dict(results)
            copy["model"] = str(type(results["model"]))
            st.download_button(label="Download metrics (.json)", data=json.dumps(copy, default=float, indent=2),
                            file_name="metrics.json", mime="application/json",
                            key="dl-metrics", width="stretch")
        except Exception as e:
            st.info(f"Metrics download unavailable: {e}")