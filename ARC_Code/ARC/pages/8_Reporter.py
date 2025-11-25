import streamlit as st

import functions.f09_Reporter as cPDF

from pathlib import Path
import io
import json


import streamlit as st
import json
import numpy as np
import pandas as pd


st.title("Reporter")

r, d, a = st.columns(3)
with r:
    st.header("PDF report")
    st.write("Include in the PDF report:")

    if st.session_state._DL_DataLoaded == False:
        st.session_state._R_DL = False
        st.session_state._R_SV = False

    if st.session_state._HasLabel == False:
        st.session_state._R_LV = False

    if st.session_state._SP_IsSplit == False:
        st.session_state._R_S = False

    if st.session_state._PP_IsPP == False:
        st.session_state._R_PP = False

    if st.session_state._TE_PTrained == False:
        st.session_state._R_TE = False

    if st.session_state._C_NPTrained == False:
        st.session_state._R_C = False

    st.session_state._R_DL = st.checkbox(
        label = "Data Loader",
        value = st.session_state._R_DL,
        key = "PDF_include_DL",
        disabled = not st.session_state._DL_DataLoaded
    )

    st.session_state._R_SV = st.checkbox(
        label = "Schema Validator",
        value = st.session_state._R_SV,
        key = "PDF_include_SV",
        disabled = not st.session_state._DL_DataLoaded
    )

    st.session_state._R_LV = st.checkbox(
        label = "Label Validator",
        value = st.session_state._R_LV,
        key = "PDF_include_LV",
        disabled = not st.session_state._HasLabel
    )

    st.session_state._R_S = st.checkbox(
        label = "Splitter",
        value = st.session_state._R_S,
        key = "PDF_include_S",
        disabled = not st.session_state._SP_IsSplit
    )

    st.session_state._R_TE = st.checkbox(
        label = "Trainer & Evaluator",
        value = st.session_state._R_TE,
        key = "PDF_include_TE",
        disabled = not st.session_state._TE_PTrained
    )

    st.session_state._R_C = st.checkbox(
        label = "Comparer",
        value = st.session_state._R_C,
        key = "PDF_include_C",
        disabled = not st.session_state._C_NPTrained
    )

    st.session_state._R_L = st.checkbox(
        label = "Log",
        value = st.session_state._R_L,
        key = "PDF_include_L"
    )
    pdf_bytes = cPDF.create_pdf_story()
    base = Path(st.session_state.get("_DataLoader_FileName", "report")).stem or "report"
    st.download_button(
        "Download report (PDF)",
        data=pdf_bytes,
        file_name=f"{base}_report.pdf",
        mime="application/pdf",
        width="stretch",
    )

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

with d:
    st.header("Datasets")
    st.write("Validated dataset")
    if st.session_state._DF is not None:
        st.download_button(label="Validated dataset as CSV",data=convert_df(st.session_state._DF),file_name=st.session_state._DL_Filename.split('.')[0]+".csv",mime="text/csv",width="stretch")
    else:
        st.button(label="Validated dataset as CSV", width="stretch", disabled=True)

    st.write("Splitted datasets")
    X_train = st.session_state._SP_X_Train
    y_train = st.session_state._SP_y_Train
    X_validate = st.session_state._SP_X_Validate
    y_validate = st.session_state._SP_y_Validate 
    X_test = st.session_state._SP_X_Test
    y_test = st.session_state._SP_y_Test

    if X_train is not None:
        st.download_button("X_train as CSV", convert_df(X_train), "X_train.csv", "text/csv", width="stretch")
    else:
        st.button(label="X_train as CSV", width="stretch", disabled=True)
    
    if y_train is not None:
        st.download_button("Download y_train", convert_df(y_train.to_frame()), "y_train.csv", "text/csv", width="stretch")
    else:
        st.button(label="y_train as CSV", width="stretch", disabled=True)
    
    if X_validate is not None:
        st.download_button("Download X_validate", convert_df(X_validate), "X_validate.csv", "text/csv", width="stretch")
    else:
        st.button(label="X_validate as CSV", width="stretch", disabled=True)
    
    if y_validate is not None:
        st.download_button("Download y_validate", convert_df(y_validate.to_frame()), "y_validate.csv", "text/csv", width="stretch")
    else:
        st.button(label="y_validate as CSV", width="stretch", disabled=True)
    
    if X_test is not None:
        st.download_button("Download X_test", convert_df(X_test), "X_test.csv", "text/csv", width="stretch")
    else:
        st.button(label="X_test as CSV", width="stretch", disabled=True)
    
    if y_test is not None:
        st.download_button("Download y_test", convert_df(y_test.to_frame()), "y_test.csv", "text/csv", width="stretch")
    else:
        st.button(label="y_test as CSV", width="stretch", disabled=True)
    
    st.write("Preprocessed datasets")
    PP_X_train = st.session_state._PP_X_Train
    PP_y_train = st.session_state._PP_y_Train
    PP_X_val = st.session_state._PP_X_Validate
    PP_y_val = st.session_state._PP_y_Validate
    PP_X_test = st.session_state._PP_X_Test
    PP_y_test = st.session_state._PP_y_Test

    if PP_X_train is not None:
        st.download_button("PP_X_train as CSV", convert_df(PP_X_train), "X_train.csv", "text/csv", width="stretch")
    else:
        st.button(label="PP_X_train as CSV", width="stretch", disabled=True)
    
    if PP_y_train is not None:
        st.download_button("PP_y_train", convert_df(PP_y_train.to_frame()), "y_train.csv", "text/csv", width="stretch")
    else:
        st.button(label="PP_y_train as CSV", width="stretch", disabled=True)
    
    if PP_X_val is not None:
        st.download_button("PP_X_validate", convert_df(PP_X_val), "X_validate.csv", "text/csv", width="stretch")
    else:
        st.button(label="PP_X_validate as CSV", width="stretch", disabled=True)
    
    if PP_y_val is not None:
        st.download_button("PP_y_validate", convert_df(PP_y_val.to_frame()), "y_validate.csv", "text/csv", width="stretch")
    else:
        st.button(label="PP_y_validate as CSV", width="stretch", disabled=True)
    
    if PP_X_test is not None:
        st.download_button("PP_X_test", convert_df(PP_X_test), "X_test.csv", "text/csv", width="stretch")
    else:
        st.button(label="PP_X_test as CSV", width="stretch", disabled=True)
    
    if PP_y_test is not None:
        st.download_button("PP_y_test", convert_df(PP_y_test.to_frame()), "y_test.csv", "text/csv", width="stretch")
    else:
        st.button(label="PP_y_test as CSV", width="stretch", disabled=True)

with a:
    st.header("Other")
    st.write("Log")

    out = []
    for e in st.session_state._LogData:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            ts, msg = e
            out.append(f"{ts}\n{msg}")
        else:
            out.append(str(e))  # fallback if your log isn't (ts, msg)
    txt = "\n\n".join(out)

    st.download_button(
        label="Download log as .txt",
        data=txt.encode("utf-8"),
        file_name="log.txt",
        mime="text/plain",
        width="stretch",
        disabled=not bool(st.session_state._LogData),
    )

    st.write("Preprocess meta data")

    def _json_ready(o):
        # numpy → native
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)):
            if np.isnan(o) or np.isinf(o): return None
            return float(o)
        if isinstance(o, (np.bool_,)): return bool(o)
        # pandas → lists/dicts
        if isinstance(o, pd.Series): return o.tolist()
        if isinstance(o, pd.DataFrame): return o.to_dict(orient="list")
        if isinstance(o, (pd.Index,)): return o.tolist()
        # sets/tuples → lists
        if isinstance(o, (set, tuple)): return list(o)
        # numpy arrays
        if hasattr(o, "tolist"): 
            try: return o.tolist()
            except Exception: pass
        # fallback
        return str(o)

    def to_json_bytes(obj) -> bytes:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=_json_ready).encode("utf-8")

    meta = st.session_state._PP_META

    if meta is None:
        st.button(label="Download PP metadata as JSON", width="stretch", disabled=True)
    else:
        st.download_button(
            "Download metadata as JSON",
            data=to_json_bytes(meta),
            file_name="preprocess_metadata.json",
            mime="application/json",
            width="stretch",
        )

    st.write("Preprocessed model")
    if st.session_state._TE_PTrained == True:
        results = st.session_state._TE_PRes
        try:
            buf = io.BytesIO()
            import pickle
            pickle.dump(results["model"], buf, protocol=pickle.HIGHEST_PROTOCOL)
            st.download_button(label="Download PP model as PKL", data=buf.getvalue(), file_name="model.pkl",
                            mime="application/octet-stream", key=f"dl-model", width="stretch")
        except Exception as e:
            st.info(f"Model download unavailable: {e}")

        try:
            copy = dict(results)
            copy["model"] = str(type(results["model"]))
            st.download_button(label="Download PP metrics as JSON", data=json.dumps(copy, default=float, indent=2),
                            file_name="metrics.json", mime="application/json",
                            key="dl-metrics", width="stretch")
        except Exception as e:
            st.info(f"Metrics download unavailable: {e}")
    else:
        st.button(label="Download PP model as PKL", width="stretch", disabled=True)
        st.button(label="Download PP metrics as JSON", width="stretch", disabled=True)