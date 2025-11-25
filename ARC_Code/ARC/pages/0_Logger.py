import streamlit as st

import functions.f00_Sidebar as sidebar
import functions.f00_Logger as logger

# generate sidebar
sidebar.sidebar()

# title
st.title("Logger")

# add custom log message
with st.form("add_log_form", clear_on_submit=True):
    msg = st.text_area("Add a custom log message", height=80)
    submitted = st.form_submit_button("Append to log", width="stretch")
    if submitted and msg.strip():
        logger.save_log(msg.strip())
        st.rerun()

# clear log button
if st.button("Clear log", type="secondary", width="stretch"):
        logger.clear_log()
        st.rerun()

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
st.divider()

# show log entries
st.subheader("Log entries:")
log = st.session_state._LogData
if not log:
    st.info("No log entries yet.")
else:
    tail = log[::-1]
    rendered = "\n\n".join(f"{t}\n{m}" for t, m in tail)
    st.code(rendered)