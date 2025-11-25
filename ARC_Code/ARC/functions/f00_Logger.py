import streamlit as st

from datetime import datetime

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# write log entry with timestamp
def save_log(message: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state._LogData.append((ts, message))

# clear log entries
def clear_log():
    st.session_state._LogData.clear()

# show latest 10 logs in sidebar
def show_log():
    st.sidebar.subheader("Log:")

    log = st.session_state._LogData

    # no logs
    if not log:
        st.sidebar.info("No log entries yet.")
        return

    # last 10 logs reversed
    tail = log[-10:][::-1]
    rendered = "\n".join(f"{t}\n{m}" for t, m in tail)
    st.sidebar.code(rendered)

    # clear log button
    if st.sidebar.button("Clear log"):
        clear_log()
        st.rerun()