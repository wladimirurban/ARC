import functions.f00_Sidebar as sidebar
import functions.f00_Logger as logger
import functions.f01_DataLoader as dl
import functions.f06_Trainer_Evaluator as te

import streamlit as st

sidebar.sidebar()
# title
st.title("Data Loader")
st.header("Upload and load datasets for analysis")
# set upload mode
c1, c2 = st.columns(2)
with c1:
    if st.button(
        label="Upload single file", 
        width="stretch"
    ):
        st.session_state._DL_Mode = "single"
with c2:
    if st.button(
        label="Upload and connect multiple files", 
        width="stretch"
    ):
        st.session_state._DL_Mode = "multiple"

# signle file upload and load
if st.session_state._DL_Mode == "single":
    #file uploader
    uploaded_file = st.file_uploader(
        label="Choose a dataset", 
        type={"pcap", "pcapng", "csv", "txt"},
        accept_multiple_files=False
    )

    if  uploaded_file is not None:
        # determine file type
        uploaded_file_type = uploaded_file.name.split('.')[-1]
        # check for PCAP(NG) file for extendet dataload
        if uploaded_file_type == "pcap" or uploaded_file_type == "pcapng":
            # Core settings
            st.subheader("General Settings")
            core = st.checkbox("Include core/meta info (timestamp, datetime, length)", value=True)
            utc_time = st.checkbox("Use UTC time (instead of local)", value=True)
            stream = st.checkbox("Stream packets with PcapReader (better for large PCAPs)", value=False)
            max_packets = st.number_input("Max packets to read (0 = all)", min_value=0, value=0)

            # L2
            with st.expander("Layer 2 (Data Link)"):
                ether = st.checkbox("Ethernet", value=False)
                vlan = st.checkbox("VLAN (802.1Q)", value=False)
                arp = st.checkbox("ARP", value=False)

            # L3
            with st.expander("Layer 3 (Network)"):
                ip4 = st.checkbox("IPv4", value=True)
                ip6 = st.checkbox("IPv6", value=True)

            # L4
            with st.expander("Layer 4 (Transport)"):
                tcp = st.checkbox("TCP", value=True)
                udp = st.checkbox("UDP", value=True)
                icmp = st.checkbox("ICMP (IPv4)", value=False)
                icmp6 = st.checkbox("ICMPv6", value=False)

            # L7
            with st.expander("Layer 7 (Application)"):
                dns = st.checkbox("DNS", value=True)
                http = st.checkbox("HTTP", value=False)
                tls = st.checkbox("TLS (SNI, ALPN, version)", value=False)
                dhcp = st.checkbox("DHCP", value=False)
                ntp = st.checkbox("NTP", value=False)

            # Payload
            with st.expander("Payload Options"):
                raw_len = st.checkbox("Include payload length", value=True)
                raw_hexdump = st.checkbox("Include payload hexdump preview", value=False)
                raw_hexdump_bytes = st.slider("Max hexdump bytes", 8, 128, 32)
        # settings for non-pcap files
        else:   
            core = False
            utc_time = False
            stream = False
            max_packets = 0

            ether = False
            vlan = False
            arp = False

            ip4 = False
            ip6 = False

            tcp = False
            udp = False
            icmp = False
            icmp6 = False

            dns = False
            http = False
            tls = False
            dhcp = False
            ntp = False

            raw_len = False
            raw_hexdump = False
            raw_hexdump_bytes = 8
        # load button
        if st.button("Load dataset"):
            st.session_state._DF = dl.load(
                uploaded_file, uploaded_file_type, 
                core, utc_time, stream, max_packets,
                ether, vlan, arp,
                ip4, ip6,
                tcp, udp, icmp, icmp6,
                dns, http, tls, dhcp, ntp,
                raw_len, raw_hexdump, raw_hexdump_bytes
            )
            # set dataset name
            st.session_state._DL_Filename = uploaded_file.name
            # reset uploaded files list
            st.session_state._DL_UploadedFiles = None
            
            if st.session_state._DF is not None:
                # log
                logger.save_log("Dataset " + st.session_state._DL_Filename + " loaded")
                # set data loaded flag
                st.session_state._DL_DataLoaded = True
                # detect  and set timestamp columns
                timestamp_candidates = dl.detect_timestamp_cols(st.session_state._DF)
                if timestamp_candidates:
                    st.session_state._HasTimeStamp = True
                    st.session_state._TimeStampCol = timestamp_candidates[0]
                st.rerun()
    # no file uploaded yet
    else:
        st.info("No file uploaded yet")

# multiple file upload and load
if st.session_state._DL_Mode == "multiple":
    # file uploader
    uploaded = st.file_uploader(
        "Upload one or more CSV files",
        type={"csv"},
        accept_multiple_files=True
    )
    # import options
    with st.expander("Import options"):
        dataset_name = st.text_input("Name for the combined dataset", value="Combined Dataset")
        add_source = st.checkbox("Add a column with source filename", value=False)
        header = st.checkbox("First row is header", value=True)
        encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "utf-16"], index=0)
        sep_choice = st.selectbox("Delimiter", ["Auto-detect", ",", ";", "\t", "|"], index=0)
        keep = st.selectbox("Column strategy (when schemas differ)",["Union (keep all columns)", "Intersection (only shared columns)"], index=0)
        drop_dupes = st.checkbox("Drop duplicate rows after combining", value=False)
    
    # load button
    if uploaded is not None:
        if st.button("Load dataset(s)"):
            # load multiple csv files
            multi_res = dl.load_multiple_csv(uploaded, sep_choice, header, encoding, keep, drop_dupes, add_source)
            # set combined dataframe
            st.session_state._DF = multi_res["combined"]
            # collect error messages
            err_msgs = multi_res["err_msgs"]
            if err_msgs:
                for msg in err_msgs:
                    st.error(msg, width="stretch")
                    
            if st.session_state._DF is not None:
                # set dataset name
                st.session_state._DL_Filename = dataset_name
                # set uploaded files list
                file_names = [f.name for f in uploaded]
                st.session_state._DF_UploadedFiles = file_names
                # log
                logger.save_log(
                    f"All datasets ({', '.join(file_names)}) combined into '{dataset_name}'"
                )
                # set data loaded flag
                st.session_state._DL_DataLoaded = True
                # detect  and set timestamp columns
                timestamp_candidates = dl.detect_timestamp_cols(st.session_state._DF)
                if timestamp_candidates:
                    st.session_state._HasTimeStamp = True
                    st.session_state._TimeStampCol = timestamp_candidates[0]
                st.rerun()
            
            else:
                # error
                logger.save_log("Failed to load datasets")
                st.error("Failed to load datasets")
    # no files uploaded yet
    else:
        st.info("No files uploaded yet")

# display & downlaod loaded data
if st.session_state._DL_DataLoaded == True and st.session_state._DF is not None:
    st.header("Information")
    # data preview
    st.subheader("Data preview: " + st.session_state._DL_Filename)

    df = st.session_state._DF
    st.dataframe(df.head(5))

    # dataset metrics
    rows, cols = df.shape
    mem_bytes = int(df.memory_usage(deep=True).sum())

    # centered metrics
    left_spacer, left, mid, right, right_spacer = st.columns([1, 1, 1, 1, 1])
    left.metric("Rows", f"{rows:,}")
    mid.metric("Columns", f"{cols:,}")
    right.metric("Memory", te.fmt_bytes(mem_bytes))

    # show uploaded files
    if st.session_state._DL_UploadedFiles is not None:
        with st.expander("Source files"):
            st.write(st.session_state._DL_UploadedFiles)

    # select timestamp and label columns
    t, l = st.columns(2)
    # timestamp column
    with t:
        st.subheader("Timestamp collumn")
        st.session_state._HasTimeStamp = st.checkbox("Dataset has a timestamp collumn", st.session_state._HasTimeStamp)
        if st.session_state._HasTimeStamp == True:
            # detect timestamp columns
            timestamp_candidates = dl.detect_timestamp_cols(st.session_state._DF)
            # selectbox for timestamp columns
            if timestamp_candidates:
                default_timestamp_col = st.session_state.get("_TimeStampCol", timestamp_candidates[0])
                default_timestamp_index = timestamp_candidates.index(default_timestamp_col) if default_timestamp_col in timestamp_candidates else 0
                st.session_state._TimeStampCol = st.selectbox("Timestamp Column", timestamp_candidates, default_timestamp_index)
            else:
                st.warning("No timestamp-like columns detected.")
    # label column
    with l:
        st.subheader("Label collumn")
        st.session_state._HasLabel = st.checkbox("Dataset has a label collumn", st.session_state._HasLabel)
        # selectbox for label columns
        if st.session_state._HasLabel == True:
            cols = st.session_state._DF.columns.tolist()
            default_label_col = st.session_state.get("_LabelCol", cols[0])  # fallback default
            default_label_index = cols.index(default_label_col) if default_label_col in cols else 0
            st.session_state._LabelCol = st.selectbox("Label Column", cols, default_label_index)

    # Download data
    st.header("Downloads")

    csv_bytes = st.session_state._DF.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download dataset as CSV",
        data=csv_bytes,
        file_name=st.session_state._DL_Filename.split('.')[0]+".csv",
        mime="text/csv",
        width="stretch"
    )
