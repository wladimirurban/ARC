import io
import pandas as pd
import datetime as dt
import warnings

from typing import Optional

from scapy.all import rdpcap, PcapReader, Ether, Raw
from scapy.layers.l2 import ARP, Dot1Q
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.inet6 import IPv6, ICMPv6EchoRequest, ICMPv6EchoReply
from scapy.layers.dns import DNS
from scapy.layers.http import HTTPRequest, HTTPResponse
from scapy.layers.tls.all import TLSClientHello, TLSServerHello
from scapy.layers.dhcp import DHCP, BOOTP
from scapy.layers.ntp import NTP

# suppress UserWarnings from scapy
warnings.filterwarnings("ignore", category=UserWarning)

# main load function
def load(file, ext, 
         core, utc_time, stream, max_packets,
         ether, vlan, arp,
         ip4, ip6,
         tcp, udp, icmp, icmp6,
         dns, http, tls, dhcp, ntp,
         raw_len, raw_hexdump, raw_hexdump_bytes
         ):
    if ext == "csv":
        return pd.read_csv(file)
    elif ext == "json":
        return pd.read_json(file)
    elif ext == "txt":
        return load_txt(file)
    elif ext in ["pcap", "pcapng"]:
        return load_pcap_select(
                file, core, utc_time, stream, max_packets,
                ether, vlan, arp,
                ip4, ip6,
                tcp, udp, icmp, icmp6,
                dns, http, tls, dhcp, ntp,
                raw_len, raw_hexdump, raw_hexdump_bytes)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

# load txt file as dataframe
def load_txt(file):
    try:
        return pd.read_csv(file, sep=None, engine='python')
    except Exception:
        with open(file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
        return pd.DataFrame({'line': lines})

# load pcap/pcapng with layer selection
def load_pcap_select(
    file_path: str,
    # core packet features
    core: bool = True,
    # use UTC for datetime conversion
    utc_time: bool = True,
    # use PcapReader to stream large files
    stream: bool = False,
    # stop early
    max_packets: Optional[int] = None,

    # L2
    ether: bool = False,
    vlan: bool = False,
    arp: bool = False,

    # L3
    ip4: bool = True,
    ip6: bool = True,

    # L4
    tcp: bool = True,
    udp: bool = True,
    icmp: bool = False,
    icmp6: bool = False,

    # L7 (common)
    dns: bool = True,
    http: bool = False,
    tls: bool = False,
    dhcp: bool = False,
    ntp: bool = False,

    # payload peek
    # length of raw payload
    raw_len: bool = True,
    # hex dump of first N bytes of raw payload
    raw_hexdump: bool = False,
    # number of bytes to include in hex dump
    raw_hexdump_bytes: int = 32,
):
    # read packets
    pkt_iter = PcapReader(file_path) if stream else rdpcap(file_path)
    records = []
    count = 0

    try:
        for pkt in pkt_iter:
            if max_packets != 0 and count >= max_packets:
                break
            count += 1

            rec = {}

            # core features
            if core:
                ts = float(pkt.time)
                rec["timestamp"] = ts
                rec["datetime"] = (dt.datetime.fromtimestamp(ts))
                # Wire length (approx): len(Packet) returns decoded length
                try:
                    rec["pkt_len"] = int(len(pkt))
                except Exception:
                    rec["pkt_len"] = None

            # L2
            if ether and Ether in pkt:
                eth = pkt[Ether]
                rec.update({
                    "eth_src": eth.src,
                    "eth_dst": eth.dst,
                    "eth_type": hex(eth.type),
                })

            if vlan and Dot1Q in pkt:
                v = pkt[Dot1Q]
                rec.update({"vlan_id": v.vlan, "vlan_pcp": v.prio})

            if arp and ARP in pkt:
                a = pkt[ARP]
                rec.update({"arp_op": a.op, "arp_psrc": a.psrc, "arp_pdst": a.pdst})

            # L3
            if ip4 and IP in pkt:
                ip = pkt[IP]
                rec.update({
                    "ip_src": ip.src,
                    "ip_dst": ip.dst,
                    "ip_proto": ip.proto,
                    "ip_ttl": ip.ttl,
                })

            if ip6 and IPv6 in pkt:
                ip6l = pkt[IPv6]
                rec.update({
                    "ip6_src": ip6l.src,
                    "ip6_dst": ip6l.dst,
                    "ip6_tc": ip6l.tc,
                    "ip6_fl": ip6l.fl,
                    "ip6_hlim": ip6l.hlim,
                })

            # L4
            if tcp and TCP in pkt:
                t = pkt[TCP]
                # TCP options quick map
                try:
                    opts = dict(t.options)
                except Exception:
                    opts = {}
                rec.update({
                    "tcp_sport": t.sport,
                    "tcp_dport": t.dport,
                    "tcp_flags": t.sprintf("%TCP.flags%"),
                    "tcp_seq": t.seq,
                    "tcp_ack": t.ack,
                    "tcp_window": t.window,
                    "tcp_opt_mss": opts.get("MSS"),
                    "tcp_opt_wscale": opts.get("WScale"),
                    "tcp_opt_sackok": 1 if ("SAckOK" in opts or ("SAckOK", b"") in getattr(t, "options", ())) else 0,
                })

            if udp and UDP in pkt:
                u = pkt[UDP]
                rec.update({
                    "udp_sport": u.sport,
                    "udp_dport": u.dport,
                    "udp_len": getattr(u, "len", None),
                })

            if icmp and ICMP in pkt:
                ic = pkt[ICMP]
                rec.update({"icmp_type": ic.type, "icmp_code": ic.code})

            if icmp6 and (ICMPv6EchoRequest in pkt or ICMPv6EchoReply in pkt):
                ic6 = pkt.getlayer(ICMPv6EchoRequest) or pkt.getlayer(ICMPv6EchoReply)
                if ic6:
                    rec.update({"icmp6_type": ic6.type, "icmp6_code": ic6.code})

            # L7
            if dns and DNS in pkt:
                d = pkt[DNS]
                rec.update({
                    "dns_qr": d.qr,
                    "dns_opcode": d.opcode,
                    "dns_rcode": d.rcode,
                    "dns_qdcount": d.qdcount,
                })

            if http and HTTPRequest and HTTPResponse:
                if HTTPRequest in pkt:
                    h = pkt[HTTPRequest]
                    rec.update({
                        "http_method": bytes(h.Method or b"").decode(errors="ignore") or None,
                        "http_host": bytes(h.Host or b"").decode(errors="ignore") or None,
                        "http_path": bytes(h.Path or b"").decode(errors="ignore") or None,
                    })
                elif HTTPResponse in pkt:
                    r = pkt[HTTPResponse]
                    status = getattr(r, "Status_Code", None)
                    rec.update({"http_status": int(status) if status else None})

            if tls and TLSClientHello and TLSServerHello:
                ch = pkt.getlayer(TLSClientHello)
                if ch:
                    sni = getattr(ch, "server_names", None)
                    alpn = getattr(ch, "alpn_protocols", None)
                    rec.update({
                        "tls_sni": (sni[0].servername.decode() if sni else None),
                        "tls_alpn": (",".join(p.protocol_name for p in (alpn or [])) or None),
                    })
                sh = pkt.getlayer(TLSServerHello)
                if sh:
                    rec.update({"tls_version": getattr(sh, "version", None)})

            if dhcp and DHCP and BOOTP and (DHCP in pkt and BOOTP in pkt):
                bootp = pkt[BOOTP]
                try:
                    dhcp_opts = dict((k, v) for k, v in pkt[DHCP].options if isinstance(k, str))
                except Exception:
                    dhcp_opts = {}
                rec.update({
                    "dhcp_msg_type": dhcp_opts.get("message-type"),
                    "dhcp_yiaddr": getattr(bootp, "yiaddr", None),
                    "dhcp_xid": getattr(bootp, "xid", None),
                })

            if ntp and NTP and NTP in pkt:
                n = pkt[NTP]
                rec.update({
                    "ntp_mode": getattr(n, "mode", None),
                    "ntp_stratum": getattr(n, "stratum", None),
                    "ntp_poll": getattr(n, "poll", None),
                })

            # payload peek
            if raw_len or raw_hexdump:
                if Raw in pkt:
                    payload = pkt[Raw].load
                    if raw_len:
                        rec["raw_len"] = len(payload)
                    if raw_hexdump:
                        head = payload[:raw_hexdump_bytes].hex()
                        rec["raw_hexdump"] = head + ("..." if len(payload) > raw_hexdump_bytes else "")
                else:
                    if raw_len:
                        rec["raw_len"] = None
                    if raw_hexdump:
                        rec["raw_hexdump"] = None

            records.append(rec)

    finally:
        # Close PcapReader if streaming
        if stream and hasattr(pkt_iter, "close"):
            pkt_iter.close()
    # create dataframe
    df = pd.DataFrame.from_records(records)

    # Normalize datetime dtype if present
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=utc_time)
    return df

def _read_csv(file, sep_choice, header, encoding) -> pd.DataFrame:
    # Auto-detect separator if chosen
    if sep_choice == "Auto-detect":
        return pd.read_csv(file, sep=None, engine="python",
                           header=0 if header else None, encoding=encoding)
    else:
        return pd.read_csv(file, sep=sep_choice.replace("\\t","\t"),
                           header=0 if header else None, encoding=encoding)

def load_multiple_csv(uploaded, sep_choice, header, encoding, keep, drop_dupes, add_source):
    combined = None
    frames = []
    err_msgs = []
    # read each file
    for f in uploaded:
        try:
            df = _read_csv(io.StringIO(f.getvalue().decode(encoding)), sep_choice, header, encoding)
            if add_source:
                df.insert(0, "_source_file", f.name)
            frames.append(df)
        except Exception as e:
            err_msgs.append(f"Error reading {f.name}: {e}")

    if frames:
        if keep.startswith("Union"):
            combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)
        else:
            # Only keep columns common to all files
            common_cols = set(frames[0].columns)
            for df in frames[1:]:
                common_cols &= set(df.columns)
            common_cols = [c for c in frames[0].columns if c in common_cols]  # keep order
            frames = [df[common_cols].copy() for df in frames]
            combined = pd.concat(frames, axis=0, ignore_index=True)

        if drop_dupes:
            combined = combined.drop_duplicates()
    return {"combined": combined, "err_msgs": err_msgs}

def detect_timestamp_cols(df):
    timestamp_cols = []
    for col in df.columns:
        series = df[col]
        # case 1: dtype is already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            timestamp_cols.append(col)
        # case 2: dtype is object/string but looks like datetime (try parse sample)
        elif series.dtype == "object":
            try:
                pd.to_datetime(series.dropna().sample(min(10, len(series))), errors="raise")
                timestamp_cols.append(col)
            except Exception:
                pass
            
    return timestamp_cols