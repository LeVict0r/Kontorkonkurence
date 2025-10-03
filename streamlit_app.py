# streamlit_app.py
# ---------------------------------------------------------------
# Erhvervskulturpriserne 2025 – Kontorspil
# Persistent storage: Google Sheets (hvis konfigureret) ellers lokale filer
# Jury (60%) + Offentlig (40%), historik, predictions & leaderboard
# Rate-limit venlig: caching + backoff + minimal init
# ---------------------------------------------------------------

import os
import json
import time
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from gspread.exceptions import APIError  # til backoff

# (Valgfrit) brug .env lokalt
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- Konfiguration ----------
APP_TITLE = "Erhvervskulturpriserne 2025 – Kontorspil"

# Lokalt fallback
DATA_DIR = os.environ.get("EKP_DATA_DIR", "data")
PREDICTIONS_CSV = os.path.join(DATA_DIR, "predictions.csv")
SNAPSHOTS_CSV  = os.path.join(DATA_DIR, "snapshots.csv")
NOMINEES_JSON  = os.path.join(DATA_DIR, "nominees.json")
JURY_CSV       = os.path.join(DATA_DIR, "jury.csv")
CONFIG_JSON    = os.path.join(DATA_DIR, "config.json")  # her kan vi gemme SPREADSHEET_ID via UI

# Vigtige scopes (inkl. Drive for delte drev mv.)
GSPREAD_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

CATEGORIES: List[str] = [
    "Årets Leder",
    "Årets Talent",
    "Årets Iværksætter",
    "Årets Medarbejderkultur",
    "Ansvarlighedsprisen",
    "Innovationsprisen",
    "Årets Virksomhed",
]
PARTICIPANTS: List[str] = ["Kristina", "Jan", "Victor", "Sara", "Mette", "Peter"]

def _secret(name: str, default: Optional[str] = None) -> Optional[str]:
    val = None
    try:
        val = st.secrets.get(name, None)
    except Exception:
        val = None
    if val is None:
        val = os.environ.get(name, None)
    return default if (val is None and default is not None) else val

ADMIN_PASSWORD = _secret("EKP_ADMIN_PASSWORD", "admin123")
PARTICIPANT_CODES = {
    "Kristina": _secret("EKP_CODE_KRISTINA", "kri-2025"),
    "Jan":      _secret("EKP_CODE_JAN",       "jan-2025"),
    "Victor":   _secret("EKP_CODE_VICTOR",    "vic-2025"),
    "Sara":     _secret("EKP_CODE_SARA",      "sar-2025"),
    "Mette":    _secret("EKP_CODE_METTE",     "met-2025"),
    "Peter":    _secret("EKP_CODE_PETER",     "pet-2025"),
}

COMPETITION_END = date(2025, 10, 26)

# ---------- Lokal config (til at gemme Sheet ID via UI) ----------
def _read_config() -> Dict[str, str]:
    try:
        if os.path.exists(CONFIG_JSON):
            with open(CONFIG_JSON, "r", encoding="utf-8") as f:
                obj = json.load(f)
                if isinstance(obj, dict):
                    return obj
    except Exception:
        pass
    return {}

def _write_config(d: Dict[str, str]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

# ---------- Helper: Backoff på gspread-kald ----------
def _with_backoff(call, *args, **kwargs):
    """
    Kør et gspread-opkald med eksponentiel backoff ved 429/Quota-rate limit.
    """
    delays = [0.3, 0.8, 1.5, 2.5, 4.0]  # sekunder
    last_err = None
    for d in delays:
        try:
            return call(*args, **kwargs)
        except APIError as e:
            msg = str(e).lower()
            if "429" in msg or "quota" in msg or "rate" in msg:
                last_err = e
                time.sleep(d)
                continue
            raise
    # sidste forsøg uden ekstra sleep
    return call(*args, **kwargs)

# ---------- Normalisering af Sheet ID ----------
def _normalize_sheet_id(s: Optional[str]) -> Optional[str]:
    """Accepter både ren ID og hele URL’er – returnér ren ID eller None."""
    if not s:
        return None
    sid = str(s).strip()
    if sid.startswith("http"):
        import re
        m = re.search(r"/d/([a-zA-Z0-9-_]+)", sid)
        return m.group(1) if m else None
    return sid

# ---------- Google Sheets: robust ID-hentning ----------
def _resolve_spreadsheet_id_with_source() -> Tuple[Optional[str], str]:
    """
    Finder SPREADSHEET_ID med klar prioritet og returnerer (id, kilde).
    Kilder i prioriteret rækkefølge:
      1) st.session_state["SPREADSHEET_ID_OVERRIDE"]
      2) data/config.json
      3) st.secrets["SPREADSHEET_ID"]
      4) st.secrets["gcp_service_account"]["SPREADSHEET_ID"]
      5) os.environ["SPREADSHEET_ID"]
    """
    # 1) session override
    sid = st.session_state.get("SPREADSHEET_ID_OVERRIDE")
    if sid:
        return _normalize_sheet_id(sid), "session_state"

    # 2) lokal config
    cfg = _read_config()
    sid = cfg.get("SPREADSHEET_ID")
    if sid:
        return _normalize_sheet_id(sid), "config.json"

    # 3) secrets topniveau
    try:
        sid = st.secrets.get("SPREADSHEET_ID", None)
        if sid:
            return _normalize_sheet_id(sid), "secrets_top"
    except Exception:
        pass

    # 4) secrets inde i gcp_service_account
    try:
        svc = st.secrets.get("gcp_service_account", None)
        if isinstance(svc, dict):
            sid = svc.get("SPREADSHEET_ID", None)
            if sid:
                return _normalize_sheet_id(sid), "secrets_gcp_block"
    except Exception:
        pass

    # 5) miljøvariabel
    sid = os.environ.get("SPREADSHEET_ID")
    if sid:
        return _normalize_sheet_id(sid), "env"

    return None, "none"

def _get_spreadsheet_id() -> Optional[str]:
    sid, _ = _resolve_spreadsheet_id_with_source()
    return sid

def _sheets_available() -> bool:
    try:
        svc = st.secrets.get("gcp_service_account", None)
    except Exception:
        svc = None
    sid = _get_spreadsheet_id()
    return bool(svc and sid)

def use_sheets() -> bool:
    """Kald denne i stedet for en global konstant, så vi kan skifte on-the-fly."""
    return _sheets_available()

# ---------- Google Sheets klient (cachet) ----------
@st.cache_resource(show_spinner=False)
def _gspread_client_cached():
    import gspread
    from google.oauth2.service_account import Credentials
    svc = st.secrets.get("gcp_service_account", None)
    if not svc:
        raise RuntimeError("Mangler [gcp_service_account] i Secrets.")
    creds = Credentials.from_service_account_info(svc, scopes=GSPREAD_SCOPES)
    return gspread.authorize(creds)

def _get_gspread() -> Tuple["gspread.client.Client", "gspread.Spreadsheet"]:
    client = _gspread_client_cached()
    sid = _get_spreadsheet_id()
    if not sid:
        raise RuntimeError("Google Sheets er ikke konfigureret: mangler et gyldigt SPREADSHEET_ID (UI/Secrets/ENV).")
    sheet = _with_backoff(client.open_by_key, _normalize_sheet_id(sid))
    return client, sheet

# ---------- Worksheet helpers ----------
def _ensure_ws(name: str, header: List[str]):
    _, sh = _get_gspread()
    try:
        ws = _with_backoff(sh.worksheet, name)
    except Exception:
        ws = _with_backoff(sh.add_worksheet, title=name, rows="100", cols=str(max(10, len(header))))
        _with_backoff(ws.update, [header])
        return ws
    # Sikr header (tjek kun én gang pr. session for at spare reads)
    session_key = f"_ws_header_checked_{name}"
    if not st.session_state.get(session_key, False):
        row1 = _with_backoff(ws.get_values, "A1:Z1")
        current = row1[0] if row1 else []
        if current != header:
            _with_backoff(ws.clear)
            _with_backoff(ws.update, [header])
        st.session_state[session_key] = True
    return ws

def _df_to_ws(name: str, df: pd.DataFrame, header: List[str]):
    ws = _ensure_ws(name, header)
    values = [header]
    if not df.empty:
        values += df[header].astype(str).values.tolist()
    _with_backoff(ws.clear)
    _with_backoff(ws.update, values)

def _ws_to_df(name: str, header: List[str]) -> pd.DataFrame:
    ws = _ensure_ws(name, header)
    values = _with_backoff(ws.get_all_values)
    if not values or len(values) == 1:
        return pd.DataFrame(columns=header)
    out = pd.DataFrame(values[1:], columns=header)
    # Typer
    for col in out.columns:
        if col in ("votes",):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
        elif col in ("jury_pct", "public_pct", "combined_pct"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

# Cache læsninger kort, for at undgå 429 (rydder vi efter skriv)
@st.cache_data(ttl=20, show_spinner=False)
def _cached_ws_to_df(name: str, header: List[str]) -> pd.DataFrame:
    return _ws_to_df(name, header)

# ---------- Storage API ----------
def ensure_storage():
    if use_sheets():
        _ensure_ws("predictions", ["participant","category","pick1","pick2","pick3","submitted_at"])
        _ensure_ws("snapshots",   ["batch_id","timestamp","category","nominee","votes"])
        _ensure_ws("jury",        ["category","nominee","jury_pct","updated_at"])
        _ensure_ws("nominees",    ["category","nominee"])
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(NOMINEES_JSON):
            with open(NOMINEES_JSON, "w", encoding="utf-8") as f:
                json.dump({cat: [] for cat in CATEGORIES}, f, ensure_ascii=False, indent=2)
        if not os.path.exists(PREDICTIONS_CSV):
            pd.DataFrame(columns=["participant","category","pick1","pick2","pick3","submitted_at"]).to_csv(PREDICTIONS_CSV, index=False)
        if not os.path.exists(SNAPSHOTS_CSV):
            pd.DataFrame(columns=["batch_id","timestamp","category","nominee","votes"]).to_csv(SNAPSHOTS_CSV, index=False)
        if not os.path.exists(JURY_CSV):
            pd.DataFrame(columns=["category","nominee","jury_pct","updated_at"]).to_csv(JURY_CSV, index=False)

def ensure_storage_once():
    """
    Kør worksheet-initialisering kun én gang pr. session (spar læsninger).
    """
    if use_sheets():
        if not st.session_state.get("_ws_ready", False):
            ensure_storage()
            st.session_state["_ws_ready"] = True
    else:
        ensure_storage()

def load_nominees() -> Dict[str, List[str]]:
    if use_sheets():
        df = _cached_ws_to_df("nominees", ["category","nominee"])
        out = {cat: [] for cat in CATEGORIES}
        for cat in CATEGORIES:
            out[cat] = df[df["category"] == cat]["nominee"].dropna().tolist()
        return out
    else:
        with open(NOMINEES_JSON, "r", encoding="utf-8") as f:
            return json.load(f)

def save_nominees(nominees: Dict[str, List[str]]):
    if use_sheets():
        rows = []
        for cat, arr in nominees.items():
            for nom in arr:
                rows.append({"category": cat, "nominee": nom})
        df = pd.DataFrame(rows, columns=["category","nominee"])
        _df_to_ws("nominees", df, ["category","nominee"])
        st.cache_data.clear()
    else:
        with open(NOMINEES_JSON, "w", encoding="utf-8") as f:
            json.dump(nominees, f, ensure_ascii=False, indent=2)

def read_predictions() -> pd.DataFrame:
    if use_sheets():
        return _cached_ws_to_df("predictions", ["participant","category","pick1","pick2","pick3","submitted_at"])
    else:
        if os.path.exists(PREDICTIONS_CSV):
            return pd.read_csv(PREDICTIONS_CSV)
        return pd.DataFrame(columns=["participant","category","pick1","pick2","pick3","submitted_at"])

def write_predictions(df: pd.DataFrame):
    if use_sheets():
        _df_to_ws("predictions", df, ["participant","category","pick1","pick2","pick3","submitted_at"])
        st.cache_data.clear()
    else:
        df.to_csv(PREDICTIONS_CSV, index=False)

def read_snapshots() -> pd.DataFrame:
    if use_sheets():
        df = _cached_ws_to_df("snapshots", ["batch_id","timestamp","category","nominee","votes"])
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    else:
        if os.path.exists(SNAPSHOTS_CSV):
            df = pd.read_csv(SNAPSHOTS_CSV)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        return pd.DataFrame(columns=["batch_id","timestamp","category","nominee","votes"])

def write_snapshots(df: pd.DataFrame):
    if not df.empty:
        df = df.sort_values(["timestamp","batch_id"], ascending=[False, False])
    if use_sheets():
        tmp = df.copy()
        if "timestamp" in tmp.columns:
            tmp["timestamp"] = tmp["timestamp"].astype(str)
        _df_to_ws("snapshots", tmp, ["batch_id","timestamp","category","nominee","votes"])
        st.cache_data.clear()
    else:
        df.to_csv(SNAPSHOTS_CSV, index=False)

def read_jury() -> pd.DataFrame:
    if use_sheets():
        return _cached_ws_to_df("jury", ["category","nominee","jury_pct","updated_at"])
    else:
        if os.path.exists(JURY_CSV):
            return pd.read_csv(JURY_CSV)
        return pd.DataFrame(columns=["category","nominee","jury_pct","updated_at"])

def write_jury(df: pd.DataFrame):
    if use_sheets():
        _df_to_ws("jury", df, ["category","nominee","jury_pct","updated_at"])
        st.cache_data.clear()
    else:
        df.to_csv(JURY_CSV, index=False)

# ---------- Beregninger ----------
def current_table_from_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame(columns=["category","nominee","votes"])
    snapshots = snapshots.sort_values(["timestamp"]).copy()
    latest = (
        snapshots.groupby("category")["timestamp"].max().reset_index().rename(columns={"timestamp":"latest_timestamp"})
    )
    merged = snapshots.merge(latest, left_on=["category","timestamp"], right_on=["category","latest_timestamp"], how="inner")
    return merged[["category","nominee","votes"]]

def public_pct_from_current(cur_votes: pd.DataFrame) -> pd.DataFrame:
    if cur_votes.empty:
        return pd.DataFrame(columns=["category","nominee","votes","public_pct"])
    df = cur_votes.copy()
    sums = df.groupby("category")["votes"].transform("sum").replace(0, pd.NA)
    df["public_pct"] = (df["votes"] / sums * 100).fillna(0.0)
    return df

def combined_rank_table(cur_votes: pd.DataFrame, jury_df: pd.DataFrame) -> pd.DataFrame:
    pub = public_pct_from_current(cur_votes)
    jury = jury_df.copy() if not jury_df.empty else pd.DataFrame(columns=["category","nominee","jury_pct"])
    if not jury.empty:
        jury["jury_pct"] = pd.to_numeric(jury["jury_pct"], errors="coerce").fillna(0.0)
    merged = pub.merge(jury[["category","nominee","jury_pct"]], on=["category","nominee"], how="left")
    merged["jury_pct"] = merged["jury_pct"].fillna(0.0)
    merged["combined_pct"] = 0.6 * merged["jury_pct"] + 0.4 * merged["public_pct"]
    return merged[["category","nominee","votes","jury_pct","public_pct","combined_pct"]]

def ranking_per_category_combined(combined_df: pd.DataFrame) -> Dict[str, List[str]]:
    ranks = {}
    for cat in CATEGORIES:
        dfc = combined_df[combined_df["category"] == cat].copy()
        if dfc.empty:
            ranks[cat] = []
            continue
        dfc = dfc.sort_values(["combined_pct","nominee"], ascending=[False, True])
        ranks[cat] = dfc["nominee"].tolist()
    return ranks

def compute_potential_points(preds: pd.DataFrame, ranks: Dict[str, List[str]]) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame(columns=["participant","category","pick1","pick2","pick3","potential_points"])

    def row_points(row) -> int:
        top = ranks.get(row["category"], [])
        pts = 0
        if len(top) >= 1 and row["pick1"] == top[0]: pts += 5
        if len(top) >= 2 and row["pick2"] == top[1]: pts += 1
        if len(top) >= 3 and row["pick3"] == top[2]: pts += 1
        return pts

    out = preds.copy()
    out["potential_points"] = out.apply(row_points, axis=1)
    return out

# ---------- UI: Login ----------
def sidebar_admin_login() -> bool:
    st.sidebar.subheader("Admin")
    with st.sidebar.expander("Admin-login", expanded=False):
        pwd = st.text_input("Adgangskode", type="password").strip()
        ok = st.button("Log ind")
    if ok:
        st.session_state["is_admin"] = (pwd == ADMIN_PASSWORD)
        if not st.session_state.get("is_admin", False):
            st.sidebar.error("Forkert adgangskode")
    return st.session_state.get("is_admin", False)

def sidebar_participant_login() -> str:
    st.sidebar.subheader("Deltager-login")
    user = st.sidebar.selectbox("Vælg deltager", PARTICIPANTS)
    code = st.sidebar.text_input("Personlig kode", type="password").strip()
    login = st.sidebar.button("Log ind som deltager")
    if login:
        expected = PARTICIPANT_CODES.get(user)
        st.session_state["participant"] = user if code == expected else None
        if st.session_state.get("participant") is None:
            st.sidebar.error("Forkert personlig kode")
    return st.session_state.get("participant")

# ---------- UI: Sektioner ----------
def section_status_overview(snapshots: pd.DataFrame):
    st.header("📊 Live stilling & grafer")
    cur = current_table_from_snapshots(snapshots)
    if cur.empty:
        st.info("Ingen snapshots endnu. Gå til Admin for at tilføje stemmetal.")
        return

    jury_df = read_jury()
    combo = combined_rank_table(cur, jury_df)

    tabs = st.tabs(CATEGORIES)
    for i, cat in enumerate(CATEGORIES):
        with tabs[i]:
            st.subheader(cat)
            cat_df = combo[combo["category"] == cat].copy()
            if cat_df.empty:
                st.write("Ingen data for denne kategori endnu.")
                continue

            cat_df = cat_df.sort_values(["combined_pct","nominee"], ascending=[False, True]).reset_index(drop=True)
            display = cat_df.rename(columns={
                "nominee": "Kandidat",
                "votes": "Off. stemmer",
                "jury_pct": "Jury %",
                "public_pct": "Offentlig %",
                "combined_pct": "Samlet % (60/40)"
            }).copy()
            for col in ["Jury %","Offentlig %","Samlet % (60/40)"]:
                display[col] = display[col].map(lambda x: round(float(x), 2))

            st.dataframe(
                display[["Kandidat","Off. stemmer","Jury %","Offentlig %","Samlet % (60/40)"]],
                use_container_width=True
            )

            hist = snapshots[snapshots["category"] == cat].copy()
            if not hist.empty:
                piv = hist.pivot_table(index="timestamp", columns="nominee", values="votes", aggfunc="last").sort_index()
                st.line_chart(piv)

def section_submit_predictions(nominees: Dict[str, List[str]]):
    st.header("📝 Afgiv dine gæt (låses efter indsendelse)")
    participant = st.session_state.get("participant")
    if not participant:
        st.info("Log ind som deltager i sidens sidebar for at indsende dine gæt.")
        return

    st.success(f"Logget ind som: {participant}")
    preds = read_predictions()

    already = set(preds[preds["participant"] == participant]["category"].unique())
    remaining = [c for c in CATEGORIES if c not in already]

    if not remaining:
        st.info("Dine gæt er allerede indsendt for alle kategorier. Kontakt admin for at nulstille, hvis nødvendigt.")
        st.dataframe(preds[preds["participant"] == participant], use_container_width=True)
        return

    with st.form("pred_form"):
        picks = {}
        for cat in remaining:
            st.subheader(cat)
            options = nominees.get(cat, [])
            if not options:
                st.warning("Ingen kandidater i denne kategori endnu – prøv igen senere.")
                continue

            c1, c2, c3 = st.columns(3)
            with c1:
                p1 = st.selectbox("🏆 Vinder", options, key=f"p1_{participant}_{cat}")
            with c2:
                p2 = st.selectbox("🥈 Runnerup", options, key=f"p2_{participant}_{cat}")
            with c3:
                p3 = st.selectbox("🥉 Taber", options, key=f"p3_{participant}_{cat}")

            if len({p1, p2, p3}) < 3:
                st.error("Vælg tre forskellige kandidater pr. kategori.")
            else:
                picks[cat] = (p1, p2, p3)

        submitted = st.form_submit_button("Indsend og lås mine gæt")

    if submitted:
        if not picks:
            st.warning("Intet gemt – tjek at der findes kandidater og at valg er forskellige.")
            return

        rows = []
        now = datetime.now().isoformat()
        for cat, (p1, p2, p3) in picks.items():
            rows.append({
                "participant": participant,
                "category": cat,
                "pick1": p1,
                "pick2": p2,
                "pick3": p3,
                "submitted_at": now,
            })

        new_df = pd.DataFrame(rows)
        out = pd.concat([read_predictions(), new_df], ignore_index=True)
        write_predictions(out)
        st.success("Dine gæt er gemt og låst for de valgte kategorier ✨")

def section_leaderboard(snapshots: pd.DataFrame):
    st.header("🏆 Leaderboard – potentielle point (live)")
    cur  = current_table_from_snapshots(snapshots)
    jury = read_jury()
    combo = combined_rank_table(cur, jury)
    ranks = ranking_per_category_combined(combo)

    preds = read_predictions()
    preds = preds[preds["participant"].isin(PARTICIPANTS)].copy()

    pts_df = compute_potential_points(preds, ranks)
    if pts_df.empty:
        st.info("Ingen forudsigelser endnu.")
        return

    sums = pts_df.groupby("participant")["potential_points"].sum().reset_index()
    sums = sums.sort_values(["potential_points","participant"], ascending=[False, True])

    with st.expander("Detaljer pr. kategori og deltager"):
        st.dataframe(pts_df.sort_values(["participant","category"]).reset_index(drop=True), use_container_width=True)

    st.subheader("Samlet stilling (jo højere potentielle point, jo bedre forudsigelser lige nu)")
    st.dataframe(sums.reset_index(drop=True), use_container_width=True)

def section_admin(nominees: Dict[str, List[str]]):
    st.header("🔧 Admin – kandidater, jury & snapshots")

    # ⚙️ Konfiguration – sæt/vis Spreadsheet ID
    with st.expander("⚙️ Konfiguration"):
        sid, src = _resolve_spreadsheet_id_with_source()
        st.write("Aktuel SPREADSHEET_ID:", sid or "None")
        st.write("Kilde:", src)
        if sid:
            st.write("Åbn direkte:", f"https://docs.google.com/spreadsheets/d/{sid}/edit")
        new_sid = st.text_input("Angiv/ret Google Sheet ID", value=sid or "", help="Selve ID'et fra URL'en mellem /d/ og /edit")
        colA, colB, colC = st.columns(3)
        with colA:
            if st.button("Gem ID til config.json"):
                cfg = _read_config()
                cfg["SPREADSHEET_ID"] = _normalize_sheet_id(new_sid) or ""
                _write_config(cfg)
                st.session_state["SPREADSHEET_ID_OVERRIDE"] = cfg["SPREADSHEET_ID"]
                st.success("SPREADSHEET_ID gemt lokalt og aktiveret for denne session.")
        with colB:
            if st.button("Ryd lokalt ID"):
                cfg = _read_config()
                if "SPREADSHEET_ID" in cfg:
                    del cfg["SPREADSHEET_ID"]
                    _write_config(cfg)
                st.session_state.pop("SPREADSHEET_ID_OVERRIDE", None)
                st.success("Lokalt ID ryddet. Appen vil nu bruge Secrets eller ENV hvis tilgængeligt.")
        with colC:
            if st.button("Initialiser ark (opret faner & headers)"):
                ensure_storage()
                st.session_state["_ws_ready"] = True
                st.success("Ark initialiseret.")

    # Kandidater
    st.subheader("Kandidater pr. kategori")
    with st.form("nominees_form"):
        edits = {}
        for cat in CATEGORIES:
            st.markdown(f"**{cat}**")
            existing = nominees.get(cat, [])
            txt = st.text_area(
                "Én kandidat pr. linje",
                value="\n".join(existing),
                key=f"nom_{cat}",
                height=120,
                help="Indsæt/ret navne. Dubletter fjernes automatisk ved gem."
            )
            edits[cat] = [x.strip() for x in txt.splitlines() if x.strip()]
        if st.form_submit_button("Gem kandidater"):
            for cat, arr in edits.items():
                seen = set(); cleaned = []
                for a in arr:
                    key = a.lower().replace(" ", "")
                    if key not in seen:
                        cleaned.append(a); seen.add(key)
                nominees[cat] = cleaned
            save_nominees(nominees)
            st.success("Kandidater gemt.")

    st.divider()

    # Jury
    st.subheader("Jury-procenter pr. kategori (vægt 60%)")
    st.caption("Angiv juryens procentfordeling pr. kategori. Summerer ideelt til ~100% per kategori.")
    jury_df = read_jury()
    if jury_df.empty:
        jury_df = pd.DataFrame(columns=["category","nominee","jury_pct","updated_at"])

    with st.form("jury_form"):
        updates = []
        for cat in CATEGORIES:
            st.markdown(f"**{cat}**")
            opts = nominees.get(cat, [])
            if not opts:
                st.warning("Ingen kandidater i denne kategori – tilføj dem ovenfor og gem først.")
                continue

            existing = {}
            if not jury_df.empty:
                tmp = jury_df[jury_df["category"] == cat]
                for _, r in tmp.iterrows():
                    try:
                        existing[str(r["nominee"])] = float(r.get("jury_pct", 0.0))
                    except Exception:
                        existing[str(r["nominee"])] = 0.0

            cols = st.columns(3)
            values = {}
            for i, nom in enumerate(opts):
                idx = i % 3
                with cols[idx]:
                    default = float(existing.get(nom, 0.0))
                    values[nom] = st.number_input(
                        f"{nom} – jury %",
                        min_value=0.0, max_value=100.0, step=0.1, value=default,
                        key=f"jury_{cat}_{nom}"
                    )

            total = sum(values.values())
            if abs(total - 100.0) > 0.5:
                st.warning(f"Summen i '{cat}' er {round(total,2)}%. Overvej ~100%. (Gemmer stadig)")

            for nom, pct in values.items():
                updates.append({"category": cat, "nominee": nom, "jury_pct": float(pct), "updated_at": datetime.now().isoformat()})

        save = st.form_submit_button("Gem jury-procenter")
        if save and updates:
            out = pd.DataFrame(updates)
            write_jury(out)
            st.success("Jury-procenter gemt.")

    st.divider()

     # Snapshots (offentlig 40%)
    st.subheader("Tilføj snapshot (offentlige stemmer – vægt 40%)")
    st.caption("Et snapshot er de aktuelle, samlede offentlige stemmetal pr. kandidat. Hver tilføjelse gemmes som historik.")

    # Stabilt snapshot-id pr. "runde" – må først ændres efter gem
    if "snapshot_id" not in st.session_state:
        st.session_state["snapshot_id"] = f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Husk valgt dato på tværs af reruns
    if "snap_date" not in st.session_state:
        st.session_state["snap_date"] = date.today()

    # Brug en form, så reruns ikke smider inputs
    with st.form("snapshot_form", clear_on_submit=False):
        st.session_state["snap_date"] = st.date_input("Dato for snapshot", value=st.session_state["snap_date"])

        valid = True
        # Inputfelter med STABILE keys: votes_{kategori}_{nominee}
        for cat in CATEGORIES:
            st.markdown(f"**{cat}**")
            opts = nominees.get(cat, [])
            if not opts:
                st.warning("Ingen kandidater i denne kategori – tilføj dem ovenfor og gem først.")
                valid = False
                continue

            cols = st.columns(3)
            for i, nom in enumerate(opts):
                idx = i % 3
                key = f"votes_{cat}_{nom}"
                with cols[idx]:
                    # Første gang er value=0, derefter bevares state via key
                    st.number_input(f"{nom}", min_value=0, step=1, key=key)

        submitted = st.form_submit_button("Gem snapshot")

    if submitted:
        if not valid:
            st.error("Kan ikke gemme: manglende kandidater i mindst én kategori.")
        else:
            # Byg rækker ud fra state
            entries = []
            snap_dt = datetime.combine(st.session_state["snap_date"], datetime.now().time()).isoformat()
            for cat in CATEGORIES:
                for nom in nominees.get(cat, []):
                    key = f"votes_{cat}_{nom}"
                    val = int(st.session_state.get(key, 0) or 0)
                    entries.append({
                        "batch_id": st.session_state["snapshot_id"],
                        "timestamp": snap_dt,
                        "category": cat,
                        "nominee": nom,
                        "votes": val,
                    })

            old = read_snapshots()
            df_new = pd.DataFrame(entries)
            out = pd.concat([old, df_new], ignore_index=True)
            write_snapshots(out)  # rydder cache i vores write_*-funktion

            st.success("Snapshot gemt.")
            # Klargør til næste runde: nyt id og nulstil kun vote-felter
            st.session_state["snapshot_id"] = f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            for k in list(st.session_state.keys()):
                if k.startswith("votes_"):
                    st.session_state.pop(k, None)

    # 🔌 Sundhedstjek – Google Sheets
    with st.expander("🔌 Sundhedstjek – Google Sheets"):
        sid, src = _resolve_spreadsheet_id_with_source()
        st.write("Lagring valgt:", "Google Sheets" if use_sheets() else "Lokale filer")
        st.write("SPREADSHEET_ID (opdaget):", sid or "None")
        st.write("Kilde:", src)
        has_svc = st.secrets.get("gcp_service_account", None) is not None
        st.write("Service account i Secrets:", "✅" if has_svc else "❌")
        if sid:
            st.write("Åbn direkte:", f"https://docs.google.com/spreadsheets/d/{sid}/edit")

        if st.button("Kør tjek nu"):
            try:
                client, sh = _get_gspread()
                st.success(f"Forbundet til Google Sheets: {sh.title}")
                ws = _ensure_ws("nominees", ["category","nominee"])
                rows_before = len(_with_backoff(ws.get_all_values))
                marker = "DIAG_" + datetime.now().strftime("%H%M%S")
                _with_backoff(ws.append_row, ["__diagnostic__", marker], value_input_option="RAW")
                rows_after = len(_with_backoff(ws.get_all_values))
                _with_backoff(ws.delete_rows, rows_after)
                st.success("Skriv/læs OK ✅ (test-række skrevet og fjernet)")
                st.write(f"Rækker før: {rows_before}, efter: {rows_after}")
            except Exception as e:
                st.error("Google Sheets-fejl – kunne ikke læse/skrive.")
                st.exception(e)
                st.info("Tjek: deling (Editor), SPREADSHEET_ID (UI/Secrets), og private_key i triple quotes.")

    st.divider()

    # Nulstil gæt
    st.subheader("Nulstil deltager-gæt (administrativt)")
    preds = read_predictions()
    who = st.selectbox("Vælg deltager", PARTICIPANTS)
    if st.button("Nulstil alle gæt for valgt deltager"):
        if preds.empty:
            st.info("Ingen gæt at nulstille.")
        else:
            preds = preds[preds["participant"] != who]
            write_predictions(preds)
            st.success(f"Gæt for {who} er nulstillet.")

    st.divider()

    # Eksport
    st.subheader("Eksport")
    st.download_button("Download kandidater (JSON)", data=json.dumps(nominees, ensure_ascii=False, indent=2), file_name="nominees.json")
    st.download_button("Download snapshots (CSV)", data=read_snapshots().to_csv(index=False), file_name="snapshots.csv")
    st.download_button("Download jury (CSV)", data=read_jury().to_csv(index=False), file_name="jury.csv")
    st.download_button("Download forudsigelser (CSV)", data=read_predictions().to_csv(index=False), file_name="predictions.csv")

# ---------- Hovedapp ----------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    ensure_storage_once()  # kør kun init én gang pr. session (spar Google-reads)
    st.title(APP_TITLE)
    st.caption(f"Lagring: {'Google Sheets' if use_sheets() else 'Lokale filer'}")

    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        st.metric("Deltagere", len(PARTICIPANTS))
    with c2:
        st.metric("Kategorier", len(CATEGORIES))
    with c3:
        st.metric("Slutdato", COMPETITION_END.strftime("%d-%m-%Y"))

    is_admin = sidebar_admin_login()
    _participant = sidebar_participant_login()

    nominees = load_nominees()
    snapshots = read_snapshots()

    section_status_overview(snapshots)
    st.divider()
    section_submit_predictions(nominees)
    st.divider()
    section_leaderboard(snapshots)

    if is_admin:
        st.divider()
        section_admin(nominees)
    else:
        st.info("Admin-sektion er skjult. Log ind i sidebar for at administrere kandidater og snapshots.")

if __name__ == "__main__":
    main()
