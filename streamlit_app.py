# streamlit_app.py
# ---------------------------------------------------------------
# Erhvervskulturpriserne 2025 – internt kontor-spil
# Bygget som en enkelt Streamlit-app, filbaseret lagring.
# ---------------------------------------------------------------

import os
import json
from datetime import datetime, date
from typing import List, Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load .env hvis tilstede
load_dotenv()

# -------------------------- Konfiguration --------------------------
APP_TITLE = "Erhvervskulturpriserne 2025 – Kontorspil"
DATA_DIR = "data"
PREDICTIONS_CSV = os.path.join(DATA_DIR, "predictions.csv")
SNAPSHOTS_CSV = os.path.join(DATA_DIR, "snapshots.csv")
NOMINEES_JSON = os.path.join(DATA_DIR, "nominees.json")

# Faste kategorier
CATEGORIES: List[str] = [
    "Årets Leder",
    "Årets Talent",
    "Årets Iværksætter",
    "Årets Medarbejderkultur",
    "Ansvarlighedsprisen",
    "Innovationsprisen",
    "Årets Virksomhed",
]

# Faste deltagere
PARTICIPANTS: List[str] = ["Kristina", "Jan", "Victor", "Sara", "Mette", "Peter"]

# Simpel adgangskode-model (skift disse i .env eller miljø)
ADMIN_PASSWORD = os.environ.get("EKP_ADMIN_PASSWORD", "admin123")
PARTICIPANT_CODES = {
    "Kristina": os.environ.get("EKP_CODE_KRISTINA", "kri-2025"),
    "Jan": os.environ.get("EKP_CODE_JAN", "jan-2025"),
    "Victor": os.environ.get("EKP_CODE_VICTOR", "vic-2025"),
    "Sara": os.environ.get("EKP_CODE_SARA", "sar-2025"),
    "Mette": os.environ.get("EKP_CODE_METTE", "met-2025"),
    "Peter": os.environ.get("EKP_CODE_PETER", "pet-2025"),
}

# Deadline (ingen endelige point før denne dato – men potentiale vises)
COMPETITION_END = date(2025, 10, 26)

# -------------------------- Hjælpefunktioner --------------------------

def ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(NOMINEES_JSON):
        # tom struktur – kan udfyldes i Admin-panelet
        with open(NOMINEES_JSON, "w", encoding="utf-8") as f:
            json.dump({cat: [] for cat in CATEGORIES}, f, ensure_ascii=False, indent=2)
    if not os.path.exists(PREDICTIONS_CSV):
        pd.DataFrame(columns=[
            "participant", "category", "pick1", "pick2", "pick3", "submitted_at"
        ]).to_csv(PREDICTIONS_CSV, index=False)
    if not os.path.exists(SNAPSHOTS_CSV):
        pd.DataFrame(columns=[
            "batch_id", "timestamp", "category", "nominee", "votes"
        ]).to_csv(SNAPSHOTS_CSV, index=False)


def load_nominees() -> Dict[str, List[str]]:
    with open(NOMINEES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_nominees(nominees: Dict[str, List[str]]):
    with open(NOMINEES_JSON, "w", encoding="utf-8") as f:
        json.dump(nominees, f, ensure_ascii=False, indent=2)


def read_predictions() -> pd.DataFrame:
    if os.path.exists(PREDICTIONS_CSV):
        return pd.read_csv(PREDICTIONS_CSV)
    return pd.DataFrame(columns=["participant", "category", "pick1", "pick2", "pick3", "submitted_at"])


def write_predictions(df: pd.DataFrame):
    df.to_csv(PREDICTIONS_CSV, index=False)


def read_snapshots() -> pd.DataFrame:
    if os.path.exists(SNAPSHOTS_CSV):
        df = pd.read_csv(SNAPSHOTS_CSV)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    return pd.DataFrame(columns=["batch_id", "timestamp", "category", "nominee", "votes"])


def write_snapshots(df: pd.DataFrame):
    if not df.empty:
        df = df.sort_values(["timestamp", "batch_id"], ascending=[False, False])
    df.to_csv(SNAPSHOTS_CSV, index=False)


def current_table_from_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Seneste stilling (seneste batch per kategori) som tabel category, nominee, votes."""
    if snapshots.empty:
        return pd.DataFrame(columns=["category", "nominee", "votes"])
    snapshots = snapshots.sort_values(["timestamp"]).copy()
    latest = (
        snapshots
        .groupby("category")
        .tail(1)[["category", "batch_id", "timestamp"]]
        .rename(columns={"batch_id": "latest_batch"})
    )
    merged = snapshots.merge(latest, on=["category", "timestamp"], how="inner")
    cur = merged[["category", "nominee", "votes"]]
    return cur


def ranking_per_category(cur_table: pd.DataFrame) -> Dict[str, List[str]]:
    ranks = {}
    for cat in CATEGORIES:
        dfc = cur_table[cur_table["category"] == cat]
        if dfc.empty:
            ranks[cat] = []
        else:
            dfc = dfc.sort_values(["votes", "nominee"], ascending=[False, True])
            ranks[cat] = dfc["nominee"].tolist()
    return ranks


def compute_potential_points(preds: pd.DataFrame, ranks: Dict[str, List[str]]) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame(columns=["participant", "category", "pick1", "pick2", "pick3", "potential_points"])

    def row_points(row) -> int:
        cat = row["category"]
        top = ranks.get(cat, [])
        pts = 0
        if len(top) >= 1 and row["pick1"] == top[0]:
            pts += 5
        if len(top) >= 2 and row["pick2"] == top[1]:
            pts += 1
        if len(top) >= 3 and row["pick3"] == top[2]:
            pts += 1
        return pts

    df = preds.copy()
    df["potential_points"] = df.apply(row_points, axis=1)
    return df


# -------------------------- UI-Komponenter --------------------------

def sidebar_admin_login() -> bool:
    st.sidebar.subheader("Admin")
    with st.sidebar.expander("Admin-login", expanded=False):
        pwd = st.text_input("Adgangskode", type="password")
        ok = st.button("Log ind")
    if ok:
        st.session_state["is_admin"] = (pwd == ADMIN_PASSWORD)
        if not st.session_state.get("is_admin", False):
            st.sidebar.error("Forkert adgangskode")
    return st.session_state.get("is_admin", False)


def sidebar_participant_login() -> str:
    st.sidebar.subheader("Deltager-login")
    user = st.sidebar.selectbox("Vælg deltager", PARTICIPANTS)
    code = st.sidebar.text_input("Personlig kode", type="password")
    login = st.sidebar.button("Log ind som deltager")
    if login:
        expected = PARTICIPANT_CODES.get(user)
        st.session_state["participant"] = user if code == expected else None
        if st.session_state.get("participant") is None:
            st.sidebar.error("Forkert personlig kode")
    return st.session_state.get("participant")


# -------------------------- Sektioner --------------------------

def section_status_overview(snapshots: pd.DataFrame):
    st.header("📊 Live stilling & grafer")
    cur = current_table_from_snapshots(snapshots)
    if cur.empty:
        st.info("Ingen snapshots endnu. Gå til Admin for at tilføje stemmetal.")
        return

    ranks = ranking_per_category(cur)

    tabs = st.tabs(CATEGORIES)
    for i, cat in enumerate(CATEGORIES):
        with tabs[i]:
            st.subheader(cat)
            cat_df = cur[cur["category"] == cat].copy()
            if cat_df.empty:
                st.write("Ingen data for denne kategori endnu.")
                continue
            disp = cat_df.sort_values(["votes", "nominee"], ascending=[False, True]).reset_index(drop=True)
            disp.index = disp.index + 1
            st.dataframe(disp, use_container_width=True)

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
                p1 = st.selectbox(f"1. plads – {cat}", options, key=f"p1_{cat}")
            with c2:
                p2 = st.selectbox(f"2. plads – {cat}", options, key=f"p2_{cat}")
            with c3:
                p3 = st.selectbox(f"3. plads – {cat}", options, key=f"p3_{cat}")
            if len({p1, p2, p3}) < 3:
                st.error("Vælg tre forskellige kandidater pr. kategori.")
            picks[cat] = (p1, p2, p3)
        submitted = st.form_submit_button("Indsend og lås mine gæt")

    if submitted:
        rows = []
        now = datetime.now().isoformat()
        for cat, (p1, p2, p3) in picks.items():
            if p1 and p2 and p3 and len({p1, p2, p3}) == 3:
                rows.append({
                    "participant": participant,
                    "category": cat,
                    "pick1": p1,
                    "pick2": p2,
                    "pick3": p3,
                    "submitted_at": now,
                })
        if rows:
            new_df = pd.DataFrame(rows)
            out = pd.concat([read_predictions(), new_df], ignore_index=True)
            write_predictions(out)
            st.success("Dine gæt er gemt og låst for de valgte kategorier ✨")
        else:
            st.warning("Intet gemt – tjek at der findes kandidater og at valg er forskellige.")


def section_leaderboard(snapshots: pd.DataFrame):
    st.header("🏆 Leaderboard – potentielle point (live)")
    cur = current_table_from_snapshots(snapshots)
    ranks = ranking_per_category(cur)

    preds = read_predictions()
    preds = preds[preds["participant"].isin(PARTICIPANTS)].copy()

    pts_df = compute_potential_points(preds, ranks)

    if pts_df.empty:
        st.info("Ingen forudsigelser endnu.")
        return

    sums = pts_df.groupby("participant")["potential_points"].sum().reset_index()
    sums = sums.sort_values(["potential_points", "participant"], ascending=[False, True])

    with st.expander("Detaljer pr. kategori og deltager"):
        st.dataframe(pts_df.sort_values(["participant", "category"]).reset_index(drop=True), use_container_width=True)

    st.subheader("Samlet stilling (jo højere potentielle point, jo bedre forudsigelser lige nu)")
    st.dataframe(sums.reset_index(drop=True), use_container_width=True)


def section_admin(nominees: Dict[str, List[str]]):
    st.header("🔧 Admin – kandidater & snapshots")

    # --- Kandidater pr. kategori ---
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
                seen = set()
                cleaned = []
                for a in arr:
                    key = a.lower().replace(" ", "")
                    if key not in seen:
                        cleaned.append(a)
                        seen.add(key)
                nominees[cat] = cleaned
            save_nominees(nominees)
            st.success("Kandidater gemt.")

    st.divider()

    # --- Snapshot indtastning ---
    st.subheader("Tilføj snapshot (ugentlig/ved behov)")
    st.caption("Et snapshot er de aktuelle, samlede stemmetal for ALLE kandidater på et tidspunkt. Hver tilføjelse gemmes som historik.")

    snapshots = read_snapshots()

    snap_date = st.date_input("Dato for snapshot", value=date.today())
    batch_id = f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    entries = []
    valid = True
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
            with cols[idx]:
                val = st.number_input(f"{nom}", min_value=0, step=1, value=0, key=f"{batch_id}_{cat}_{nom}")
                entries.append({
                    "batch_id": batch_id,
                    "timestamp": datetime.combine(snap_date, datetime.now().time()).isoformat(),
                    "category": cat,
                    "nominee": nom,
                    "votes": int(val),
                })

    if st.button("Gem snapshot"):
        if not valid:
            st.error("Kan ikke gemme: manglende kandidater i mindst én kategori.")
        else:
            import pandas as _pd
            df_new = _pd.DataFrame(entries)
            out = _pd.concat([snapshots, df_new], ignore_index=True)
            write_snapshots(out)
            st.success("Snapshot gemt.")

    st.divider()

    # --- Nulstil en deltagers gæt (hvis nødvendigt) ---
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

    # --- Eksport ---
    st.subheader("Eksport")
    st.download_button("Download kandidater (JSON)", data=json.dumps(nominees, ensure_ascii=False, indent=2), file_name="nominees.json")
    st.download_button("Download snapshots (CSV)", data=read_snapshots().to_csv(index=False), file_name="snapshots.csv")
    st.download_button("Download forudsigelser (CSV)", data=read_predictions().to_csv(index=False), file_name="predictions.csv")


# -------------------------- Hovedapp --------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    ensure_storage()
    st.title(APP_TITLE)
    st.caption("Internt hygge-spil til kontoret – data gemmes lokalt i ./data/")

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
