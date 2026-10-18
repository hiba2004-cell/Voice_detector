"""
Streamlit app: register voices from the browser microphone, then test who is speaking.
Run:  streamlit run app.py
"""
import html

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import voice_core as vc

st.set_page_config(page_title="Who is speaking?", page_icon="🎙️", layout="wide")

# ------------------------------------------------------------------ design tokens
INK = "#1B1F3B"
MUTED = "#6B7194"
VIOLET = "#5B3DF5"
VOICE = "#10B981"      # useful information (voice)
NOISE = "#F43F5E"      # noise / silence
AMBER = "#F59E0B"      # unknown voice
PERSON_COLORS = ["#5B3DF5", "#0284C7", "#EA580C", "#DB2777", "#0D9488", "#7C3AED"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
.stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input, .stApp textarea { font-family: 'Manrope', sans-serif; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 4rem; }
section[data-testid="stSidebar"] { border-right: 1px solid #E6E8F5; }
h3 { font-weight: 800 !important; letter-spacing: -0.01em; }

/* hero */
.hero { display: flex; align-items: center; justify-content: space-between; gap: 24px;
  padding: 28px 34px; border-radius: 24px; color: #fff; margin-bottom: 1.25rem;
  background: linear-gradient(120deg, #4426D8 0%, #6D4AFF 55%, #1FA8E8 100%); }
.hero-title { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; }
.hero-sub { margin-top: 10px; font-size: 1.02rem; opacity: .93; max-width: 34rem; line-height: 1.5; }

/* tabs as pills */
.stTabs [data-baseweb="tab-list"] { gap: 10px; border-bottom: none; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs button[data-baseweb="tab"] { background: #fff; border: 1px solid #E6E8F5; border-radius: 999px;
  padding: 8px 22px; height: auto; }
.stTabs button[data-baseweb="tab"] p { font-weight: 700; font-size: 1rem; color: #1B1F3B; }
.stTabs button[data-baseweb="tab"][aria-selected="true"] { background: #5B3DF5; border-color: #5B3DF5; }
.stTabs button[data-baseweb="tab"][aria-selected="true"] p { color: #fff; }
.stButton > button { border-radius: 12px; font-weight: 700; padding: .55rem 1.2rem; }

/* cards */
.card { background: #fff; border: 1px solid #E6E8F5; border-radius: 16px; padding: 16px 20px; }
.stats { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 6px 0 12px; }
.stat-label { display: flex; align-items: center; gap: 8px; color: #6B7194; font-weight: 600; font-size: .92rem; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.stat-value { font-size: 2.5rem; font-weight: 800; line-height: 1.15; letter-spacing: -0.02em; margin-top: 4px; }
.ratio { display: flex; height: 14px; border-radius: 99px; overflow: hidden; background: #ECEEF8; margin-bottom: 8px; }
.ratio span { display: block; height: 100%; }

/* result */
.result { display: flex; align-items: center; gap: 20px; padding: 22px 28px; border-radius: 22px; margin: 8px 0 18px; }
.result.ok { background: linear-gradient(135deg, #047857, #10B981); color: #fff; }
.result.unknown { background: linear-gradient(135deg, #FBBF24, #F59E0B); color: #3B2A00; }
.result.noise { background: linear-gradient(135deg, #BE123C, #F43F5E); color: #fff; }
.avatar { flex: none; width: 66px; height: 66px; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; font-size: 1.9rem; font-weight: 800; background: rgba(255,255,255,.22); }
.result.unknown .avatar { background: rgba(59,42,0,.12); }
.result-body { flex: 1; min-width: 0; }
.result-name { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; }
.result-sub { margin-top: 6px; font-size: .98rem; opacity: .93; line-height: 1.4; }
.result-score { text-align: right; font-size: 2.5rem; font-weight: 800; line-height: 1; }
.result-score small { display: block; font-size: .8rem; font-weight: 600; opacity: .85; margin-top: 6px; }

/* notes, chips, timeline */
.note { border-radius: 12px; padding: 12px 16px; font-weight: 600; margin: 8px 0 12px; }
.note.ok { background: #E7F8F1; color: #047857; }
.note.warn { background: #FFF4DB; color: #8A5300; }
.note.bad { background: #FFE9ED; color: #BE123C; }
.chips { display: flex; flex-wrap: wrap; gap: 10px; margin: 6px 0 14px; }
.chip { display: inline-flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #E6E8F5;
  border-radius: 99px; padding: 6px 16px 6px 6px; font-weight: 700; }
.chip b { width: 28px; height: 28px; border-radius: 50%; color: #fff; display: inline-flex;
  align-items: center; justify-content: center; font-size: .85rem; }
.chip em { font-style: normal; color: #6B7194; font-weight: 600; font-size: .85rem; }
.timeline { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
.seg { flex: 1 1 120px; border-radius: 12px; padding: 10px 12px; display: flex; flex-direction: column; gap: 2px; }
.seg b { font-size: .95rem; }
.seg span { font-size: .78rem; opacity: .9; }

@media (max-width: 640px) {
  .hero svg { display: none; }
  .stats { grid-template-columns: 1fr; }
  .result { flex-wrap: wrap; }
  .result-score { text-align: left; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("rec_key", 0)
ss.setdefault("flash", None)

MIN_SPEECH_SEC = 3.0   # needed to register a clip
MIN_TEST_SEC = 1.0     # needed to test


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading the voice model (first time downloads it)...")
def load_model():
    return vc.get_model()


def md(s: str):
    """Render HTML on one line (avoids Markdown treating indented lines as code)."""
    st.markdown(" ".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


def person_color(name, db):
    names = sorted(db)
    return PERSON_COLORS[names.index(name) % len(PERSON_COLORS)] if name in names else MUTED


def note(kind, text):
    md(f'<div class="note {kind}">{text}</div>')


def waveform_svg():
    hs = [18, 34, 52, 26, 64, 40, 72, 30, 56, 22, 44, 68, 36, 20, 50, 28, 60, 38, 24, 46]
    rects = "".join(
        f'<rect x="{i * 12}" y="{40 - h / 2}" width="6" height="{h}" rx="3" fill="#fff" '
        f'opacity="{0.95 if i % 5 == 2 else 0.4}"/>' for i, h in enumerate(hs))
    return f'<svg viewBox="0 0 240 80" width="240" height="80" aria-hidden="true">{rects}</svg>'


def chips(db):
    items = "".join(
        f'<span class="chip"><b style="background:{person_color(n, db)}">{html.escape(n[:1].upper())}</b>'
        f'{html.escape(n)}<em>{len(v)} clip{"s" if len(v) != 1 else ""}</em></span>'
        for n, v in db.items())
    md(f'<div class="chips">{items}</div>')


def show_ratio(mask):
    """% of useful information (voice) vs % of noise / silence."""
    voice, noise = vc.voice_noise_percent(mask)
    md(f"""
    <div class="stats">
      <div class="card"><div class="stat-label"><i class="dot" style="background:{VOICE}"></i>Useful information (voice)</div>
        <div class="stat-value" style="color:#059669">{voice:.0f}%</div></div>
      <div class="card"><div class="stat-label"><i class="dot" style="background:{NOISE}"></i>Noise / silence</div>
        <div class="stat-value" style="color:#E11D48">{noise:.0f}%</div></div>
    </div>
    <div class="ratio"><span style="width:{voice}%;background:{VOICE}"></span>
      <span style="width:{noise}%;background:{NOISE}"></span></div>
    """)


def result_card(kind, avatar, title, sub, score=None):
    right = f'<div class="result-score">{score}<small>similarity</small></div>' if score else ""
    md(f"""
    <div class="result {kind}">
      <div class="avatar">{avatar}</div>
      <div class="result-body"><div class="result-name">{html.escape(title)}</div>
        <div class="result-sub">{sub}</div></div>
      {right}
    </div>
    """)


def level_chart(levels, mask, thr):
    """Bars per 0.5 s: green = voice, red = noise. Dashed line = speech threshold."""
    if len(levels) == 0:
        return
    lo = float(min(levels.min(), thr)) - 5
    hi = float(max(levels.max(), thr)) + 5
    df = pd.DataFrame({
        "t": [f"{i * vc.CHUNK_SEC:.1f}" for i in range(len(levels))],
        "level": levels, "base": lo,
        "kind": np.where(mask, "Voice", "Noise / silence"),
    })
    bars = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("t:O", title="seconds", sort=None, scale=alt.Scale(paddingInner=0.25),
                axis=alt.Axis(labelAngle=0, labelOverlap=True)),
        y=alt.Y("level:Q", title="level (dB)", scale=alt.Scale(domain=[lo, hi])),
        y2="base:Q",
        color=alt.Color("kind:N", legend=alt.Legend(orient="top", title=None),
                        scale=alt.Scale(domain=["Voice", "Noise / silence"], range=[VOICE, NOISE])),
        tooltip=["t", alt.Tooltip("level:Q", format=".1f"), "kind"],
    )
    rule = alt.Chart(pd.DataFrame({"y": [thr]})).mark_rule(
        color=VIOLET, strokeDash=[6, 4], strokeWidth=2).encode(y="y:Q")
    chart = (bars + rule).properties(height=230).configure_view(strokeWidth=0) \
        .configure_axis(gridColor="#ECEEF8", domain=False, labelColor=MUTED, titleColor=MUTED) \
        .configure_legend(labelColor=INK, labelFontSize=12)
    st.altair_chart(chart, theme=None)
    st.caption("Dashed line = speech threshold. Bars above it count as voice.")


def scores_chart(scores, best, threshold):
    names = list(scores)
    df = pd.DataFrame({"Person": names, "Similarity": [scores[n] for n in names],
                       "kind": ["Best" if n == best else "Other" for n in names]})
    x_min = min(0.0, min(scores.values()))
    base = alt.Chart(df).encode(y=alt.Y("Person:N", sort="-x", title=None,
                                        axis=alt.Axis(labelFontSize=13, labelColor=INK)))
    bars = base.mark_bar(cornerRadiusEnd=8, size=22).encode(
        x=alt.X("Similarity:Q", scale=alt.Scale(domain=[x_min, 1]), title=None),
        color=alt.Color("kind:N", legend=None,
                        scale=alt.Scale(domain=["Best", "Other"], range=[VIOLET, "#C9CCE8"])))
    labels = base.mark_text(align="left", dx=6, fontWeight="bold", color=INK).encode(
        x="Similarity:Q", text=alt.Text("Similarity:Q", format=".2f"))
    rule = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(
        color=AMBER, strokeDash=[6, 4], strokeWidth=2).encode(x="x:Q")
    chart = (bars + labels + rule).properties(height=max(90, 46 * len(df))) \
        .configure_view(strokeWidth=0) \
        .configure_axis(gridColor="#ECEEF8", domain=False, labelColor=MUTED)
    st.altair_chart(chart, theme=None)
    st.caption("Dashed line = match threshold.")


def timeline(rows, db):
    blocks = ""
    for r in rows:
        label = r["Label"]
        if label == "Noise / silence":
            bg, fg = "#FFE4E9", "#BE123C"
        elif label == "Unknown":
            bg, fg = "#FDE7B0", "#7A4B00"
        else:
            bg, fg = person_color(label, db), "#fff"
        sim = "" if r["Similarity"] is None else f' · {r["Similarity"]:.2f}'
        blocks += (f'<div class="seg" style="background:{bg};color:{fg}"><b>{html.escape(label)}</b>'
                   f'<span>{r["From (s)"]:.1f}–{r["To (s)"]:.1f} s{sim}</span></div>')
    md(f'<div class="timeline">{blocks}</div>')


# ------------------------------------------------------------------ sidebar
db = vc.load_db()
with st.sidebar:
    st.header("Settings")
    margin = st.slider("Speech margin above noise (dB)", 3, 25, 10,
                       help="Higher = ignores more background noise")
    abs_db = st.slider("Minimum voice loudness (dB)", -60, -20, -40,
                       help="Anything quieter than this is treated as noise/silence")
    threshold = st.slider("Match threshold", 0.2, 0.9, 0.50, 0.01,
                          help="Higher = stricter, more 'Unknown' results")
    st.divider()
    st.header("Registered people")
    if not db:
        st.caption("Nobody yet. Use the Register voices tab.")
    for person in list(db):
        c1, c2 = st.columns([4, 1])
        with c1:
            md(f'<div class="chip"><b style="background:{person_color(person, db)}">'
               f'{html.escape(person[:1].upper())}</b>{html.escape(person)}</div>')
        if c2.button("🗑", key=f"del_{person}", help=f"Delete {person}"):
            del db[person]
            vc.save_db(db)
            st.rerun()

# ------------------------------------------------------------------ header
md(f"""
<div class="hero">
  <div>
    <div class="hero-title">Who is speaking?</div>
    <div class="hero-sub">Register each voice once, then test any recording.
      Background noise is filtered out before matching.</div>
  </div>
  {waveform_svg()}
</div>
""")

tab_reg, tab_test = st.tabs(["Register voices", "Test a voice"])

# ------------------------------------------------------------------ register
with tab_reg:
    if ss["flash"]:
        note("ok", ss["flash"])
        ss["flash"] = None

    st.write("Type the person's name, record 10-20 seconds of them talking normally, and save. "
             "Repeat **3 or more times** per person for better accuracy.")
    name = st.text_input("Person's name", placeholder="e.g. Hiba").strip()
    audio = st.audio_input("Record", sample_rate=vc.SR, key=f"rec_{ss['rec_key']}")

    if audio is not None:
        wav = vc.read_audio(audio.getvalue())
        levels, mask, thr, floor = vc.detect_speech(wav, margin, abs_db)
        speech_sec = float(mask.sum() * vc.CHUNK_SEC)
        level_chart(levels, mask, thr)
        show_ratio(mask)

        if speech_sec < MIN_SPEECH_SEC:
            note("warn", f"Only {speech_sec:.1f} s of voice detected (need {MIN_SPEECH_SEC:.0f} s). "
                         "Speak closer to the microphone and record again.")
        else:
            note("ok", f"{speech_sec:.1f} s of voice detected, "
                       f"{len(wav) / vc.SR - speech_sec:.1f} s of noise/silence removed.")

        can_save = bool(name) and speech_sec >= MIN_SPEECH_SEC
        if st.button(f"Save recording for {name or '...'}", disabled=not can_save, type="primary"):
            load_model()
            with st.spinner("Learning this voice..."):
                e = vc.embed(vc.chunks_audio(wav, mask))
            db = vc.load_db()
            db.setdefault(name, []).append(e)
            vc.save_db(db)
            ss["rec_key"] += 1          # clears the recorder
            ss["flash"] = (f"Saved clip {len(db[name])} for {html.escape(name)}. "
                           "Record another one or open the Test a voice tab.")
            st.rerun()
        if not name:
            st.caption("Enter a name to enable saving.")

    if db:
        st.subheader("Registered people")
        chips(db)

# ------------------------------------------------------------------ test
with tab_test:
    if not db:
        note("warn", "Register at least one voice in the first tab.")
    else:
        st.write("Press record, say a sentence, then stop. The result appears automatically.")
        test = st.audio_input("Speak now", sample_rate=vc.SR, key="test_audio")

        if test is not None:
            load_model()
            wav = vc.read_audio(test.getvalue())
            levels, mask, thr, floor = vc.detect_speech(wav, margin, abs_db)
            speech_sec = float(mask.sum() * vc.CHUNK_SEC)

            if speech_sec < MIN_TEST_SEC:
                result_card("noise", "🔇", "No voice detected",
                            "This recording is mostly noise or silence. Speak closer to the microphone, "
                            "or lower the minimum voice loudness in the sidebar.")
                st.subheader("Recording")
                level_chart(levels, mask, thr)
                show_ratio(mask)
            else:
                with st.spinner("Identifying..."):
                    e = vc.embed(vc.chunks_audio(wav, mask))
                    scores = vc.score_all(e, db)
                    who, score = vc.decide(scores, threshold)
                    best = max(scores, key=scores.get)

                if who == "Unknown":
                    result_card("unknown", "?", "Unknown voice",
                                f"Closest match: {html.escape(best)} ({score:.2f}), "
                                f"below the {threshold:.2f} threshold.", f"{score:.2f}")
                else:
                    result_card("ok", html.escape(who[:1].upper()), who,
                                f"Matched against {len(db)} registered "
                                f"{'person' if len(db) == 1 else 'people'}.", f"{score:.2f}")

                left, right = st.columns([3, 2], gap="large")
                with left:
                    st.subheader("Recording")
                    level_chart(levels, mask, thr)
                    show_ratio(mask)
                with right:
                    st.subheader("Similarity")
                    scores_chart(scores, best, threshold)

                # who spoke when: 2 s windows, hop 1 s
                rows, n_chunks, win = [], len(mask), 4
                for i in range(0, max(n_chunks - win + 1, 1), 2):
                    j = min(i + win, n_chunks)
                    t0 = i * vc.CHUNK_SEC
                    seg = vc.chunks_audio(wav, mask, i, j)
                    if mask[i:j].mean() >= 0.5 and len(seg) >= vc.SR:
                        w_who, w_score = vc.decide(vc.score_all(vc.embed(seg), db), threshold)
                        rows.append({"From (s)": t0, "To (s)": j * vc.CHUNK_SEC,
                                     "Label": w_who, "Similarity": round(w_score, 2)})
                    else:
                        rows.append({"From (s)": t0, "To (s)": j * vc.CHUNK_SEC,
                                     "Label": "Noise / silence", "Similarity": None})
                if len(rows) > 1:
                    st.subheader("Who spoke when")
                    timeline(rows, db)
