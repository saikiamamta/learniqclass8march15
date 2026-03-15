"""
LearnIQ v2 — CBSE Grade 8 Science AI Tutor
Streamlit Cloud compatible version
"""

import os, json, random, datetime, sqlite3, hashlib
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# ── Streamlit Cloud: read secrets if available ──
if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# ─── CONFIGURATION ─────────────────────────────────────────────────────────
SUBJECT_LABEL = "CBSE Grade 8 Science"
PDF_DIR       = "./pdfs"
FAISS_DIR     = "./faiss_index"
DB_PATH       = "./learniq_analytics.db"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
NUM_CHUNKS    = 6
EMBED_MODEL   = "text-embedding-3-small"
LLM_MODEL     = "gpt-4o-mini"
TEMPERATURE   = 0.6
MAX_TOKENS    = 900
MAX_FOLLOWUPS = 5

CHAPTERS = {
    "hecu101.pdf": "Ch 1 — Crop Production and Management",
    "hecu102.pdf": "Ch 2 — Microorganisms: Friend and Foe",
    "hecu103.pdf": "Ch 3 — Synthetic Fibres and Plastics",
    "hecu104.pdf": "Ch 4 — Materials: Metals and Non-Metals",
    "hecu105.pdf": "Ch 5 — Coal and Petroleum",
    "hecu106.pdf": "Ch 6 — Combustion and Flame",
    "hecu107.pdf": "Ch 7 — Conservation of Plants and Animals",
    "hecu108.pdf": "Ch 8 — Cell — Structure and Functions",
    "hecu109.pdf": "Ch 9 — Reproduction in Animals",
    "hecu110.pdf": "Ch 10 — Reaching the Age of Adolescence",
    "hecu111.pdf": "Ch 11 — Force and Pressure",
    "hecu112.pdf": "Ch 12 — Friction",
    "hecu113.pdf": "Ch 13 — Sound",
    "hecu1cc.pdf": "Ch 14 — Chemical Effects of Electric Current",
    "hecu1ps.pdf": "Ch 15 — Some Natural Phenomena",
}
CHAPTER_LIST = list(CHAPTERS.values())

MOTIVATIONAL_QUOTES = [
    ('"The beautiful thing about learning is that nobody can take it away from you."', '— B.B. King'),
    ('"Education is the most powerful weapon which you can use to change the world."', '— Nelson Mandela'),
    ('"The more that you read, the more things you will know."', '— Dr. Seuss'),
    ('"Curiosity is the engine of achievement."', '— Ken Robinson'),
    ('"In the middle of every difficulty lies opportunity."', '— Albert Einstein'),
]

# ─── DATABASE ──────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT, student_name TEXT,
        timestamp TEXT, mode TEXT, chapter TEXT,
        question TEXT, response_length INTEGER)""")
    c.execute("""CREATE TABLE IF NOT EXISTS quiz_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT, student_name TEXT,
        timestamp TEXT, chapter TEXT,
        score INTEGER, total INTEGER, competency_level TEXT)""")
    conn.commit()
    conn.close()

def log_interaction(sid, sname, mode, chapter, question, rlen):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO interactions VALUES (NULL,?,?,?,?,?,?,?)",
            (sid, sname, datetime.datetime.now().isoformat(), mode, chapter, question[:200], rlen))
        conn.commit()
        conn.close()
    except: pass

def log_quiz(sid, sname, chapter, score, total):
    pct = (score/total)*100 if total else 0
    level = "Not Started" if pct==0 else "Basic" if pct<40 else "Good" if pct<60 else "Proficient" if pct<80 else "Master"
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO quiz_results VALUES (NULL,?,?,?,?,?,?,?)",
            (sid, sname, datetime.datetime.now().isoformat(), chapter, score, total, level))
        conn.commit()
        conn.close()
    except: pass
    return level

def get_competency(sid, chapter):
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT competency_level FROM quiz_results WHERE student_id=? AND chapter=? ORDER BY timestamp DESC LIMIT 1",
            (sid, chapter)).fetchone()
        conn.close()
        return row[0] if row else "Not Started"
    except: return "Not Started"

def get_analytics():
    try:
        conn = sqlite3.connect(DB_PATH)
        data = {
            "total_students": conn.execute("SELECT COUNT(DISTINCT student_id) FROM interactions").fetchone()[0],
            "total_interactions": conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0],
            "chapter_access": conn.execute("SELECT chapter, COUNT(*) FROM interactions WHERE chapter!='' GROUP BY chapter ORDER BY COUNT(*) DESC LIMIT 10").fetchall(),
            "quiz_avg": conn.execute("SELECT chapter, AVG(score*100.0/total), COUNT(*) FROM quiz_results GROUP BY chapter").fetchall(),
            "competency_dist": conn.execute("SELECT competency_level, COUNT(*) FROM quiz_results GROUP BY competency_level").fetchall(),
            "recent_students": conn.execute("SELECT DISTINCT student_name, MAX(timestamp) FROM interactions GROUP BY student_id ORDER BY MAX(timestamp) DESC LIMIT 8").fetchall(),
        }
        conn.close()
        return data
    except:
        return {"total_students":0,"total_interactions":0,"chapter_access":[],"quiz_avg":[],"competency_dist":[],"recent_students":[]}

# ─── VECTOR STORE ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📚 Indexing textbook — one-time setup, please wait...")
def build_retriever():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    faiss_path = Path(FAISS_DIR)

    # Reload from disk if already built
    if faiss_path.exists() and (faiss_path / "index.faiss").exists():
        vs = FAISS.load_local(
            FAISS_DIR, embeddings,
            allow_dangerous_deserialization=True
        )
        return vs.as_retriever(search_kwargs={"k": NUM_CHUNKS})

    # Load PDFs
    pdf_dir = Path(PDF_DIR)
    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True)
        st.error(f"Created '{PDF_DIR}/' — add your PDFs there and restart.")
        st.stop()

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        st.error(f"No PDFs found in '{PDF_DIR}/'. Add chapter PDFs and restart.")
        st.stop()

    raw_pages = []
    prog = st.progress(0, text="Loading PDFs...")
    for i, p in enumerate(pdf_files):
        loader = PyPDFLoader(str(p))
        pages = loader.load()
        for page in pages:
            page.metadata["source_file"] = p.name
        raw_pages.extend(pages)
        prog.progress((i+1)/len(pdf_files), text=f"Loading {p.name}...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(raw_pages)

    # Build FAISS index (no batch limit issues!)
    prog.progress(0.9, text="Building search index...")
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(FAISS_DIR)
    prog.empty()

    return vs.as_retriever(search_kwargs={"k": NUM_CHUNKS})

def get_chapter_badges(docs):
    badges = set()
    for doc in docs:
        src = doc.metadata.get("source_file", "")
        badges.add(CHAPTERS.get(src, src.replace(".pdf","") if src else "Textbook"))
    return sorted(badges)

# ─── LLM ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

@st.cache_resource
def build_qa_chain(_retriever):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are LearnIQ, a warm Socratic science tutor for CBSE Grade 8.

STRICT RULE: Only answer from the context below.
If not found say: "I couldn't find this in your textbook. Please ask your teacher!"

TEACHING SEQUENCE:
1. State the core fact simply (1-2 sentences)
2. Explain WHY/HOW with a simple real-life analogy
3. Give one concrete real-life example
4. Ask ONE simple check question — then STOP and wait

Use bold for key terms. Keep language warm. End with 🌟

Context: {context}
Student question: {question}
Response:"""
    )
    return RetrievalQA.from_chain_type(
        llm=get_llm(), chain_type="stuff", retriever=_retriever,
        return_source_documents=True, chain_type_kwargs={"prompt": prompt}
    )

def llm_call(system, user):
    llm = get_llm()
    return llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user)
    ]).content

# ─── TUTOR MODE ────────────────────────────────────────────────────────────
def page_tutor(qa_chain, sid, sname):
    st.markdown('<div class="mode-card tutor-card"><span class="mode-icon">🎓</span><div><div class="mode-title">Personalized Tutor</div><div class="mode-desc">Ask me anything from your Science textbook!</div></div></div>', unsafe_allow_html=True)

    for k,v in [("msgs",[]),("fup_count",0),("cur_ch","")]:
        if k not in st.session_state: st.session_state[k] = v

    for msg in st.session_state.msgs:
        with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"]=="user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("badges"):
                st.markdown(" ".join(f'<span class="badge">{b}</span>' for b in msg["badges"]), unsafe_allow_html=True)

    user_input = st.chat_input("Type your Science question here... 💬")

    if user_input:
        sci_words = ["what","why","how","explain","define","cell","force","sound","crop",
                     "micro","fibre","metal","coal","fire","plant","animal","friction",
                     "electric","reproduce","adolescence","chapter","science","atom","energy",
                     "heat","light","water","soil","food","air","pressure","wave","current"]
        is_offtopic = not any(w in user_input.lower() for w in sci_words) and len(user_input.split()) < 5

        if is_offtopic:
            q, a = random.choice(MOTIVATIONAL_QUOTES)
            reply = f"{q}\n*{a}*\n\n😄 Looks like a little detour! That's okay — but your Science book has amazing things waiting! What topic shall we explore? 🚀"
            st.session_state.msgs.append({"role":"user","content":user_input})
            st.session_state.msgs.append({"role":"assistant","content":reply,"badges":[]})
            st.rerun()
            return

        if st.session_state.fup_count >= MAX_FOLLOWUPS:
            reply = "😊 Great questions today! We've explored this topic deeply. For remaining doubts, please ask your teacher — they'll love your curiosity! 👩‍🏫\n\nShall we explore a **new topic**? 🌟"
            st.session_state.fup_count = 0
            st.session_state.msgs.append({"role":"user","content":user_input})
            st.session_state.msgs.append({"role":"assistant","content":reply,"badges":[]})
            st.rerun()
            return

        with st.spinner("🤔 Thinking..."):
            # Build conversation history
            history_text = ""
            for m in st.session_state.msgs[-6:]:
                role = "Student" if m["role"] == "user" else "Tutor"
                history_text += f"{role}: {m['content'][:300]}\n\n"

            # Retrieve context
            docs = qa_chain.retriever.invoke(user_input)
            context = "\n\n".join([d.page_content for d in docs]) if docs else ""
            badges = get_chapter_badges(docs)

            # Check if student is answering tutor's question
            is_answering = (len(st.session_state.msgs) > 0 and
                           st.session_state.msgs[-1]["role"] == "assistant" and
                           "?" in st.session_state.msgs[-1]["content"])

            if is_answering:
                system_p = """You are LearnIQ, a warm Socratic science tutor for CBSE Grade 8.
The student just answered your check question. Do this:
1. ✅ or ❌ — Tell them clearly if correct or not
2. If wrong: gently correct with the right answer and simple explanation
3. If right: praise warmly and add one interesting extra fact
4. Give a REAL-LIFE example connecting to this concept
5. Ask ONE new build-up question to go deeper — then STOP and wait
Keep it warm, encouraging, age-appropriate. Bold key terms. End with 🌟"""
                user_p = f"Conversation:\n{history_text}\nStudent's answer: {user_input}\n\nTextbook context:\n{context}\n\nEvaluate and continue:"
            else:
                system_p = """You are LearnIQ, a warm Socratic science tutor for CBSE Grade 8.
STRICT: Only answer from textbook context. If not found say so.
SEQUENCE: 1) Core fact 2) WHY/HOW with analogy 3) Real-life example 4) ONE check question then STOP.
Bold key terms. Warm language. End with 🌟"""
                user_p = f"Textbook context:\n{context}\n\nStudent question: {user_input}\n\nResponse:"

            answer = llm_call(system_p, user_p)

        if st.session_state.cur_ch and badges and any(b in st.session_state.cur_ch for b in badges):
            st.session_state.fup_count += 1
        else:
            st.session_state.fup_count = 1
            st.session_state.cur_ch = badges[0] if badges else ""

        log_interaction(sid, sname, "Tutor", badges[0] if badges else "General", user_input, len(answer))
        st.session_state.msgs.append({"role":"user","content":user_input})
        st.session_state.msgs.append({"role":"assistant","content":answer,"badges":badges})
        st.rerun()

    if st.session_state.msgs:
        if st.button("🔄 Start New Topic"):
            st.session_state.msgs = []
            st.session_state.fup_count = 0
            st.rerun()

# ─── SUMMARY MODE ──────────────────────────────────────────────────────────
def page_summary(retriever, sid, sname):
    st.markdown('<div class="mode-card summary-card-hdr"><span class="mode-icon">📋</span><div><div class="mode-title">Quick Summary Master</div><div class="mode-desc">One-page chapter summary with key concepts & exam tips</div></div></div>', unsafe_allow_html=True)

    chapter = st.selectbox("📚 Select Chapter", CHAPTER_LIST, key="sum_ch")

    if st.button("📄 Generate Summary", use_container_width=True):
        with st.spinner("📝 Building your chapter summary..."):
            docs = retriever.invoke(f"{chapter} concepts definitions facts")
            context = "\n\n".join([d.page_content for d in docs]) if docs else "No context found."

            summary = llm_call(
                "You are a textbook summarizer for CBSE Grade 8 Science. Be factual and structured. No Bloom's Taxonomy. No questions. No tutoring. Only summarize.",
                f"""Write a one-page summary of "{chapter}" using this textbook context.

## 📚 {chapter}

### 🔍 What This Chapter Is About
(2-3 sentence overview)

### 📌 Key Concepts
(6-8 bullet points with one-line explanations)

### 📖 Important Definitions
(5-6 key terms and definitions)

### ⚡ Must-Remember Facts
(5-6 important exam facts)

### 🌍 Real-Life Examples
(2-3 everyday examples)

### ⚠️ Common Mistakes to Avoid
(3-4 misconceptions with correct explanations)

### 📝 Exam Tips
(2-3 tips)

Context: {context}"""
            )

        log_interaction(sid, sname, "Summary", chapter, "summary", len(summary))
        st.markdown(summary)
        if docs:
            st.markdown(" ".join(f'<span class="badge">{b}</span>' for b in get_chapter_badges(docs)), unsafe_allow_html=True)
        st.download_button("⬇️ Download Summary", data=summary,
            file_name=f"Summary_{chapter[:25].replace(' ','_')}.md",
            mime="text/markdown", use_container_width=True)

# ─── PROJECTS MODE ─────────────────────────────────────────────────────────
def page_projects(qa_chain, sid, sname):
    st.markdown('<div class="mode-card projects-card-hdr"><span class="mode-icon">🔬</span><div><div class="mode-title">Practical Projects Master</div><div class="mode-desc">Safe, fun experiments to master concepts</div></div></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    chapter  = c1.selectbox("📚 Chapter", CHAPTER_LIST, key="proj_ch")
    location = c2.selectbox("📍 Where?", ["Home","School","Both"], key="proj_loc")

    if st.button("🚀 Generate Projects", use_container_width=True):
        with st.spinner("🔭 Creating project ideas..."):
            result = qa_chain.invoke({"query": f"Main concepts and activities in {chapter}"})
            context = result["result"]
            projects = llm_call(
                "You create safe practical science projects for Class 8 students in India.",
                f"""Create 3 safe projects for "{chapter}" doable at {location}.

For EACH project:
### 🔬 Project [N]: [Name]
**Concept:** (chapter link)
**🎯 You will learn:** (1-2 sentences)
**🧰 Materials:** (simple, safe, available in India)
**📋 Steps:** (5-7 numbered steps)
**👀 Observe:** (what to notice)
**💡 Science behind it:** (textbook link)
**🌟 Extension:** (creative twist)

Context: {context}"""
            )
        log_interaction(sid, sname, "Projects", chapter, "projects", len(projects))
        st.markdown(projects)
        st.download_button("⬇️ Download Projects", data=projects,
            file_name=f"Projects_{chapter[:25].replace(' ','_')}.md",
            mime="text/markdown", use_container_width=True)

# ─── QUIZ MODE ─────────────────────────────────────────────────────────────
def page_quiz(sid, sname):
    st.markdown('<div class="mode-card quiz-card-hdr"><span class="mode-icon">🏆</span><div><div class="mode-title">Quiz Master</div><div class="mode-desc">6 questions across Bloom\'s Taxonomy levels</div></div></div>', unsafe_allow_html=True)

    for k,v in [("qstate","select"),("qqns",[]),("qans",{}),("qch",""),("qres",None)]:
        if k not in st.session_state: st.session_state[k] = v

    if st.session_state.qstate == "select":
        chapter = st.selectbox("📚 Choose Chapter", CHAPTER_LIST, key="quiz_ch_sel")
        level_icons = {"Not Started":"⚪","Basic":"🔴","Good":"🟡","Proficient":"🟢","Master":"🏆"}
        comp = get_competency(sid, chapter)
        st.info(f"{level_icons.get(comp,'⚪')} Your current level: **{comp}**")

        if st.button("🎯 Start Quiz!", use_container_width=True):
            with st.spinner("🎲 Generating quiz..."):
                raw = llm_call(
                    "Return ONLY valid JSON. No markdown. No extra text.",
                    f"""Create 6 quiz questions for "{chapter}" (CBSE Grade 8).
One per Bloom's level: Remember, Understand, Apply, Analyse, Evaluate, Create.
JSON format:
{{"questions":[{{"level":"Remember","question":"...","options":["A) ...","B) ...","C) ...","D) ..."],"correct":"A","explanation":"..."}}]}}
Vary correct answers. Clear questions for Class 8."""
                )
            try:
                clean = raw.strip().replace("```json","").replace("```","").strip()
                data = json.loads(clean)
                st.session_state.qqns = data["questions"]
                st.session_state.qans = {}
                st.session_state.qch  = chapter
                st.session_state.qstate = "active"
                st.rerun()
            except Exception as e:
                st.error(f"Quiz generation failed, please try again. ({e})")

    elif st.session_state.qstate == "active":
        st.markdown(f"### 📝 {st.session_state.qch}")
        bloom_icons = {"Remember":"🔵","Understand":"🟢","Apply":"🟡","Analyse":"🟠","Evaluate":"🔴","Create":"🟣"}

        with st.form("quiz_form"):
            for i, q in enumerate(st.session_state.qqns):
                icon = bloom_icons.get(q.get("level",""), "⚪")
                st.markdown(f'<div class="qbox"><span class="bloom-pill">{icon} {q.get("level","")}</span><br><strong>Q{i+1}. {q["question"]}</strong></div>', unsafe_allow_html=True)
                st.session_state.qans[i] = st.radio(
                    f"q{i}", q.get("options",[]),
                    key=f"qr_{i}", index=None,
                    label_visibility="collapsed")
                st.markdown("---")
            submitted = st.form_submit_button("📊 Submit Answers", use_container_width=True)

        if submitted:
            unanswered = [i+1 for i in range(len(st.session_state.qqns)) if not st.session_state.qans.get(i)]
            if unanswered:
                st.warning(f"Please answer all questions! Missing: Q{', Q'.join(map(str,unanswered))}")
            else:
                score = 0
                results = []
                for i, q in enumerate(st.session_state.qqns):
                    chosen = st.session_state.qans.get(i,"")
                    cl = q.get("correct","A")
                    ok = chosen.startswith(cl + ")")
                    if ok: score += 1
                    results.append({"q":q["question"],"level":q.get("level",""),
                        "chosen":chosen,
                        "correct":next((o for o in q["options"] if o.startswith(cl+")")),chosen),
                        "explanation":q.get("explanation",""),"ok":ok})
                comp = log_quiz(sid, sname, st.session_state.qch, score, 6)
                log_interaction(sid, sname, "Quiz", st.session_state.qch, f"score:{score}/6", score)
                st.session_state.qres = {"score":score,"results":results,"comp":comp}
                st.session_state.qstate = "results"
                st.rerun()

        if st.button("❌ Cancel"):
            st.session_state.qstate = "select"
            st.rerun()

    elif st.session_state.qstate == "results":
        d = st.session_state.qres
        pct = int((d["score"]/6)*100)
        col = "#dc2626" if pct<40 else "#d97706" if pct<70 else "#16a34a"
        msgs = {100:"🏆 PERFECT! Absolute master!",80:"🌟 Outstanding!",60:"👍 Good job!",40:"💪 Keep going!",0:"😊 Don't worry, try again!"}
        msg = next(v for k,v in sorted(msgs.items(),reverse=True) if pct>=k)

        st.markdown(f"""
        <div style="background:white;border:3px solid {col};border-radius:20px;padding:2rem;text-align:center;margin:1rem 0">
            <div style="font-size:3rem;font-weight:900;color:{col}">{d['score']}/6</div>
            <div style="font-size:1.1rem;font-weight:700;color:#1a1a2e">{pct}% — {d['comp']}</div>
            <div style="color:#475569;font-size:0.85rem">{st.session_state.qch}</div>
        </div>
        <div style="background:#fefce8;border-left:5px solid #f59e0b;padding:12px 16px;border-radius:0 10px 10px 0;color:#78350f;font-weight:700;margin-bottom:1rem">{msg}</div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 Feedback")
        for i, r in enumerate(d["results"]):
            bg = "#f0fdf4" if r["ok"] else "#fef2f2"
            icon = "✅" if r["ok"] else "❌"
            correct_line = "" if r["ok"] else f"<br>✔️ <strong>Correct:</strong> {r['correct']}"
            st.markdown(f'<div style="background:{bg};border-radius:10px;padding:12px 16px;margin-bottom:8px;color:#1a1a2e;border:1px solid #e2e8f0">{icon} <strong>Q{i+1} ({r["level"]}):</strong> {r["q"]}<br>Your answer: <em>{r["chosen"]}</em>{correct_line}<br>💡 {r["explanation"]}</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        if c1.button("🔄 Retake Quiz", use_container_width=True):
            st.session_state.qstate = "select"
            st.rerun()
        if c2.button("📋 Study Chapter", use_container_width=True):
            st.session_state.qstate = "select"
            st.session_state.active_mode = "📋 Summary Master"
            st.rerun()

# ─── TEACHER DASHBOARD ────────────────────────────────────────────────────
def page_teacher():
    st.markdown('<div class="mode-card teacher-card-hdr"><span class="mode-icon">👩‍🏫</span><div><div class="mode-title">Teacher Dashboard</div><div class="mode-desc">Class analytics & student progress</div></div></div>', unsafe_allow_html=True)

    if "t_auth" not in st.session_state: st.session_state.t_auth = False
    if not st.session_state.t_auth:
        pwd = st.text_input("🔐 Teacher Password", type="password")
        if st.button("Login"):
            if hashlib.md5(pwd.encode()).hexdigest() == "4a8a08f09d37b73795649038408b5f33":
                st.session_state.t_auth = True
                st.rerun()
            else:
                st.error("Wrong password. Default: abc123")
        st.caption("Default password: abc123")
        return

    data = get_analytics()
    c1,c2,c3 = st.columns(3)
    c1.metric("👥 Students", data["total_students"])
    c2.metric("💬 Interactions", data["total_interactions"])
    c3.metric("📚 Chapters", len(data["chapter_access"]))
    st.markdown("---")

    if data["chapter_access"]:
        st.markdown("#### 📚 Most Accessed Chapters")
        max_c = max(c for _,c in data["chapter_access"]) or 1
        for ch, cnt in data["chapter_access"]:
            w = int((cnt/max_c)*100)
            st.markdown(f'<div style="color:#1e3a5f;font-weight:600;font-size:0.85rem">{ch or "General"} <span style="color:#0369a1">({cnt})</span></div><div style="background:#e2e8f0;border-radius:6px;height:12px;margin-bottom:6px"><div style="background:#3b82f6;width:{w}%;height:12px;border-radius:6px"></div></div>', unsafe_allow_html=True)

    if data["quiz_avg"]:
        st.markdown("#### 🏆 Quiz Performance")
        for ch, avg, attempts in data["quiz_avg"]:
            if avg:
                col = "#dc2626" if avg<40 else "#d97706" if avg<70 else "#16a34a"
                st.markdown(f'<div style="color:#1e3a5f;font-weight:600;font-size:0.85rem">{ch or "Unknown"} <span style="color:{col}">{avg:.0f}% avg ({attempts} attempts)</span></div><div style="background:#e2e8f0;border-radius:6px;height:12px;margin-bottom:6px"><div style="background:{col};width:{min(avg,100):.0f}%;height:12px;border-radius:6px"></div></div>', unsafe_allow_html=True)

    if data["recent_students"]:
        st.markdown("#### 👤 Recent Students")
        for name, last in data["recent_students"]:
            st.markdown(f"**{name}** · Last active: {str(last)[:16]}")

    if st.button("🔒 Logout"):
        st.session_state.t_auth = False
        st.rerun()

# ─── MAIN ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="LearnIQ", page_icon="🔬",
                       layout="wide", initial_sidebar_state="expanded")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@700;800&family=Nunito:wght@400;600;700&display=swap');

    html, body, .stApp { background:#f1f5f9 !important; }
    .stApp { font-family:'Nunito',sans-serif !important; }
    .main .block-container { background:#f1f5f9 !important; max-width:900px; }

    .stMarkdown p, .stMarkdown li, .stMarkdown h1,
    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    p, li { color:#1e293b !important; }

    [data-testid="stChatMessage"] {
        background:#ffffff !important; border:1px solid #cbd5e1 !important;
        border-radius:14px !important; margin-bottom:8px !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li { color:#1e293b !important; }

    .mode-card { display:flex; align-items:center; gap:1rem;
        padding:1rem 1.5rem; border-radius:16px; margin-bottom:1.5rem; }
    .mode-icon  { font-size:2rem; }
    .mode-title { font-family:'Baloo 2',cursive; font-size:1.5rem; font-weight:800; }
    .mode-desc  { font-size:0.85rem; font-weight:600; margin-top:2px; }
    .tutor-card        { background:#1e3a5f; }
    .tutor-card .mode-title, .tutor-card .mode-desc { color:#bfdbfe !important; }
    .summary-card-hdr  { background:#14532d; }
    .summary-card-hdr .mode-title, .summary-card-hdr .mode-desc { color:#bbf7d0 !important; }
    .projects-card-hdr { background:#78350f; }
    .projects-card-hdr .mode-title, .projects-card-hdr .mode-desc { color:#fde68a !important; }
    .quiz-card-hdr     { background:#3b0764; }
    .quiz-card-hdr .mode-title, .quiz-card-hdr .mode-desc { color:#ede9fe !important; }
    .teacher-card-hdr  { background:#7f1d1d; }
    .teacher-card-hdr .mode-title, .teacher-card-hdr .mode-desc { color:#fee2e2 !important; }

    .badge { display:inline-block; background:#1e3a5f; color:#bfdbfe !important;
        font-size:0.72rem; font-weight:700; padding:3px 10px;
        border-radius:20px; margin:3px 3px 0 0; }

    .qbox { background:#ffffff; border:2px solid #e2e8f0; border-radius:12px;
        padding:14px 18px; margin-bottom:10px; color:#1e293b !important; }
    .bloom-pill { display:inline-block; background:#1e3a5f; color:#bfdbfe !important;
        border-radius:8px; padding:2px 10px; font-size:0.75rem;
        font-weight:700; margin-bottom:8px; }

    .stButton > button {
        background:#1e3a5f !important; color:#ffffff !important;
        border:none !important; border-radius:10px !important;
        font-weight:700 !important; width:100% !important; cursor:pointer !important;
    }
    .stButton > button:hover { background:#0f3460 !important; }

    .stSelectbox label { color:#1e293b !important; font-weight:700 !important; }
    .stSelectbox [data-baseweb="select"] > div {
        background:#ffffff !important; color:#1e293b !important;
        border:2px solid #cbd5e1 !important; border-radius:10px !important; }

    section[data-testid="stSidebar"] { background:#0f172a !important; }
    section[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
    section[data-testid="stSidebar"] h3 { color:#7dd3fc !important; }
    section[data-testid="stSidebar"] input {
        background:#1e293b !important; color:#ffffff !important;
        border:1px solid #334155 !important; border-radius:8px !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background:#1e293b !important; color:#ffffff !important;
        border:1px solid #334155 !important; }
    section[data-testid="stSidebar"] .stButton > button {
        background:#1e293b !important; color:#e2e8f0 !important;
        border:1px solid #334155 !important; border-radius:10px !important; }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background:#334155 !important; color:#7dd3fc !important; }
    </style>
    """, unsafe_allow_html=True)

    init_db()

    for k,v in [("logged_in",False),("student_name",""),
                ("student_id",""),("active_mode","🎓 Tutor Mode")]:
        if k not in st.session_state: st.session_state[k] = v

    # ── SIDEBAR ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 0.5rem">
            <span style="font-family:'Baloo 2',cursive;font-size:2rem;font-weight:800;color:#ffffff">
                🔬 Learn<span style="color:#fbbf24">IQ</span>
            </span><br>
            <span style="color:#94a3b8;font-size:0.8rem">Grade 8 Science</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        if not st.session_state.logged_in:
            st.markdown("### 👤 Who's learning?")
            name  = st.text_input("Your Name", placeholder="e.g. Arjun Sharma")
            grade = st.selectbox("Class", ["Class 8","Class 7","Class 9"])
            if st.button("🚀 Start Learning!"):
                if name.strip():
                    st.session_state.student_name = name.strip()
                    st.session_state.student_id = hashlib.md5(
                        f"{name.strip()}{grade}".encode()).hexdigest()[:10]
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.warning("Please enter your name!")
        else:
            st.markdown(f"""
            <div style="background:#1e293b;border-radius:12px;padding:12px;margin-bottom:8px;border:1px solid #334155">
                <div style="font-family:'Baloo 2',cursive;font-size:1rem;color:#7dd3fc;font-weight:700">
                    👋 {st.session_state.student_name}
                </div>
                <div style="font-size:0.75rem;color:#94a3b8">ID: {st.session_state.student_id}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔄 Switch Student"):
                st.session_state.logged_in = False
                st.rerun()

        st.markdown("---")
        st.markdown("### 📚 Modes")

        for m in ["🎓 Tutor Mode","📋 Summary Master","🔬 Projects Master",
                  "🏆 Quiz Master","👩‍🏫 Teacher Dashboard"]:
            if st.button(m, key=f"nav_{m}"):
                st.session_state.active_mode = m
                st.rerun()

        st.markdown(f"""
        <div style="background:#1e3a5f;border-radius:8px;padding:6px 12px;margin-top:8px;
                    color:#7dd3fc;font-size:0.8rem;font-weight:700;border:1px solid #2563eb">
            ✅ {st.session_state.active_mode}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption(f"Model: {LLM_MODEL}")
        st.caption("~$0.0005 per question")

    # ── LANDING ────────────────────────────────────────────────────────────
    if not st.session_state.logged_in:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1e3a5f,#0f172a);border-radius:20px;
                    padding:2rem;margin-bottom:2rem">
            <div style="font-family:'Baloo 2',cursive;font-size:2.2rem;font-weight:800;color:#ffffff">
                🔬 Learn<span style="color:#fbbf24">IQ</span>
                <span style="color:#7dd3fc"> — Grade 8 Science</span>
            </div>
            <div style="color:#cbd5e1;margin-top:6px">
                Your personalized AI Science tutor · Powered by your NCERT textbook
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        for col, icon, title, desc, bg in zip(
            [c1,c2,c3,c4], ["🎓","📋","🔬","🏆"],
            ["Tutor Mode","Quick Summary","Projects","Quiz Master"],
            ["Socratic Q&A","Chapter summaries","Safe experiments","Bloom's quizzes"],
            ["#1e3a5f","#14532d","#78350f","#3b0764"]
        ):
            col.markdown(f"""
            <div style="background:{bg};border-radius:16px;padding:1.2rem;text-align:center">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-family:'Baloo 2',cursive;font-weight:800;color:#ffffff;margin:6px 0">{title}</div>
                <div style="font-size:0.8rem;color:#cbd5e1">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="text-align:center;margin-top:2rem;color:#475569;font-weight:600">👈 Enter your name in the sidebar to start!</div>', unsafe_allow_html=True)
        return

    # ── APP HEADER ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1e3a5f,#0f172a);border-radius:16px;
                padding:1.2rem 1.8rem;margin-bottom:1.5rem">
        <span style="font-family:'Baloo 2',cursive;font-size:1.8rem;font-weight:800;color:#ffffff">
            🔬 Learn<span style="color:#fbbf24">IQ</span>
        </span>
        <span style="color:#7dd3fc;font-size:1.2rem;font-weight:700"> — Grade 8 Science</span>
        <div style="color:#cbd5e1;font-size:0.9rem;margin-top:4px">
            Welcome back, <strong style="color:#fbbf24">{st.session_state.student_name}</strong>! 🌟
        </div>
    </div>
    """, unsafe_allow_html=True)

    retriever = build_retriever()
    qa_chain  = build_qa_chain(retriever)

    mode = st.session_state.active_mode
    sid, sname = st.session_state.student_id, st.session_state.student_name

    if   mode == "🎓 Tutor Mode":          page_tutor(qa_chain, sid, sname)
    elif mode == "📋 Summary Master":      page_summary(retriever, sid, sname)
    elif mode == "🔬 Projects Master":     page_projects(qa_chain, sid, sname)
    elif mode == "🏆 Quiz Master":         page_quiz(sid, sname)
    elif mode == "👩‍🏫 Teacher Dashboard":  page_teacher()

if __name__ == "__main__":
    main()
