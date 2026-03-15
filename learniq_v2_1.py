"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     LearnIQ v2 — Personalized AI Tutor · CBSE Grade 8 Science              ║
║     Bloom's Taxonomy + PAIRS Framework + Socratic Pedagogy                 ║
║     Modes: Tutor | Summary | Projects | Quiz | Teacher Dashboard           ║
╚══════════════════════════════════════════════════════════════════════════════╝

SETUP:
  1. pip install -r requirements.txt
  2. Create .env file:  OPENAI_API_KEY=sk-...
  3. Place all chapter PDFs in ./pdfs/ folder
  4. streamlit run learniq_v2.py

🔁 SWAP SUBJECT/GRADE — change this ONE variable (line ~85):
     SUBJECT_LABEL = "CBSE Grade 8 Science"

⚑ OS-SPECIFIC API KEY:
   macOS/Linux:  export OPENAI_API_KEY="sk-..."
   Windows CMD:  set OPENAI_API_KEY=sk-...
   PowerShell:   $env:OPENAI_API_KEY="sk-..."
   .env file:    OPENAI_API_KEY=sk-...   (recommended — used here)

⚑ SWAP ChromaDB → Pinecone (2 lines — see bottom of file)
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — IMPORTS & ENV
# ─────────────────────────────────────────────────────────────────────────────
import os, json, random, datetime, sqlite3, hashlib
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()   # reads .env file — must be before any OpenAI calls

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CONFIGURATION  (🔁 change these for a different subject/grade)
# ─────────────────────────────────────────────────────────────────────────────

# 🔁 CHECKPOINT — SWAP SUBJECT/GRADE: change this ONE variable
SUBJECT_LABEL   = "CBSE Grade 8 Science"

PDF_DIR         = "./pdfs"                    # folder containing all chapter PDFs
CHROMA_DIR      = "./chroma_db_v2"            # persisted vector store
DB_PATH         = "./learniq_analytics.db"    # SQLite analytics database
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
NUM_CHUNKS      = 4
EMBED_MODEL     = "text-embedding-3-small"
LLM_MODEL       = "gpt-4o-mini"
TEMPERATURE     = 0.6                         # warm, teacher-like tone
MAX_TOKENS      = 900
MAX_FOLLOWUPS   = 5                           # max follow-up questions before deferring to teacher

# Chapter mapping for badges (update if your PDFs cover different chapters)
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

# Motivational quotes for off-topic questions
MOTIVATIONAL_QUOTES = [
    ("\"The beautiful thing about learning is that nobody can take it away from you.\"", "— B.B. King"),
    ("\"Education is the most powerful weapon which you can use to change the world.\"", "— Nelson Mandela"),
    ("\"The more that you read, the more things you will know.\"", "— Dr. Seuss"),
    ("\"Curiosity is the engine of achievement.\"", "— Ken Robinson"),
    ("\"In the middle of every difficulty lies opportunity.\"", "— Albert Einstein"),
]

# Student competency levels for quiz tracking
COMPETENCY_LEVELS = ["Not Started", "Basic", "Good", "Proficient", "Master"]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SQLITE ANALYTICS DATABASE
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    """Create analytics tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Student sessions
    c.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT, student_name TEXT,
        start_time TEXT, end_time TEXT,
        mode TEXT, chapter TEXT
    )""")

    # Individual interactions
    c.execute("""CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT, student_name TEXT,
        timestamp TEXT, mode TEXT,
        chapter TEXT, question TEXT,
        response_length INTEGER, time_spent_sec INTEGER
    )""")

    # Quiz results per student per chapter
    c.execute("""CREATE TABLE IF NOT EXISTS quiz_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT, student_name TEXT,
        timestamp TEXT, chapter TEXT,
        score INTEGER, total INTEGER,
        competency_level TEXT
    )""")

    # Concept difficulty tracker
    c.execute("""CREATE TABLE IF NOT EXISTS concept_difficulty (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT, chapter TEXT,
        concept TEXT, access_count INTEGER,
        avg_followups REAL, timestamp TEXT
    )""")

    conn.commit()
    conn.close()


def log_interaction(student_id, student_name, mode, chapter, question, response_len):
    """Log every interaction for teacher dashboard."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO interactions
        (student_id, student_name, timestamp, mode, chapter, question, response_length, time_spent_sec)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (student_id, student_name, datetime.datetime.now().isoformat(),
         mode, chapter, question[:200], response_len, 0))
    conn.commit()
    conn.close()


def log_quiz_result(student_id, student_name, chapter, score, total):
    """Log quiz score and update competency level."""
    pct = (score / total) * 100 if total > 0 else 0
    if pct == 0:        level = "Not Started"
    elif pct < 40:      level = "Basic"
    elif pct < 60:      level = "Good"
    elif pct < 80:      level = "Proficient"
    else:               level = "Master"

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO quiz_results
        (student_id, student_name, timestamp, chapter, score, total, competency_level)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (student_id, student_name, datetime.datetime.now().isoformat(),
         chapter, score, total, level))
    conn.commit()
    conn.close()
    return level


def get_student_competency(student_id, chapter):
    """Get latest competency level for a student-chapter pair."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("""SELECT competency_level FROM quiz_results
        WHERE student_id=? AND chapter=?
        ORDER BY timestamp DESC LIMIT 1""", (student_id, chapter)).fetchone()
    conn.close()
    return row[0] if row else "Not Started"


def get_teacher_analytics():
    """Fetch aggregated analytics for teacher dashboard."""
    conn = sqlite3.connect(DB_PATH)
    data = {}

    data["total_students"] = conn.execute(
        "SELECT COUNT(DISTINCT student_id) FROM interactions").fetchone()[0]

    data["total_interactions"] = conn.execute(
        "SELECT COUNT(*) FROM interactions").fetchone()[0]

    data["chapter_access"] = conn.execute("""
        SELECT chapter, COUNT(*) as cnt FROM interactions
        WHERE chapter != '' GROUP BY chapter ORDER BY cnt DESC""").fetchall()

    data["quiz_avg"] = conn.execute("""
        SELECT chapter, AVG(score*100.0/total) as avg_score, COUNT(*) as attempts
        FROM quiz_results GROUP BY chapter ORDER BY avg_score""").fetchall()

    data["competency_dist"] = conn.execute("""
        SELECT competency_level, COUNT(*) FROM quiz_results
        GROUP BY competency_level""").fetchall()

    data["recent_students"] = conn.execute("""
        SELECT DISTINCT student_name, student_id,
        MAX(timestamp) as last_seen FROM interactions
        GROUP BY student_id ORDER BY last_seen DESC LIMIT 10""").fetchall()

    conn.close()
    return data


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PDF LOADING & VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="📚 Indexing your textbook — one-time setup, please wait...")
def build_retriever():
    """Load PDFs, chunk, embed, persist ChromaDB. Reloads from disk on restart."""
    chroma_path = Path(CHROMA_DIR)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    if chroma_path.exists() and any(chroma_path.iterdir()):
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": NUM_CHUNKS})

    # Load all PDFs from ./pdfs folder
    pdf_dir = Path(PDF_DIR)
    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True)
        st.error(f"Created '{PDF_DIR}/' folder. Please add your chapter PDFs there and restart.")
        st.stop()

    raw_pages = []
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        st.error(f"No PDFs found in '{PDF_DIR}/'. Add your chapter PDFs and restart.")
        st.stop()

    progress = st.progress(0, text="Loading PDFs...")
    for i, pdf_path in enumerate(pdf_files):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        # Tag each page with its source filename for chapter badge
        for page in pages:
            page.metadata["source_file"] = pdf_path.name
        raw_pages.extend(pages)
        progress.progress((i + 1) / len(pdf_files), text=f"Loading {pdf_path.name}...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(raw_pages)

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR,
    )
    progress.empty()
    return vectorstore.as_retriever(search_kwargs={"k": NUM_CHUNKS})


def get_chapter_from_docs(source_docs):
    """Extract chapter name(s) from retrieved source documents."""
    badges = set()
    for doc in source_docs:
        src = doc.metadata.get("source_file", "")
        chapter = CHAPTERS.get(src, src.replace(".pdf", "").upper() if src else "Textbook")
        badges.add(chapter)
    return sorted(badges)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LLM CHAINS & PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)


@st.cache_resource(show_spinner=False)
def build_qa_chain(_retriever):
    """RetrievalQA chain for document-grounded answers."""
    tutor_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are LearnIQ — a warm, encouraging, Socratic science tutor for CBSE Grade 8 students.

STRICT RULE: Only answer from the retrieved context below.
If not found in context, say: "I couldn't find this in your textbook. Please ask your teacher!"
Never invent facts.

TEACHING METHOD — follow this sequence every time:
1. ASSESS: Gauge the student's level from how they asked the question.
2. REMEMBER: Give the core fact in 1-2 simple sentences.
3. UNDERSTAND: Explain WHY/HOW using a fun real-life analogy.
4. APPLY: Give one real-world example a Class 8 student can relate to.
5. CHECK: Ask ONE simple follow-up question to check understanding. Then STOP and wait.
6. Always end with an emoji and a motivating line like "You're doing great! 🌟"

FORMAT:
- Use short paragraphs (3-4 lines max each)
- Bold key terms
- Add 1 relevant emoji per section
- Keep language age-appropriate, warm, conversational

Retrieved context:
{context}

Student question: {question}

Your Socratic response (remember to end with a CHECK question and wait for student):
"""
    )
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": tutor_prompt},
    )


def direct_llm_call(system_prompt, user_prompt):
    """Direct LLM call for modes that don't need retrieval (summary, quiz gen, projects)."""
    llm = get_llm()
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages)
    return response.content


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — FEATURE: TUTOR MODE
# ─────────────────────────────────────────────────────────────────────────────

def render_tutor_mode(qa_chain, student_id, student_name):
    """Personalized Socratic tutor with follow-up tracking."""

    st.markdown("""
    <div class="mode-header tutor-header">
        🎓 Personalized Tutor
        <div class="mode-subtitle">Ask me anything from your Science textbook!</div>
    </div>
    """, unsafe_allow_html=True)

    # Init session state
    if "tutor_messages" not in st.session_state:
        st.session_state.tutor_messages = []
    if "followup_count" not in st.session_state:
        st.session_state.followup_count = 0
    if "current_chapter" not in st.session_state:
        st.session_state.current_chapter = ""
    if "awaiting_answer" not in st.session_state:
        st.session_state.awaiting_answer = False

    # Display chat history
    for msg in st.session_state.tutor_messages:
        with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("badges"):
                badge_html = " ".join(
                    f'<span class="chapter-badge">📖 {b}</span>' for b in msg["badges"]
                )
                st.markdown(f'<div class="badge-row">{badge_html}</div>', unsafe_allow_html=True)

    # Chapter selector for context
    col1, col2 = st.columns([3, 1])
    with col2:
        selected_chapter = st.selectbox(
            "📚 Chapter focus",
            ["All Chapters"] + list(CHAPTERS.values()),
            key="chapter_selector"
        )

    # Chat input
    user_input = st.chat_input("Type your question here... 💬")

    if user_input:
        # Check if question is off-topic (heuristic: very short or no science keywords)
        science_keywords = ["what", "why", "how", "explain", "define", "difference",
                           "example", "chapter", "cell", "force", "sound", "crop",
                           "micro", "fibre", "metal", "coal", "fire", "plant", "animal",
                           "friction", "electric", "phenomena", "reproduce", "adolescence"]
        is_off_topic = len(user_input.split()) < 3 and not any(
            kw in user_input.lower() for kw in science_keywords
        )

        if is_off_topic and len(user_input.split()) < 4:
            quote, author = random.choice(MOTIVATIONAL_QUOTES)
            response_text = f"""
{quote}
*{author}*

😄 Looks like you're taking a little detour! That's okay — even the best explorers take breaks.
But your Science textbook is waiting for you, and there's so much cool stuff to discover!

Let me bring you back — here's something interesting: **Science is literally all around you!**
Every time you cook food, ride a bike, or hear music — that's Science in action! 🔬

Now, shall we dive back in? What topic from your Science chapter would you like to explore? 🚀
"""
            st.session_state.tutor_messages.append({"role": "user", "content": user_input})
            st.session_state.tutor_messages.append({
                "role": "assistant", "content": response_text, "badges": []
            })
            st.rerun()
            return

        # Check follow-up limit
        if st.session_state.followup_count >= MAX_FOLLOWUPS:
            response_text = """
😊 You've asked some really great questions! I can see you're really thinking hard about this topic.

We've explored this concept together quite deeply now. For any remaining questions,
I'd suggest **talking to your teacher** — they can explain it in person with examples
specific to your class! Teachers love curious students like you! 👩‍🏫

Meanwhile, shall we move to a **new topic or chapter**? Just type your next question! 🌟
"""
            st.session_state.followup_count = 0
            st.session_state.tutor_messages.append({"role": "user", "content": user_input})
            st.session_state.tutor_messages.append({
                "role": "assistant", "content": response_text, "badges": []
            })
            st.rerun()
            return

        # Run RAG chain
        with st.spinner("🤔 Thinking..."):
            result = qa_chain.invoke({"query": user_input})

        answer = result["result"]
        source_docs = result.get("source_documents", [])
        badges = get_chapter_from_docs(source_docs)

        # Track follow-ups
        if st.session_state.current_chapter and badges and \
           any(b in st.session_state.current_chapter for b in badges):
            st.session_state.followup_count += 1
        else:
            st.session_state.followup_count = 1
            st.session_state.current_chapter = badges[0] if badges else ""

        # Log to analytics
        chapter_str = badges[0] if badges else "General"
        log_interaction(student_id, student_name, "Tutor", chapter_str, user_input, len(answer))

        st.session_state.tutor_messages.append({"role": "user", "content": user_input})
        st.session_state.tutor_messages.append({
            "role": "assistant", "content": answer, "badges": badges
        })
        st.rerun()

    # Clear button
    if st.session_state.tutor_messages:
        if st.button("🔄 Start New Topic", key="clear_tutor"):
            st.session_state.tutor_messages = []
            st.session_state.followup_count = 0
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — FEATURE: QUICK SUMMARY MASTER
# ─────────────────────────────────────────────────────────────────────────────

def render_summary_mode(retriever, student_id, student_name):
    """
    Summary mode uses its OWN dedicated prompt — NOT the Bloom's tutor prompt.
    It retrieves chapter chunks then asks for a structured factual summary.
    """
    st.markdown("""
    <div class="mode-header summary-header">
        📋 Quick Summary Master
        <div class="mode-subtitle">One-page chapter summaries with key concepts & misconceptions</div>
    </div>
    """, unsafe_allow_html=True)

    chapter = st.selectbox("Select a Chapter", list(CHAPTERS.values()), key="summary_chapter")

    if st.button("📄 Generate Summary", key="gen_summary", use_container_width=True):
        with st.spinner("📝 Building your chapter summary..."):

            # Step 1: Retrieve relevant chunks for this chapter
            docs = retriever.invoke(
                f"{chapter} main concepts definitions key points facts"
            )
            # Combine retrieved text as context
            context = "\n\n".join([d.page_content for d in docs]) if docs else ""

            # Step 2: Use a DEDICATED summary prompt — no Bloom's pedagogy here
            summary_system = """You are a textbook summarizer for CBSE Grade 8 Science.
Your job is to create a clear, structured, factual one-page chapter summary.
DO NOT use Bloom's Taxonomy. DO NOT ask questions. DO NOT be a tutor.
Only answer from the provided context. Be factual and concise.
Use simple language appropriate for Class 8 students."""

            summary_user = f"""Using the context below, write a one-page summary of:
"{chapter}" from CBSE Grade 8 Science.

Structure your summary EXACTLY like this:

## 📚 {chapter} — Quick Summary

### 🔑 What This Chapter Is About
(2-3 sentences overview of the chapter topic)

### 📌 Key Concepts
(List 5-8 main concepts with a one-line explanation each)

### 📖 Important Definitions
(List 5-6 key terms with their definitions)

### ⚡ Must-Remember Facts
(List 5-6 important facts students must know for exams)

### 🌍 Real-Life Connections
(2-3 examples of where this chapter topic appears in daily life)

### ⚠️ Common Misconceptions
(List 3-4 things students often get wrong about this topic, and the correct explanation)

### 📝 Exam Tips
(2-3 short tips for answering exam questions on this chapter)

Context from textbook:
{context}

Write the summary now. Be factual, clear, and structured. No questions, no Bloom's levels."""

            summary = direct_llm_call(summary_system, summary_user)

        log_interaction(student_id, student_name, "Summary", chapter, "summary request", len(summary))

        # Display in a clean card
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-title">📚 {chapter}</div>
            <div style="color:#166534;font-size:0.8rem;margin-top:4px;">
                ✅ Based on your NCERT textbook
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(summary)

        # Source badge
        if docs:
            badges = get_chapter_from_docs(docs)
            badge_html = " ".join(f'<span class="chapter-badge">📖 {b}</span>' for b in badges)
            st.markdown(f'<div class="badge-row">{badge_html}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Download button — goes to user's Downloads automatically via browser
        st.download_button(
            label="⬇️ Download Summary (saves to your Downloads folder)",
            data=f"# {chapter} — Quick Summary\n\n{summary}",
            file_name=f"LearnIQ_Summary_{chapter[:30].replace(' ','_').replace('—','')}.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — FEATURE: PRACTICAL PROJECTS MASTER
# ─────────────────────────────────────────────────────────────────────────────

def render_projects_mode(qa_chain, student_id, student_name):
    st.markdown("""
    <div class="mode-header projects-header">
        🔬 Practical Projects Master
        <div class="mode-subtitle">Safe, fun projects to master concepts at home or school</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        chapter = st.selectbox("📚 Chapter", list(CHAPTERS.values()), key="proj_chapter")
    with col2:
        location = st.selectbox("📍 Where?", ["Home", "School", "Both"], key="proj_location")

    if st.button("🚀 Generate Projects", key="gen_projects", use_container_width=True):
        with st.spinner("🔭 Creating project ideas..."):
            prompt = f"""
For the chapter "{chapter}" in CBSE Grade 8 Science, create 3 safe, practical projects
that can be done at {location} by a 13-14 year old student.

For EACH project provide:
🎯 PROJECT NAME & CONCEPT LINK
📌 Objective (what will the student learn)
🧰 Materials Required (simple, household/school items only — SAFE for Class 8)
📋 Step-by-step Activity (5-7 simple steps)
🔍 What to Observe
💡 How this links to the textbook concept
🌟 Fun Extension (one creative twist)

Make projects exciting, safe, and directly linked to chapter concepts.
Use only materials easily available in India.
"""
            result = qa_chain.invoke({"query": prompt})
            projects_text = result["result"]

        log_interaction(student_id, student_name, "Projects", chapter, "projects request", len(projects_text))

        st.markdown(f'<div class="projects-container">', unsafe_allow_html=True)
        st.markdown(projects_text)
        st.markdown('</div>', unsafe_allow_html=True)

        st.download_button(
            "⬇️ Download Project Brief",
            data=f"# Projects: {chapter}\n\n{projects_text}",
            file_name=f"projects_{chapter[:20].replace(' ','_')}.md",
            mime="text/markdown"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — FEATURE: QUIZ MASTER
# ─────────────────────────────────────────────────────────────────────────────

def render_quiz_mode(qa_chain, student_id, student_name):
    st.markdown("""
    <div class="mode-header quiz-header">
        🏆 Quiz Master
        <div class="mode-subtitle">Test yourself! 6 questions across Bloom's Taxonomy levels</div>
    </div>
    """, unsafe_allow_html=True)

    # Init quiz state
    if "quiz_state" not in st.session_state:
        st.session_state.quiz_state = "select"  # select → active → results
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "quiz_chapter" not in st.session_state:
        st.session_state.quiz_chapter = ""
    if "quiz_results_data" not in st.session_state:
        st.session_state.quiz_results_data = None

    # ── STEP 1: Select chapter ──
    if st.session_state.quiz_state == "select":
        chapter = st.selectbox("📚 Choose a Chapter", list(CHAPTERS.values()), key="quiz_chapter_sel")

        # Show current competency
        competency = get_student_competency(student_id, chapter)
        level_colors = {
            "Not Started": "⚪", "Basic": "🔴", "Good": "🟡",
            "Proficient": "🟢", "Master": "🏆"
        }
        st.markdown(f"""
        <div class="competency-bar">
            {level_colors.get(competency, '⚪')} Your level for this chapter:
            <strong>{competency}</strong>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🎯 Start Quiz!", key="start_quiz", use_container_width=True):
            with st.spinner("🎲 Generating your quiz..."):
                quiz_prompt = f"""
Create exactly 6 quiz questions for "{chapter}" (CBSE Grade 8 Science).
One question per Bloom's Taxonomy level: Remember, Understand, Apply, Analyse, Evaluate, Create.

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{{
  "questions": [
    {{
      "level": "Remember",
      "question": "question text here",
      "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
      "correct": "A",
      "explanation": "brief explanation why A is correct"
    }}
  ]
}}

Make questions appropriate for Class 8, clear, and directly from chapter content.
Vary the correct answer positions (not always A).
"""
                system_prompt = (
                    "You are a quiz generator. Return ONLY valid JSON, no markdown code blocks, "
                    "no preamble, no explanation. Just the raw JSON object."
                )
                raw = direct_llm_call(system_prompt, quiz_prompt)

                try:
                    # Strip any accidental markdown fences
                    clean = raw.strip().replace("```json", "").replace("```", "").strip()
                    quiz_data = json.loads(clean)
                    st.session_state.quiz_questions = quiz_data["questions"]
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_chapter = chapter
                    st.session_state.quiz_state = "active"
                    st.session_state.quiz_results_data = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Quiz generation failed. Please try again. ({e})")

    # ── STEP 2: Active Quiz ──
    elif st.session_state.quiz_state == "active":
        st.markdown(f"### 📝 Quiz: {st.session_state.quiz_chapter}")
        st.markdown(f"*Answer all 6 questions, one from each Bloom's level*")

        bloom_colors = {
            "Remember": "🔵", "Understand": "🟢", "Apply": "🟡",
            "Analyse": "🟠", "Evaluate": "🔴", "Create": "🟣"
        }

        with st.form("quiz_form"):
            for i, q in enumerate(st.session_state.quiz_questions):
                level = q.get("level", "")
                color = bloom_colors.get(level, "⚪")
                st.markdown(f"""
                <div class="quiz-question">
                    <span class="bloom-tag">{color} {level}</span>
                    <strong>Q{i+1}. {q['question']}</strong>
                </div>
                """, unsafe_allow_html=True)
                options = q.get("options", [])
                st.session_state.quiz_answers[i] = st.radio(
                    f"Q{i+1}", options, key=f"q_{i}", label_visibility="collapsed"
                )
                st.markdown("---")

            submitted = st.form_submit_button("📊 Submit Quiz!", use_container_width=True)

        if submitted:
            # Evaluate answers
            score = 0
            results = []
            for i, q in enumerate(st.session_state.quiz_questions):
                chosen = st.session_state.quiz_answers.get(i, "")
                correct_letter = q.get("correct", "A")
                # Check if chosen answer starts with the correct letter
                is_correct = chosen.startswith(correct_letter + ")")
                if is_correct:
                    score += 1
                results.append({
                    "question": q["question"],
                    "level": q.get("level", ""),
                    "chosen": chosen,
                    "correct_letter": correct_letter,
                    "correct_option": next(
                        (o for o in q["options"] if o.startswith(correct_letter + ")")), ""
                    ),
                    "explanation": q.get("explanation", ""),
                    "is_correct": is_correct,
                })

            competency = log_quiz_result(
                student_id, student_name,
                st.session_state.quiz_chapter, score, 6
            )
            log_interaction(student_id, student_name, "Quiz",
                          st.session_state.quiz_chapter,
                          f"Quiz score: {score}/6", score)

            st.session_state.quiz_results_data = {
                "score": score, "results": results, "competency": competency
            }
            st.session_state.quiz_state = "results"
            st.rerun()

        if st.button("❌ Cancel Quiz"):
            st.session_state.quiz_state = "select"
            st.rerun()

    # ── STEP 3: Results ──
    elif st.session_state.quiz_state == "results":
        data = st.session_state.quiz_results_data
        score = data["score"]
        competency = data["competency"]

        # Score display
        pct = int((score / 6) * 100)
        score_color = "#ff4444" if pct < 40 else "#ffaa00" if pct < 70 else "#44bb44"
        st.markdown(f"""
        <div class="score-card" style="border-color: {score_color}">
            <div class="score-number" style="color: {score_color}">{score}/6</div>
            <div class="score-label">{pct}% — {competency}</div>
            <div class="score-chapter">{st.session_state.quiz_chapter}</div>
        </div>
        """, unsafe_allow_html=True)

        # Motivational message
        if pct == 100:
            msg = "🏆 PERFECT SCORE! You're an absolute master! Incredible work!"
        elif pct >= 80:
            msg = "🌟 Outstanding! You really know this chapter well!"
        elif pct >= 60:
            msg = "👍 Good job! A little more practice and you'll be a master!"
        elif pct >= 40:
            msg = "💪 Keep going! Every attempt makes you stronger. Try again?"
        else:
            msg = "😊 Don't worry! Review the chapter summary and try again. You've got this!"

        st.markdown(f'<div class="motivation-msg">{msg}</div>', unsafe_allow_html=True)

        # Detailed feedback per question
        st.markdown("### 📋 Question-by-Question Feedback")
        for i, r in enumerate(data["results"]):
            icon = "✅" if r["is_correct"] else "❌"
            color = "#d4edda" if r["is_correct"] else "#f8d7da"
            st.markdown(f"""
            <div class="feedback-card" style="background:{color}">
                {icon} <strong>Q{i+1} ({r['level']})</strong>: {r['question']}<br>
                Your answer: <em>{r['chosen']}</em><br>
                {"" if r['is_correct'] else f"✔️ Correct answer: <em>{r['correct_option']}</em><br>"}
                💡 {r['explanation']}
            </div>
            """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retake Quiz (new questions)", use_container_width=True):
                st.session_state.quiz_state = "select"
                st.rerun()
        with col2:
            if st.button("📚 Study this Chapter", use_container_width=True):
                st.session_state.quiz_state = "select"
                st.session_state["active_mode"] = "📋 Summary Master"
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — FEATURE: TEACHER DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def render_teacher_dashboard():
    st.markdown("""
    <div class="mode-header teacher-header">
        👩‍🏫 Teacher Dashboard
        <div class="mode-subtitle">Class analytics, student progress & concept difficulty</div>
    </div>
    """, unsafe_allow_html=True)

    # Simple teacher password
    if "teacher_auth" not in st.session_state:
        st.session_state.teacher_auth = False

    if not st.session_state.teacher_auth:
        pwd = st.text_input("🔐 Teacher Password", type="password")
        if st.button("Login"):
            if hashlib.md5(pwd.encode()).hexdigest() == "4a8a08f09d37b73795649038408b5f33":  # "abc123"
                st.session_state.teacher_auth = True
                st.rerun()
            else:
                st.error("Incorrect password. Default: abc123")
        st.caption("Default password: abc123 — change in production!")
        return

    data = get_teacher_analytics()

    # KPI row
    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Total Students", data["total_students"])
    col2.metric("💬 Total Interactions", data["total_interactions"])
    col3.metric("📊 Chapters Accessed", len(data["chapter_access"]))

    st.markdown("---")

    # Chapter access frequency
    if data["chapter_access"]:
        st.markdown("#### 📚 Most Accessed Chapters (difficulty indicator)")
        for chapter, count in data["chapter_access"][:8]:
            bar_width = min(int((count / max(c for _, c in data["chapter_access"])) * 100), 100)
            st.markdown(f"""
            <div class="dash-bar-label">{chapter or 'General'} <span class="dash-count">({count} accesses)</span></div>
            <div class="dash-bar-bg"><div class="dash-bar-fill" style="width:{bar_width}%"></div></div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Quiz performance
    if data["quiz_avg"]:
        st.markdown("#### 🏆 Chapter Quiz Performance (avg score %)")
        for chapter, avg_score, attempts in data["quiz_avg"]:
            if avg_score:
                color = "#ff4444" if avg_score < 40 else "#ffaa00" if avg_score < 70 else "#44bb44"
                st.markdown(f"""
                <div class="dash-bar-label">{chapter or 'Unknown'}
                    <span class="dash-count">{avg_score:.0f}% avg ({attempts} attempts)</span>
                </div>
                <div class="dash-bar-bg">
                    <div class="dash-bar-fill" style="width:{min(avg_score,100):.0f}%;background:{color}"></div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # Competency distribution
    if data["competency_dist"]:
        st.markdown("#### 🎓 Class Competency Distribution")
        cols = st.columns(len(data["competency_dist"]))
        for i, (level, count) in enumerate(data["competency_dist"]):
            cols[i].metric(level or "Unknown", count)

    st.markdown("---")

    # Recent students
    if data["recent_students"]:
        st.markdown("#### 👤 Recent Student Activity")
        for name, sid, last_seen in data["recent_students"]:
            st.markdown(f"**{name}** · Last active: {last_seen[:16]}")

    if st.button("🔒 Logout"):
        st.session_state.teacher_auth = False
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — STREAMLIT UI MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="LearnIQ — Grade 8 Science",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── GLOBAL CSS ──
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@400;600;800&family=Nunito:wght@400;600;700&display=swap');

    /* ══ RESET & BASE ══ */
    html, body, [class*="css"], .stApp {
        font-family: 'Nunito', sans-serif !important;
        background-color: #f0f4f8 !important;
        color: #1a1a2e !important;
    }

    /* ══ MAIN CONTENT AREA — white background, dark text ══ */
    .main .block-container {
        background-color: #f0f4f8 !important;
        padding-top: 1.5rem !important;
    }

    /* All regular text in main area must be dark */
    .main p, .main li, .main span, .main div,
    .stMarkdown p, .stMarkdown li { color: #1a1a2e !important; }

    /* ══ APP HEADER ══ */
    .app-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        border-radius: 20px;
        padding: 1.4rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .app-header .logo {
        font-family: 'Baloo 2', cursive;
        font-size: 2rem; font-weight: 800;
        color: #ffffff !important;
    }
    .app-header .logo .iq  { color: #ffd700 !important; }
    .app-header .logo .sci { color: #7dd3fc !important; }
    .app-header .tagline   { color: #cbd5e1 !important; font-size: 0.9rem; margin-top: 4px; }

    /* ══ MODE HEADERS — dark bg, bright accent text ══ */
    .mode-header {
        font-family: 'Baloo 2', cursive;
        font-size: 1.5rem; font-weight: 800;
        padding: 1rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border-left: 6px solid;
    }
    .mode-subtitle {
        font-family: 'Nunito', sans-serif;
        font-size: 0.85rem; font-weight: 600;
        margin-top: 4px; opacity: 0.9;
    }
    .tutor-header    { background:#1e3a5f; border-color:#7dd3fc; color:#7dd3fc !important; }
    .tutor-header .mode-subtitle { color:#bfdbfe !important; }
    .summary-header  { background:#14532d; border-color:#86efac; color:#86efac !important; }
    .summary-header .mode-subtitle { color:#bbf7d0 !important; }
    .projects-header { background:#78350f; border-color:#fcd34d; color:#fcd34d !important; }
    .projects-header .mode-subtitle { color:#fde68a !important; }
    .quiz-header     { background:#3b0764; border-color:#d8b4fe; color:#d8b4fe !important; }
    .quiz-header .mode-subtitle { color:#ede9fe !important; }
    .teacher-header  { background:#7f1d1d; border-color:#fca5a5; color:#fca5a5 !important; }
    .teacher-header .mode-subtitle { color:#fee2e2 !important; }

    /* ══ FEATURE CARDS (landing page) ══ */
    .feature-card {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.4rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        transition: transform 0.2s;
    }
    .feature-card:hover { transform: translateY(-4px); }
    .feature-card .fc-icon  { font-size: 2.2rem; }
    .feature-card .fc-title { font-family:'Baloo 2',cursive; font-size:1.1rem;
                               font-weight:800; color:#1e3a5f !important; margin: 8px 0 4px; }
    .feature-card .fc-desc  { font-size:0.8rem; color:#475569 !important; }

    /* ══ CHAPTER BADGES ══ */
    .chapter-badge {
        display: inline-block;
        background: #1e3a5f;
        color: #7dd3fc !important;
        font-size: 0.72rem; font-weight: 700;
        padding: 3px 10px; border-radius: 20px;
        margin: 3px 3px 0 0;
    }
    .badge-row { margin-top: 0.6rem; }

    /* ══ CHAT MESSAGES ══ */
    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 14px !important;
        margin-bottom: 10px !important;
        padding: 12px !important;
        color: #1a1a2e !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span { color: #1a1a2e !important; }

    /* ══ QUIZ COMPONENTS ══ */
    .quiz-question {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        color: #1a1a2e !important;
    }
    .bloom-tag {
        display: inline-block;
        background: #1e3a5f;
        color: #7dd3fc !important;
        border-radius: 8px;
        padding: 2px 10px;
        font-size: 0.75rem; font-weight: 700;
        margin-bottom: 8px;
    }
    .feedback-card {
        border-radius: 10px; padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.9rem; color: #1a1a2e !important;
        border: 1px solid #cbd5e1;
    }
    .score-card {
        text-align: center; border: 3px solid;
        border-radius: 20px; padding: 2rem;
        margin: 1rem 0; background: #ffffff;
    }
    .score-number { font-family:'Baloo 2',cursive; font-size:3.5rem; font-weight:800; }
    .score-label  { font-size:1.2rem; font-weight:700; margin-top:4px; color:#1a1a2e !important; }
    .score-chapter { color:#475569 !important; font-size:0.85rem; margin-top:6px; }
    .motivation-msg {
        background: #fefce8;
        border-left: 5px solid #f59e0b;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
        color: #78350f !important;
        font-weight: 700; margin: 1rem 0;
    }

    /* ══ SUMMARY & PROJECTS ══ */
    .summary-card {
        background: #f0fdf4;
        border: 2px solid #86efac;
        border-radius: 12px;
        padding: 1rem 1.5rem; margin-bottom: 1rem;
    }
    .summary-title {
        font-family:'Baloo 2',cursive; font-size:1.4rem;
        color: #14532d !important; font-weight:800;
    }
    .projects-container {
        background: #fffbeb;
        border: 2px solid #fcd34d;
        border-radius: 12px; padding: 1rem 1.5rem;
        color: #1a1a2e !important;
    }

    /* ══ COMPETENCY BAR ══ */
    .competency-bar {
        background: #eff6ff;
        border: 2px solid #bfdbfe;
        border-radius: 10px; padding: 10px 16px;
        margin-bottom: 1rem; font-size:0.9rem;
        color: #1e3a5f !important; font-weight: 600;
    }

    /* ══ TEACHER DASHBOARD ══ */
    .dash-bar-label { font-size:0.85rem; color:#1e3a5f !important; font-weight:600; margin-bottom:3px; margin-top:8px; }
    .dash-count { color:#0369a1 !important; font-weight:700; }
    .dash-bar-bg { background:#e2e8f0; border-radius:6px; height:14px; margin-bottom:4px; }
    .dash-bar-fill { background:#3b82f6; height:14px; border-radius:6px; transition:width 0.4s; }

    /* ══ SIDEBAR — dark theme ══ */
    section[data-testid="stSidebar"] {
        background: #1a1a2e !important;
        border-right: 3px solid #0f3460 !important;
    }
    /* ALL text in sidebar should be light */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #7dd3fc !important; }

    .student-card {
        background: #16213e;
        border-radius: 12px; padding: 12px;
        margin-bottom: 12px;
        border: 1px solid #0f3460;
    }
    .student-name { font-family:'Baloo 2',cursive; font-size:1.1rem; color:#7dd3fc !important; font-weight:700; }
    .student-meta { font-size:0.78rem; color:#94a3b8 !important; }

    /* Sidebar input fields */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] .stTextInput input {
        background: #16213e !important;
        color: #ffffff !important;
        border: 1px solid #0f3460 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background: #16213e !important;
        color: #ffffff !important;
    }

    /* ══ SIDEBAR NAV BUTTONS ══ */
    section[data-testid="stSidebar"] .stButton button {
        background: #16213e !important;
        color: #e2e8f0 !important;
        border: 1px solid #0f3460 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background: #0f3460 !important;
        color: #7dd3fc !important;
        border-color: #7dd3fc !important;
    }
    section[data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: #0f3460 !important;
        color: #7dd3fc !important;
        border: 2px solid #7dd3fc !important;
    }

    /* ══ MAIN AREA BUTTONS ══ */
    .main .stButton button {
        background: #1e3a5f !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 0.5rem 1.2rem !important;
    }
    .main .stButton button:hover {
        background: #0f3460 !important;
        transform: translateY(-1px);
    }

    /* ══ SELECTBOX & INPUTS in main area ══ */
    .main .stSelectbox label,
    .main .stTextInput label { color: #1a1a2e !important; font-weight: 600 !important; }

    /* ══ USAGE INFO captions in sidebar ══ */
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small { color: #94a3b8 !important; }

    </style>
    """, unsafe_allow_html=True)

    # ── Init DB ──
    init_db()

    # ── Sidebar: Student Login ──
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0;">
            <span style="font-family:'Baloo 2',cursive; font-size:1.8rem; font-weight:800; color:#58a6ff;">
                🔬 Learn<span style="color:#f0c000;">IQ</span>
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 👤 Who's learning today?")

        if "student_name" not in st.session_state:
            st.session_state.student_name = ""
        if "student_id" not in st.session_state:
            st.session_state.student_id = ""
        if "logged_in" not in st.session_state:
            st.session_state.logged_in = False

        if not st.session_state.logged_in:
            name = st.text_input("Your Name", placeholder="e.g. Priya Sharma")
            grade = st.selectbox("Class", ["Class 8", "Class 7", "Class 9"])
            if st.button("🚀 Start Learning!", use_container_width=True):
                if name.strip():
                    st.session_state.student_name = name.strip()
                    st.session_state.student_id = hashlib.md5(
                        f"{name.strip()}{grade}".encode()
                    ).hexdigest()[:10]
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.warning("Please enter your name!")
        else:
            st.markdown(f"""
            <div class="student-card">
                <div class="student-name">👋 {st.session_state.student_name}</div>
                <div class="student-meta">ID: {st.session_state.student_id}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔄 Switch Student"):
                st.session_state.logged_in = False
                st.rerun()

        st.markdown("---")

        # Navigation
        st.markdown("### 📚 Modes")
        if "active_mode" not in st.session_state:
            st.session_state.active_mode = "🎓 Tutor Mode"

        modes = [
            "🎓 Tutor Mode",
            "📋 Summary Master",
            "🔬 Projects Master",
            "🏆 Quiz Master",
            "👩‍🏫 Teacher Dashboard",
        ]
        for mode in modes:
            is_active = st.session_state.active_mode == mode
            btn_style = "primary" if is_active else "secondary"
            if st.button(mode, use_container_width=True, type=btn_style, key=f"nav_{mode}"):
                st.session_state.active_mode = mode
                st.rerun()

        st.markdown("---")
        st.markdown("### 💰 Usage Info")
        st.caption(f"Model: `{LLM_MODEL}`")
        st.caption(f"Embedding: `{EMBED_MODEL}`")
        st.caption("Each question ≈ $0.0005")
        st.caption("$5 budget ≈ 10,000 questions")

    # ── Main content: require login first ──
    if not st.session_state.logged_in:
        st.markdown("""
        <div class="app-header">
            <div>
                <div class="logo">🔬 Learn<span>IQ</span> — <em>Grade 8 Science</em></div>
                <div class="tagline">Your personalized AI science tutor · Powered by your NCERT textbook</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        for col, icon, title, desc in zip(
            [col1, col2, col3, col4],
            ["🎓", "📋", "🔬", "🏆"],
            ["Tutor Mode", "Quick Summary", "Projects", "Quiz Master"],
            ["Socratic Q&A with Bloom's ladder",
             "1-page chapter summaries",
             "Safe home/school experiments",
             "6-level Bloom's quizzes"]
        ):
            col.markdown(f"""
            <div class="feature-card">
                <div class="fc-icon">{icon}</div>
                <div class="fc-title">{title}</div>
                <div class="fc-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center;margin-top:2rem;color:#475569;font-size:0.95rem;font-weight:600;">
            👈 Enter your name in the sidebar to start learning!
        </div>
        """, unsafe_allow_html=True)
        return

    # ── App header (when logged in) ──
    st.markdown(f"""
    <div class="app-header">
        <div>
            <div class="logo">🔬 Learn<span class="iq">IQ</span> —
            <span class="sci">Grade 8 Science</span></div>
            <div class="tagline">Welcome back, <strong>{st.session_state.student_name}</strong>!
            Let's explore Science today 🌟</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load RAG chain (cached) ──
    retriever = build_retriever()
    qa_chain = build_qa_chain(retriever)

    # ── Route to active mode ──
    mode = st.session_state.active_mode
    sid = st.session_state.student_id
    sname = st.session_state.student_name

    if mode == "🎓 Tutor Mode":
        render_tutor_mode(qa_chain, sid, sname)
    elif mode == "📋 Summary Master":
        render_summary_mode(retriever, sid, sname)   # ← passes retriever, not qa_chain
    elif mode == "🔬 Projects Master":
        render_projects_mode(qa_chain, sid, sname)
    elif mode == "🏆 Quiz Master":
        render_quiz_mode(qa_chain, sid, sname)
    elif mode == "👩‍🏫 Teacher Dashboard":
        render_teacher_dashboard()


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# ⚑ SWAP ChromaDB → PINECONE (exactly 2 lines)
#
# LINE 1 — Replace import:
#   FROM: from langchain_community.vectorstores import Chroma
#   TO:   from langchain_pinecone import PineconeVectorStore
#         import pinecone
#         pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="us-east-1")
#
# LINE 2 — Replace Chroma.from_documents(...) with:
#   vectorstore = PineconeVectorStore.from_documents(
#       documents=chunks, embedding=embeddings,
#       index_name="learniq-grade8",
#   )
# And replace Chroma(persist_directory=...) reload with:
#   vectorstore = PineconeVectorStore.from_existing_index(
#       index_name="learniq-grade8", embedding=embeddings
#   )
# ══════════════════════════════════════════════════════════════════════════════
