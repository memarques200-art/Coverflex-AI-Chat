import streamlit as st
import pickle
import os
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(
    page_title="Coverflex AI",
    page_icon="https://www.coverflex.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

#MainMenu, footer, header { visibility: hidden; }

*, *::before, *::after {
    font-family: 'DM Sans', sans-serif !important;
    box-sizing: border-box;
}

:root {
    --peach: #F07855;
    --peach-light: #FEF0EB;
    --peach-mid: #F5B09A;
    --peach-dark: #C95A35;
    --navy: #1C1B2E;
    --white: #FFFFFF;
    --bg: #F7F6F3;
    --border: #E8E4DF;
    --text: #2A2724;
    --muted: #8C877F;
    --success-bg: #EDF7F2;
    --success-border: #A3D9BC;
    --success-text: #1A6640;
    --warn-bg: #FEF6EC;
    --warn-border: #F5D09A;
    --warn-text: #8C5A00;
}

/* App background */
.stApp { background: var(--bg) !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 270px !important;
    max-width: 270px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
    overflow-x: hidden !important;
}
/* Always show collapse arrow, style it */
[data-testid="collapsedControl"] {
    background: var(--peach) !important;
    color: white !important;
    border-radius: 0 8px 8px 0 !important;
    width: 20px !important;
    top: 48% !important;
}

/* ── SIDEBAR BUTTONS ── */
div[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    padding: 7px 12px !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.15s !important;
    justify-content: flex-start !important;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--peach-light) !important;
    color: var(--peach-dark) !important;
}

/* New chat button special style */
div[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-secondary"]:first-of-type,
.new-chat-btn .stButton > button {
    background: var(--peach) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(240,120,85,0.25) !important;
}

/* ── CHAT INPUT ── */
.stChatInput > div {
    border-radius: 16px !important;
    border: 1.5px solid var(--border) !important;
    background: var(--white) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
    transition: all 0.2s !important;
}
.stChatInput > div:focus-within {
    border-color: var(--peach) !important;
    box-shadow: 0 0 0 3px rgba(240,120,85,0.12) !important;
}
.stChatInput textarea {
    font-size: 14px !important;
    color: var(--text) !important;
    background: transparent !important;
    padding: 14px 18px !important;
}
.stChatInput textarea::placeholder { color: var(--muted) !important; }

/* ── CHAT MESSAGES ── */
.stChatMessage {
    background: transparent !important;
    padding: 6px 0 !important;
    max-width: 720px !important;
    margin: 0 auto !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stChatMessageContent"] p {
    font-size: 14px !important;
    line-height: 1.8 !important;
    color: var(--text) !important;
}

/* Assistant message */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 16px 20px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}

/* User message */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--peach-light) !important;
    border: 1px solid var(--peach-mid) !important;
    border-radius: 16px !important;
    padding: 14px 20px !important;
}

/* Avatars */
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarUser"] {
    width: 32px !important;
    height: 32px !important;
    border-radius: 8px !important;
    min-width: 32px !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--peach) !important; }

/* Expander */
.streamlit-expanderHeader {
    font-size: 12px !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Bottom padding for chat */
section.main > div { padding-bottom: 20px !important; }
</style>
""", unsafe_allow_html=True)

# ── FUNCTIONS ──────────────────────────────────────────────
MEMORY_DIR = "chat_histories"
VECTORSTORE_DIR = "coverflex_vectorstore"
os.makedirs(MEMORY_DIR, exist_ok=True)

def get_conversation_files():
    files = [f for f in os.listdir(MEMORY_DIR) if f.endswith('.pkl')]
    files.sort(reverse=True)
    return files

def load_conversation(cid):
    fp = os.path.join(MEMORY_DIR, f"{cid}.pkl")
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            return pickle.load(f)
    return []

def save_conversation(cid, messages):
    with open(os.path.join(MEMORY_DIR, f"{cid}.pkl"), "wb") as f:
        pickle.dump(messages, f)

def new_conversation():
    cid = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.cid = cid
    st.session_state.messages = []
    st.session_state.show_welcome = True
    save_conversation(cid, [])

def load_vectorstore():
    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        try:
            emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            return Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=emb)
        except:
            return None
    return None

def generate_response(user_msg, history, docs):
    ctx = ""
    if docs:
        ctx = "Informação oficial da Coverflex:\n\n"
        for d in docs[:3]:
            ctx += d.page_content[:600] + "\n\n"

    hist = ""
    for m in history[-6:]:
        hist += f"{'Utilizador' if m['role']=='user' else 'Assistente'}: {m['content']}\n"

    system = """És o assistente de conhecimento interno da Coverflex — fintech portuguesa líder em benefícios flexíveis, fundada em 2021, com presença em Portugal, Espanha e Itália.

Respondes SEMPRE em Português de Portugal. Tom profissional, claro e direto — como um colega sénior bem informado.

Regras:
1. Usa a informação disponível na knowledge base
2. Se não tiveres informação, indica qual equipa contactar (RH: rh@coverflex.com, Suporte: help@coverflex.com, IT: it-support@coverflex.com)
3. Sê concreto e específico — sem generalidades
4. Valores Coverflex: Extreme Employee Obsession, transparência, simplicidade"""

    content = f"{ctx}\n\nConversa anterior:\n{hist}\n\nPergunta: {user_msg}" if hist else f"{ctx}\n\nPergunta: {user_msg}"

    try:
        r = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            max_tokens=1000, temperature=0.65
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"❌ Erro: {str(e)}"

# ── INIT ───────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()
if "app_init" not in st.session_state:
    st.session_state.app_init = True
    st.session_state.cid = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.messages = []
    st.session_state.show_welcome = True
if "quick_q" not in st.session_state:
    st.session_state.quick_q = None
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:

    # Header / Logo
    st.markdown("""
    <div style="padding:22px 18px 14px;border-bottom:1px solid #E8E4DF;">
      <div style="display:flex;align-items:center;gap:10px;">
        <div style="width:38px;height:38px;background:#F07855;border-radius:11px;
                    display:flex;align-items:center;justify-content:center;
                    box-shadow:0 4px 14px rgba(240,120,85,0.28);flex-shrink:0;">
          <span style="color:white;font-size:14px;font-weight:800;letter-spacing:-0.5px;">CF</span>
        </div>
        <div>
          <div style="font-size:15px;font-weight:700;color:#1C1B2E;line-height:1.2;">Coverflex</div>
          <div style="font-size:10px;color:#F07855;font-weight:600;
                      letter-spacing:0.07em;text-transform:uppercase;">Knowledge Assistant</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # New chat button
    st.markdown('<div style="padding:14px 14px 6px;">', unsafe_allow_html=True)
    if st.button("＋  Nova Conversa", use_container_width=True, key="new_chat"):
        new_conversation()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # KB status
    kb_ok = st.session_state.vectorstore is not None
    st.markdown(f"""
    <div style="margin:2px 14px 10px;padding:9px 13px;
                background:{'#EDF7F2' if kb_ok else '#FEF6EC'};
                border:1px solid {'#A3D9BC' if kb_ok else '#F5D09A'};
                border-radius:9px;font-size:12px;font-weight:500;
                color:{'#1A6640' if kb_ok else '#8C5A00'};">
      {'✅  Base de conhecimento ativa' if kb_ok else '⚠️  Sem documentos — carrega ficheiros abaixo'}
    </div>
    """, unsafe_allow_html=True)

    # Quick topics
    st.markdown("""
    <div style="padding:6px 18px 5px;">
      <span style="font-size:10px;font-weight:600;letter-spacing:0.1em;
                   color:#B5AFA8;text-transform:uppercase;">Tópicos</span>
    </div>
    """, unsafe_allow_html=True)

    TOPICS = [
        ("💳", "Cartão Coverflex",    "Como ativo o cartão Coverflex?"),
        ("🏖️", "Férias & Tempo Livre","Quantos dias de férias tenho na Coverflex?"),
        ("🏠", "Remote Work Budget",  "O que é o Remote Work Budget?"),
        ("📚", "Personal Growth",     "O que é o Personal Growth Budget?"),
        ("👶", "Licença Parental",    "Qual é a licença parental na Coverflex?"),
        ("❤️", "Caring Days",         "O que são os Caring Days?"),
        ("📈", "Stock Options",       "Como funcionam as stock options?"),
        ("🏥", "Seguro de Saúde",     "Como funciona o seguro de saúde?"),
        ("🔄", "Reembolsos",          "Como faço um pedido de reembolso?"),
        ("👋", "Onboarding",          "Quais são os primeiros passos no onboarding?"),
        ("📞", "Contactos & Suporte", "Quais os contactos do suporte Coverflex?"),
    ]

    for icon, label, question in TOPICS:
        if st.button(f"{icon}  {label}", key=f"t_{label}", use_container_width=True):
            st.session_state.quick_q = question
            st.session_state.show_welcome = False
            if not st.session_state.messages:
                st.session_state.cid = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.rerun()

    # Recent chats
    convs = get_conversation_files()
    if convs:
        st.markdown("""
        <div style="padding:12px 18px 5px;margin-top:4px;border-top:1px solid #E8E4DF;">
          <span style="font-size:10px;font-weight:600;letter-spacing:0.1em;
                       color:#B5AFA8;text-transform:uppercase;">Recentes</span>
        </div>
        """, unsafe_allow_html=True)
        for conv in convs[:7]:
            cid = conv.replace('.pkl', '')
            try:
                d = datetime.strptime(cid, "%Y%m%d_%H%M%S")
                name = d.strftime("%d/%m · %H:%M")
            except:
                name = cid[:12]
            active = cid == st.session_state.cid
            if st.button(
                f"{'🟠' if active else '💬'}  {name}",
                key=f"h_{cid}", use_container_width=True
            ):
                st.session_state.cid = cid
                st.session_state.messages = load_conversation(cid)
                st.session_state.show_welcome = False
                st.rerun()

    # Train
    st.markdown('<div style="border-top:1px solid #E8E4DF;padding:10px 14px 14px;">', unsafe_allow_html=True)
    with st.expander("📁  Treinar com documentos"):
        files = st.file_uploader("PDF ou TXT", type=["pdf","txt"], accept_multiple_files=True)
        if files and st.button("🚀 Treinar agora", use_container_width=True):
            with st.spinner("A processar..."):
                docs = []
                for f in files:
                    tmp = f"tmp_{f.name}"
                    with open(tmp,"wb") as fp: fp.write(f.getbuffer())
                    loader = PyPDFLoader(tmp) if f.name.endswith('.pdf') else TextLoader(tmp, encoding="utf-8")
                    docs.extend(loader.load())
                    os.remove(tmp)
                chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
                emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vs = Chroma.from_documents(chunks, emb, persist_directory=VECTORSTORE_DIR)
                vs.persist()
                st.session_state.vectorstore = vs
                st.success(f"✅ {len(chunks)} fragmentos!")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── MAIN AREA ──────────────────────────────────────────────

SUGGESTIONS = [
    ("🍽️", "Cartão de refeição",   "Qual é o limite diário do cartão de refeição Coverflex?"),
    ("🎁", "Benefícios flexíveis", "Como funcionam os benefícios flexíveis da Coverflex?"),
    ("🏥", "Seguro de saúde",      "Como funciona o seguro de saúde na Coverflex?"),
    ("🏠", "Remote Work Budget",   "O que é o Remote Work Budget da Coverflex?"),
    ("📈", "Stock Options",        "Como funcionam as stock options na Coverflex?"),
    ("👋", "Novo colaborador",     "Quais são os primeiros passos como novo colaborador Coverflex?"),
]

# WELCOME SCREEN
if st.session_state.show_welcome and not st.session_state.messages:

    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
                padding:60px 20px 32px;text-align:center;">
      <div style="width:76px;height:76px;background:#F07855;border-radius:22px;
                  display:flex;align-items:center;justify-content:center;
                  margin-bottom:22px;box-shadow:0 10px 30px rgba(240,120,85,0.28);">
        <span style="color:white;font-size:26px;font-weight:800;letter-spacing:-1px;">CF</span>
      </div>
      <h1 style="font-size:2rem;font-weight:700;color:#1C1B2E;margin:0 0 10px;
                 font-family:'DM Sans',sans-serif;letter-spacing:-0.5px;">
        Bem-vindo ao Coverflex AI
      </h1>
      <p style="font-size:14px;color:#8C877F;max-width:440px;line-height:1.7;margin:0 0 6px;">
        O teu assistente interno para benefícios, compensação e cultura Coverflex.
      </p>
      <p style="font-size:11px;color:#C4BFB8;margin:0 0 40px;">
        Powered by Groq · Llama 3.3 70B · Coverflex Knowledge Base
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Suggestion grid — clicking the button IS the card
    c1, c2 = st.columns(2, gap="medium")
    cols = [c1, c2]
    for i, (icon, label, question) in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            # Full-width styled button that looks like a card
            st.markdown(f"""
            <style>
            div[data-testid="stButton"] > button[key="s_{label}"] {{
                background: white !important;
                border: 1px solid #E8E4DF !important;
                border-radius: 14px !important;
                padding: 16px 18px !important;
                text-align: left !important;
                height: auto !important;
                display: flex !important;
                align-items: center !important;
                gap: 12px !important;
                margin-bottom: 10px !important;
                font-size: 13px !important;
                font-weight: 500 !important;
                color: #2A2724 !important;
                transition: all 0.15s !important;
                white-space: normal !important;
            }}
            div[data-testid="stButton"] > button[key="s_{label}"]:hover {{
                border-color: #F5B09A !important;
                background: #FEF0EB !important;
                color: #C95A35 !important;
            }}
            </style>
            """, unsafe_allow_html=True)

            if st.button(
                f"{icon}  **{label}**",
                key=f"s_{label}",
                use_container_width=True,
                help=question
            ):
                st.session_state.quick_q = question
                st.session_state.show_welcome = False
                if not st.session_state.messages:
                    st.session_state.cid = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.rerun()

# CHAT VIEW
else:
    st.markdown('<div style="max-width:720px;margin:0 auto;padding:28px 20px 120px;">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    st.markdown('</div>', unsafe_allow_html=True)

# ── CHAT INPUT ─────────────────────────────────────────────
st.markdown('<div style="max-width:720px;margin:0 auto;padding:0 20px 20px;">', unsafe_allow_html=True)
if prompt := st.chat_input("Pergunta sobre a Coverflex..."):
    st.session_state.show_welcome = False
    if not st.session_state.messages:
        st.session_state.cid = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_conversation(st.session_state.cid, st.session_state.messages)
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("A pensar..."):
            docs = st.session_state.vectorstore.similarity_search(prompt, k=3) if st.session_state.vectorstore else []
            resp = generate_response(prompt, st.session_state.messages[:-1], docs)
        st.markdown(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})
        save_conversation(st.session_state.cid, st.session_state.messages)
st.markdown('</div>', unsafe_allow_html=True)

# ── QUICK QUESTION ─────────────────────────────────────────
if st.session_state.quick_q:
    q = st.session_state.quick_q
    st.session_state.quick_q = None
    st.session_state.show_welcome = False
    st.session_state.messages.append({"role": "user", "content": q})
    save_conversation(st.session_state.cid, st.session_state.messages)
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("A pensar..."):
            docs = st.session_state.vectorstore.similarity_search(q, k=3) if st.session_state.vectorstore else []
            resp = generate_response(q, st.session_state.messages[:-1], docs)
        st.markdown(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})
        save_conversation(st.session_state.cid, st.session_state.messages)
    st.rerun()
