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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&display=swap');

#MainMenu, footer, header { visibility: hidden; }
*, *::before, *::after { font-family: 'DM Sans', sans-serif !important; box-sizing: border-box; }

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
}

.stApp { background: var(--bg) !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: none !important;
    min-width: 260px !important;
    max-width: 260px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* Sidebar text colors on dark bg */
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }

/* Collapse button */
[data-testid="collapsedControl"] {
    background: var(--peach) !important;
    border-radius: 0 10px 10px 0 !important;
    top: 50% !important;
    color: white !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 9px !important;
    color: rgba(255,255,255,0.82) !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(240,120,85,0.25) !important;
    border-color: rgba(240,120,85,0.5) !important;
    color: white !important;
}

/* ── CHAT INPUT ── */
.stChatInput > div {
    border-radius: 14px !important;
    border: 1.5px solid var(--border) !important;
    background: var(--white) !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06) !important;
}
.stChatInput > div:focus-within {
    border-color: var(--peach) !important;
    box-shadow: 0 0 0 3px rgba(240,120,85,0.12) !important;
}
.stChatInput textarea {
    font-size: 14px !important;
    padding: 14px 18px !important;
    background: transparent !important;
}
.stChatInput textarea::placeholder { color: var(--muted) !important; }

/* ── MESSAGES ── */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
    max-width: 740px !important;
    margin: 0 auto !important;
    box-shadow: none !important;
}
[data-testid="stChatMessageContent"] p {
    font-size: 14px !important;
    line-height: 1.8 !important;
    color: var(--text) !important;
    margin-bottom: 8px !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 16px 20px !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--peach-light) !important;
    border: 1px solid var(--peach-mid) !important;
    border-radius: 16px !important;
    padding: 14px 20px !important;
}

/* Main area buttons (related questions) */
.main .stButton > button {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-size: 12px !important;
    padding: 8px 14px !important;
    text-align: left !important;
    transition: all 0.15s !important;
}
.main .stButton > button:hover {
    background: var(--peach-light) !important;
    border-color: var(--peach-mid) !important;
    color: var(--peach-dark) !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--peach) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── FUNCTIONS ──────────────────────────────────────────────
MEMORY_DIR = "chat_histories"
VECTORSTORE_DIR = "coverflex_vectorstore"
os.makedirs(MEMORY_DIR, exist_ok=True)

RELATED = {
    "cartão": ["Como consulto o meu saldo?", "Onde posso usar o cartão Coverflex?", "Como bloqueio o cartão se o perder?"],
    "refeição": ["Posso usar o cartão em apps de entrega?", "Qual é o limite de vales por transação?", "Como funciona o saldo de refeição?"],
    "benefícios": ["Como ativo os meus benefícios?", "Que categorias de benefícios existem?", "Como faço um reembolso?"],
    "férias": ["Como solicito dias de férias?", "O que são os Caring Days?", "Qual é a licença parental?"],
    "seguro": ["Como adiciono dependentes ao seguro?", "Quais são os períodos de carência?", "Onde posso usar o seguro de saúde?"],
    "remote": ["O que é o Personal Growth Budget?", "O que é o Onboarding Budget?", "Como funciona o trabalho remoto?"],
    "stock": ["O que é o VSOP?", "Como são geridas as stock options?", "Quais são os outros benefícios internos?"],
    "reembolso": ["Quanto tempo demora o reembolso?", "Que documentos preciso?", "Quais categorias permitem reembolso?"],
    "onboarding": ["Que ferramentas vou precisar?", "Quando recebo o cartão Coverflex?", "Como adiciono dependentes ao seguro?"],
    "default": ["Como funciona o cartão Coverflex?", "Quais são os benefícios disponíveis?", "Como contacto o suporte?"]
}

def get_related(question):
    q = question.lower()
    for key in RELATED:
        if key in q:
            return RELATED[key]
    return RELATED["default"]

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
    st.session_state.last_question = None
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

Respondes SEMPRE em Português de Portugal. Tom profissional, claro e direto.

Regras:
1. Usa a informação da knowledge base disponível
2. Se não tiveres informação, indica qual equipa contactar (RH: rh@coverflex.com, Suporte: help@coverflex.com)
3. Sê concreto e específico
4. Não comeces sempre com "Olá!" — varia a abertura"""

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
    st.session_state.last_question = None
if "quick_q" not in st.session_state:
    st.session_state.quick_q = None
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True
if "last_question" not in st.session_state:
    st.session_state.last_question = None

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding:24px 18px 16px;border-bottom:1px solid rgba(255,255,255,0.1);">
      <div style="display:flex;align-items:center;gap:10px;">
        <div style="width:38px;height:38px;background:#F07855;border-radius:11px;
                    display:flex;align-items:center;justify-content:center;
                    box-shadow:0 4px 14px rgba(240,120,85,0.4);flex-shrink:0;">
          <span style="color:white;font-size:14px;font-weight:800;letter-spacing:-0.5px;">CF</span>
        </div>
        <div>
          <div style="font-size:15px;font-weight:700;color:white;line-height:1.2;">Coverflex</div>
          <div style="font-size:10px;color:#F07855;font-weight:600;
                      letter-spacing:0.07em;text-transform:uppercase;">Knowledge Assistant</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding:14px 14px 6px;">', unsafe_allow_html=True)
    if st.button("＋  Nova Conversa", use_container_width=True, key="new_chat"):
        new_conversation()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # KB status
    kb_ok = st.session_state.vectorstore is not None
    st.markdown(f"""
    <div style="margin:2px 14px 12px;padding:9px 13px;
                background:{'rgba(163,217,188,0.15)' if kb_ok else 'rgba(245,208,154,0.15)'};
                border:1px solid {'rgba(163,217,188,0.4)' if kb_ok else 'rgba(245,208,154,0.4)'};
                border-radius:9px;font-size:12px;font-weight:500;
                color:{'#7DDFB0' if kb_ok else '#F5C97A'};">
      {'✅  Base de conhecimento ativa' if kb_ok else '⚠️  Sem documentos carregados'}
    </div>
    """, unsafe_allow_html=True)

    # Topics
    st.markdown("""
    <div style="padding:4px 18px 6px;">
      <span style="font-size:10px;font-weight:600;letter-spacing:0.1em;
                   color:rgba(255,255,255,0.35);text-transform:uppercase;">Tópicos</span>
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
        ("📞", "Contactos",           "Quais os contactos do suporte Coverflex?"),
    ]

    for icon, label, question in TOPICS:
        if st.button(f"{icon}  {label}", key=f"t_{label}", use_container_width=True):
            st.session_state.quick_q = question
            st.session_state.show_welcome = False
            if not st.session_state.messages:
                st.session_state.cid = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.rerun()

    # Recent
    convs = get_conversation_files()
    if convs:
        st.markdown("""
        <div style="padding:12px 18px 6px;margin-top:4px;
                    border-top:1px solid rgba(255,255,255,0.08);">
          <span style="font-size:10px;font-weight:600;letter-spacing:0.1em;
                       color:rgba(255,255,255,0.35);text-transform:uppercase;">Recentes</span>
        </div>
        """, unsafe_allow_html=True)
        for conv in convs[:6]:
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
    st.markdown("""
    <div style="border-top:1px solid rgba(255,255,255,0.08);
                padding:10px 14px 16px;margin-top:8px;">
    """, unsafe_allow_html=True)
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

# ── MAIN ───────────────────────────────────────────────────

# Colored top bar
st.markdown("""
<div style="background:linear-gradient(135deg,#F07855 0%,#E85D3A 100%);
            padding:14px 32px;display:flex;align-items:center;
            justify-content:space-between;box-shadow:0 2px 12px rgba(240,120,85,0.25);">
  <div style="display:flex;align-items:center;gap:10px;">
    <div style="width:30px;height:30px;background:rgba(255,255,255,0.2);
                border-radius:8px;display:flex;align-items:center;justify-content:center;">
      <span style="color:white;font-size:12px;font-weight:800;">CF</span>
    </div>
    <span style="color:white;font-size:15px;font-weight:600;">Coverflex AI Assistant</span>
  </div>
  <span style="color:rgba(255,255,255,0.7);font-size:11px;">
    Powered by Groq · Llama 3.3 70B
  </span>
</div>
""", unsafe_allow_html=True)

# WELCOME SCREEN
if st.session_state.show_welcome and not st.session_state.messages:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
                padding:50px 20px 28px;text-align:center;">
      <div style="width:72px;height:72px;background:linear-gradient(135deg,#F07855,#E85D3A);
                  border-radius:20px;display:flex;align-items:center;justify-content:center;
                  margin-bottom:20px;box-shadow:0 10px 28px rgba(240,120,85,0.3);">
        <span style="color:white;font-size:24px;font-weight:800;letter-spacing:-1px;">CF</span>
      </div>
      <h1 style="font-size:1.9rem;font-weight:700;color:#1C1B2E;margin:0 0 10px;
                 letter-spacing:-0.5px;">Bem-vindo ao Coverflex AI</h1>
      <p style="font-size:14px;color:#8C877F;max-width:420px;line-height:1.7;margin:0 0 40px;">
        O teu assistente interno para benefícios, compensação e cultura Coverflex.
        Pergunta qualquer coisa.
      </p>
    </div>
    """, unsafe_allow_html=True)

    SUGGESTIONS = [
        ("🍽️", "Cartão de refeição",   "Qual é o limite diário do cartão de refeição Coverflex?"),
        ("🎁", "Benefícios flexíveis", "Como funcionam os benefícios flexíveis da Coverflex?"),
        ("🏥", "Seguro de saúde",      "Como funciona o seguro de saúde na Coverflex?"),
        ("🏠", "Remote Work",          "O que é o Remote Work Budget da Coverflex?"),
        ("📈", "Stock Options",        "Como funcionam as stock options na Coverflex?"),
        ("👋", "Novo colaborador",     "Quais são os primeiros passos como novo colaborador?"),
    ]

    c1, c2 = st.columns(2, gap="medium")
    for i, (icon, label, question) in enumerate(SUGGESTIONS):
        with (c1 if i % 2 == 0 else c2):
            if st.button(
                f"{icon}  {label}",
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
    st.markdown('<div style="max-width:740px;margin:0 auto;padding:24px 20px 20px;">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Related questions after last answer
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last_q = st.session_state.last_question or ""
        related = get_related(last_q)

        st.markdown("""
        <div style="max-width:740px;margin:16px auto 4px;padding:0 20px;">
          <div style="font-size:11px;font-weight:600;color:#B5AFA8;
                      letter-spacing:0.08em;text-transform:uppercase;
                      margin-bottom:10px;">💡 Perguntas relacionadas</div>
        </div>
        """, unsafe_allow_html=True)

        rcols = st.columns(3)
        for i, rq in enumerate(related):
            with rcols[i % 3]:
                if st.button(rq, key=f"rel_{rq}_{len(st.session_state.messages)}", use_container_width=True):
                    st.session_state.quick_q = rq
                    st.rerun()

        # New conversation button
        st.markdown('<div style="text-align:center;margin:16px 0 8px;">', unsafe_allow_html=True)
        if st.button("↩  Voltar ao início", key="back_home"):
            new_conversation()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# CHAT INPUT
st.markdown('<div style="max-width:740px;margin:0 auto;padding:0 20px 20px;">', unsafe_allow_html=True)
if prompt := st.chat_input("Pergunta sobre a Coverflex..."):
    st.session_state.show_welcome = False
    st.session_state.last_question = prompt
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

# QUICK QUESTION HANDLER
if st.session_state.quick_q:
    q = st.session_state.quick_q
    st.session_state.quick_q = None
    st.session_state.show_welcome = False
    st.session_state.last_question = q
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
