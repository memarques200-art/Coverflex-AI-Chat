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
    --peach-2: #F28C6E;
    --peach-light: #FEF0EB;
    --peach-mid: #F5B09A;
    --peach-dark: #C95A35;
    --white: #FFFFFF;
    --bg: #FDF8F6;
    --border: #EDE8E3;
    --text: #1C1B2E;
    --muted: #9C968E;
    --sidebar-w: 300px;
}

.stApp { background: var(--bg) !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--border) !important;
    min-width: var(--sidebar-w) !important;
    max-width: var(--sidebar-w) !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.04) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; overflow-x: hidden !important; }
[data-testid="collapsedControl"] {
    background: var(--peach) !important;
    color: white !important;
    border-radius: 0 10px 10px 0 !important;
    top: 50% !important;
    box-shadow: 2px 0 8px rgba(240,120,85,0.3) !important;
}

/* Sidebar all buttons reset */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    color: #3A3530 !important;
    font-size: 13.5px !important;
    font-weight: 400 !important;
    padding: 9px 14px !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.15s !important;
    display: flex !important;
    align-items: center !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--peach-light) !important;
    color: var(--peach-dark) !important;
}

/* ── MAIN BUTTONS ── */
.main .stButton > button {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-size: 13px !important;
    padding: 10px 16px !important;
    transition: all 0.15s !important;
    font-weight: 400 !important;
}
.main .stButton > button:hover {
    background: var(--peach-light) !important;
    border-color: var(--peach-mid) !important;
    color: var(--peach-dark) !important;
}

/* ── CHAT INPUT ── */
.stChatInput > div {
    border-radius: 16px !important;
    border: 1.5px solid var(--border) !important;
    background: var(--white) !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06) !important;
    transition: all 0.2s !important;
}
.stChatInput > div:focus-within {
    border-color: var(--peach) !important;
    box-shadow: 0 0 0 3px rgba(240,120,85,0.1) !important;
}
.stChatInput textarea {
    font-size: 14px !important;
    padding: 14px 18px !important;
    background: transparent !important;
    color: var(--text) !important;
}
.stChatInput textarea::placeholder { color: var(--muted) !important; }

/* ── MESSAGES ── */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
    max-width: 720px !important;
    margin: 0 auto !important;
    box-shadow: none !important;
}
[data-testid="stChatMessageContent"] p {
    font-size: 14px !important;
    line-height: 1.8 !important;
    color: var(--text) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 18px 22px !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--peach-light) !important;
    border: 1px solid var(--peach-mid) !important;
    border-radius: 16px !important;
    padding: 14px 20px !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--peach) !important; }

/* Search input */
[data-testid="stSidebar"] input[type="text"] {
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    background: #F7F4F1 !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

section.main > div { padding-bottom: 20px !important; }
</style>
""", unsafe_allow_html=True)

# ── FUNCTIONS ──────────────────────────────────────────────
MEMORY_DIR = "chat_histories"
VECTORSTORE_DIR = "coverflex_vectorstore"
os.makedirs(MEMORY_DIR, exist_ok=True)

CATEGORIES = {
    "General": {
        "icon": "✨", "color": "#F07855", "bg": "#FEF0EB",
        "question": "Como posso ajudar-te hoje?"
    },
    "Cartão Refeição": {
        "icon": "🍽️", "color": "#E05A8A", "bg": "#FDE8F2",
        "question": "Qual é o limite diário do cartão de refeição?"
    },
    "Benefícios": {
        "icon": "🎁", "color": "#7B61FF", "bg": "#F0EDFF",
        "question": "Como funcionam os benefícios flexíveis?"
    },
    "Seguro Saúde": {
        "icon": "❤️", "color": "#E0453A", "bg": "#FDE8E7",
        "question": "Como funciona o seguro de saúde?"
    },
    "Descontos": {
        "icon": "🏷️", "color": "#2DAE7A", "bg": "#E6F7F1",
        "question": "Que descontos estão disponíveis?"
    },
    "Reembolsos": {
        "icon": "🔄", "color": "#0EA5E9", "bg": "#E0F5FE",
        "question": "Como faço um pedido de reembolso?"
    },
}

RELATED = {
    "cartão": ["Como consulto o meu saldo?", "Onde posso usar o cartão?", "Como bloqueio o cartão?"],
    "refeição": ["Posso usar em apps de entrega?", "Qual o limite por transação?", "Como recarrego o saldo?"],
    "benefícios": ["Como ativo os benefícios?", "Que categorias existem?", "Como faço um reembolso?"],
    "férias": ["Como solicito dias de férias?", "O que são os Caring Days?", "Qual é a licença parental?"],
    "seguro": ["Como adiciono dependentes?", "Quais são os períodos de carência?", "Onde usar o seguro?"],
    "remote": ["O que é o Personal Growth Budget?", "O que é o Onboarding Budget?", "Como funciona o trabalho remoto?"],
    "stock": ["O que é o VSOP?", "Como são geridas as stock options?", "Quais são os outros benefícios?"],
    "reembolso": ["Quanto tempo demora?", "Que documentos preciso?", "Quais categorias permitem reembolso?"],
    "onboarding": ["Que ferramentas vou precisar?", "Quando recebo o cartão?", "Como adiciono dependentes?"],
    "default": ["Como funciona o cartão Coverflex?", "Quais são os benefícios disponíveis?", "Como contacto o suporte?"]
}

def get_related(q):
    q = q.lower()
    for k in RELATED:
        if k in q:
            return RELATED[k]
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
    st.session_state.last_q = None
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
    system = """És o assistente de conhecimento interno da Coverflex — fintech portuguesa líder em benefícios flexíveis, fundada em 2021, presente em Portugal, Espanha e Itália.

Respondes SEMPRE em Português de Portugal. Tom profissional, claro e direto — como um colega sénior bem informado.

Regras:
1. Usa a informação da knowledge base
2. Se não tiveres informação, indica qual equipa contactar (RH: rh@coverflex.com, Suporte: help@coverflex.com)
3. Sê concreto e específico
4. Não comeces sempre com "Olá!" — varia as aberturas
5. Usa formatação clara quando útil (listas, negrito)"""

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
    st.session_state.last_q = None
if "quick_q" not in st.session_state:
    st.session_state.quick_q = None
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True
if "last_q" not in st.session_state:
    st.session_state.last_q = None
if "search_term" not in st.session_state:
    st.session_state.search_term = ""

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:

    # Logo header
    st.markdown("""
    <div style="padding:20px 20px 14px;border-bottom:1px solid #EDE8E3;">
      <div style="display:flex;align-items:center;gap:10px;">
        <div style="width:40px;height:40px;background:linear-gradient(135deg,#F07855,#E85D3A);
                    border-radius:12px;display:flex;align-items:center;justify-content:center;
                    box-shadow:0 4px 12px rgba(240,120,85,0.3);flex-shrink:0;">
          <span style="color:white;font-size:18px;">✦</span>
        </div>
        <div>
          <div style="font-size:16px;font-weight:700;color:#1C1B2E;line-height:1.2;">Coverflex</div>
          <div style="font-size:10px;color:#F07855;font-weight:600;
                      letter-spacing:0.08em;text-transform:uppercase;">AI Assistant</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # New Chat button
    st.markdown('<div style="padding:14px 16px 10px;">', unsafe_allow_html=True)
    if st.button("＋  Nova Conversa", use_container_width=True, key="new_chat"):
        new_conversation()
        st.rerun()
    # Style the new chat button specially
    st.markdown("""
    <style>
    [data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type > button {
        background: linear-gradient(135deg,#F07855,#E85D3A) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 3px 10px rgba(240,120,85,0.3) !important;
        border-radius: 12px !important;
        padding: 11px 16px !important;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type > button:hover {
        opacity: 0.9 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Categories
    st.markdown("""
    <div style="padding:4px 20px 8px;">
      <span style="font-size:10px;font-weight:600;letter-spacing:0.1em;
                   color:#B5AFA8;text-transform:uppercase;">Knowledge Base</span>
    </div>
    """, unsafe_allow_html=True)

    for cat_name, cat_data in CATEGORIES.items():
        col_a, col_b = st.columns([1, 8])
        with col_a:
            st.markdown(f"""
            <div style="width:30px;height:30px;background:{cat_data['bg']};border-radius:8px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:14px;margin-top:4px;">
                {cat_data['icon']}
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            if st.button(cat_name, key=f"cat_{cat_name}", use_container_width=True):
                st.session_state.quick_q = cat_data['question']
                st.session_state.show_welcome = False
                if not st.session_state.messages:
                    st.session_state.cid = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.rerun()

    # Search + Recent chats
    convs = get_conversation_files()
    if convs:
        st.markdown("""
        <div style="padding:14px 16px 6px;border-top:1px solid #EDE8E3;margin-top:8px;">
          <span style="font-size:10px;font-weight:600;letter-spacing:0.1em;
                       color:#B5AFA8;text-transform:uppercase;">Conversas Recentes</span>
        </div>
        """, unsafe_allow_html=True)

        # Search box
        search = st.text_input("", placeholder="🔍  Pesquisar conversas...",
                               key="search_box", label_visibility="collapsed")

        for conv in convs[:8]:
            cid = conv.replace('.pkl', '')
            try:
                d = datetime.strptime(cid, "%Y%m%d_%H%M%S")
                name = d.strftime("%d/%m · %H:%M")
            except:
                name = cid[:12]

            if search and search.lower() not in name.lower():
                continue

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
    <div style="border-top:1px solid #EDE8E3;padding:10px 16px 16px;margin-top:8px;">
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

# Breadcrumb nav bar
current_page = "New Chat" if st.session_state.show_welcome else "Conversa"
st.markdown(f"""
<div style="background:white;border-bottom:1px solid #EDE8E3;
            padding:12px 28px;display:flex;align-items:center;gap:8px;">
  <a href="#" style="color:#9C968E;font-size:13px;text-decoration:none;">🏠 Início</a>
  <span style="color:#C8C3BC;font-size:13px;">›</span>
  <span style="color:#1C1B2E;font-size:13px;font-weight:500;">{current_page}</span>
  <div style="margin-left:auto;display:flex;align-items:center;gap:6px;">
    <div style="width:8px;height:8px;border-radius:50%;
                background:{'#2DAE7A' if st.session_state.vectorstore else '#F0A030'};"></div>
    <span style="font-size:11px;color:#9C968E;">
      {'KB Ativa' if st.session_state.vectorstore else 'Sem KB'}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# WELCOME SCREEN
if st.session_state.show_welcome and not st.session_state.messages:

    # Peach gradient hero
    st.markdown("""
    <div style="background:linear-gradient(160deg,#FEF0EB 0%,#FDF8F6 60%,#FDF8F6 100%);
                padding:60px 40px 40px;text-align:center;position:relative;overflow:hidden;">
      <div style="position:absolute;top:-40px;right:-40px;width:200px;height:200px;
                  background:radial-gradient(circle,rgba(240,120,85,0.12),transparent 70%);
                  border-radius:50%;"></div>
      <div style="position:absolute;bottom:-30px;left:-30px;width:150px;height:150px;
                  background:radial-gradient(circle,rgba(240,120,85,0.08),transparent 70%);
                  border-radius:50%;"></div>
      <div style="width:80px;height:80px;
                  background:linear-gradient(135deg,#F07855 0%,#E85D3A 100%);
                  border-radius:24px;display:inline-flex;align-items:center;
                  justify-content:center;margin-bottom:24px;
                  box-shadow:0 12px 32px rgba(240,120,85,0.3);">
        <span style="color:white;font-size:36px;">✦</span>
      </div>
      <h1 style="font-size:2.2rem;font-weight:700;color:#1C1B2E;margin:0 0 12px;
                 letter-spacing:-0.5px;">
        Bem-vindo ao <span style="color:#F07855;">Coverflex AI</span>
      </h1>
      <p style="font-size:15px;color:#6C6660;max-width:460px;margin:0 auto 8px;line-height:1.7;">
        O teu assistente inteligente para benefícios e compensação flexível.
      </p>
      <p style="font-size:11px;color:#B5AFA8;margin:0;">
        ✦ Powered by Coverflex AI · Verifica sempre informações importantes
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Suggestion cards
    st.markdown('<div style="padding:28px 32px 0;">', unsafe_allow_html=True)

    SUGGESTIONS = [
        ("🍽️", "#FDE8F2", "Qual é o limite diário do cartão de refeição Coverflex?",  "Cartão Refeição", "›"),
        ("🎁", "#F0EDFF", "Como funcionam os benefícios flexíveis da Coverflex?",      "Benefícios",     "›"),
        ("🏥", "#FDE8E7", "Como funciona o seguro de saúde na Coverflex?",             "Seguro Saúde",   "›"),
        ("🏠", "#FEF0EB", "O que é o Remote Work Budget da Coverflex?",                "Remote Work",    "›"),
        ("📈", "#E6F7F1", "Como funcionam as stock options na Coverflex?",             "Stock Options",  "›"),
        ("👋", "#FEF0EB", "Quais são os primeiros passos como novo colaborador?",      "Onboarding",     "›"),
    ]

    c1, c2 = st.columns(2, gap="medium")
    for i, (icon, bg, question, label, arrow) in enumerate(SUGGESTIONS):
        with (c1 if i % 2 == 0 else c2):
            if st.button(
                f"{icon}  {label}  {arrow}",
                key=f"s_{label}",
                use_container_width=True,
                help=question
            ):
                st.session_state.quick_q = question
                st.session_state.show_welcome = False
                if not st.session_state.messages:
                    st.session_state.cid = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align:center;padding:32px;color:#B5AFA8;font-size:11px;">
      ✦ Powered by Coverflex AI · Always verify important information
    </div>
    """, unsafe_allow_html=True)

# CHAT VIEW
else:
    st.markdown('<div style="max-width:720px;margin:0 auto;padding:24px 20px 20px;">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Related questions after last assistant message
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        related = get_related(st.session_state.last_q or "")
        st.markdown("""
        <div style="margin:20px 0 10px;text-align:center;">
          <span style="font-size:10px;font-weight:600;color:#B5AFA8;
                       letter-spacing:0.1em;text-transform:uppercase;">
            💡 Perguntas relacionadas
          </span>
        </div>
        """, unsafe_allow_html=True)

        rcols = st.columns(3)
        for i, rq in enumerate(related):
            with rcols[i]:
                if st.button(rq, key=f"r_{rq}_{len(st.session_state.messages)}", use_container_width=True):
                    st.session_state.quick_q = rq
                    st.rerun()

        st.markdown('<div style="text-align:center;margin:12px 0 4px;">', unsafe_allow_html=True)
        if st.button("🏠  Voltar ao início", key="back_home"):
            new_conversation()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# CHAT INPUT
st.markdown('<div style="max-width:720px;margin:0 auto;padding:0 20px 24px;">', unsafe_allow_html=True)
if prompt := st.chat_input("Pergunta sobre benefícios Coverflex..."):
    st.session_state.show_welcome = False
    st.session_state.last_q = prompt
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
    st.session_state.last_q = q
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
