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

# ============================================
# CONFIGURAR GROQ
# ============================================
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ============================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================
st.set_page_config(
    page_title="Coverflex AI",
    page_icon="https://www.coverflex.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

    #MainMenu, footer, header { visibility: hidden; }
    * { font-family: 'DM Sans', sans-serif; }

    :root {
        --cf-peach: #F4886C;
        --cf-peach-light: #FDF1ED;
        --cf-peach-mid: #F7B49E;
        --cf-peach-dark: #D4613F;
        --cf-navy: #1A1A2E;
        --cf-white: #FFFFFF;
        --cf-gray-50: #FAFAF8;
        --cf-gray-100: #F5F3F0;
        --cf-gray-200: #EAE7E2;
        --cf-gray-400: #B5B0A8;
        --cf-gray-600: #6B6560;
        --cf-gray-800: #2E2B28;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 20px;
    }

    .stApp { background: var(--cf-gray-50); }

    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: var(--cf-white) !important;
        border-right: 1px solid var(--cf-gray-200) !important;
        min-width: 280px !important;
        max-width: 280px !important;
    }

    [data-testid="stSidebar"] > div {
        padding: 0 !important;
    }

    /* Force sidebar always visible */
    [data-testid="collapsedControl"] {
        display: flex !important;
        background: var(--cf-peach) !important;
        color: white !important;
        border-radius: 0 var(--radius-md) var(--radius-md) 0 !important;
        top: 50% !important;
    }

    /* ── BUTTONS ── */
    .stButton > button {
        background: transparent !important;
        border: 1px solid var(--cf-gray-200) !important;
        border-radius: var(--radius-md) !important;
        color: var(--cf-gray-800) !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        padding: 8px 14px !important;
        transition: all 0.15s ease !important;
        text-align: left !important;
        width: 100% !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .stButton > button:hover {
        background: var(--cf-peach-light) !important;
        border-color: var(--cf-peach-mid) !important;
        color: var(--cf-peach-dark) !important;
    }

    /* ── CHAT INPUT ── */
    .stChatInput {
        padding: 16px 24px !important;
        background: var(--cf-white) !important;
        border-top: 1px solid var(--cf-gray-200) !important;
        position: fixed !important;
        bottom: 0 !important;
        right: 0 !important;
        left: 280px !important;
        z-index: 100 !important;
    }

    .stChatInput textarea {
        border-radius: 40px !important;
        border: 1.5px solid var(--cf-gray-200) !important;
        background: var(--cf-gray-50) !important;
        font-size: 14px !important;
        padding: 12px 20px !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: all 0.15s !important;
    }

    .stChatInput textarea:focus {
        border-color: var(--cf-peach) !important;
        background: var(--cf-white) !important;
        box-shadow: 0 0 0 3px rgba(244,136,108,0.15) !important;
    }

    /* ── CHAT MESSAGES ── */
    .stChatMessage {
        background: transparent !important;
        padding: 8px 24px !important;
        max-width: 760px !important;
        margin: 0 auto !important;
    }

    [data-testid="stChatMessageContent"] {
        font-size: 14px !important;
        line-height: 1.75 !important;
        color: var(--cf-gray-800) !important;
    }

    /* User message bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: var(--cf-peach-light) !important;
        border-radius: var(--radius-lg) !important;
        border: 1px solid var(--cf-peach-mid) !important;
        padding: 12px 20px !important;
    }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--cf-gray-200); border-radius: 4px; }

    /* ── MAIN CONTENT PADDING for fixed input ── */
    .main { padding-bottom: 80px !important; }

    /* ── WELCOME CARDS ── */
    .welcome-card {
        background: white;
        border: 1px solid #EAE7E2;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 10px;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .welcome-card:hover {
        border-color: #F7B49E;
        background: #FDF1ED;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# FUNÇÕES
# ============================================
MEMORY_DIR = "chat_histories"
VECTORSTORE_DIR = "coverflex_vectorstore"

if not os.path.exists(MEMORY_DIR):
    os.makedirs(MEMORY_DIR)

def get_conversation_files():
    files = [f for f in os.listdir(MEMORY_DIR) if f.endswith('.pkl')]
    files.sort(reverse=True)
    return files

def load_conversation(conversation_id):
    filepath = os.path.join(MEMORY_DIR, f"{conversation_id}.pkl")
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return []

def save_conversation(conversation_id, messages):
    filepath = os.path.join(MEMORY_DIR, f"{conversation_id}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(messages, f)

def create_new_conversation():
    conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.current_conversation = conversation_id
    st.session_state.messages = []
    st.session_state.show_welcome = True
    save_conversation(conversation_id, [])

def load_vectorstore():
    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            return Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)
        except:
            return None
    return None

def generate_response(user_message, conversation_history, relevant_docs):
    context = ""
    if relevant_docs:
        context = "Informação oficial da Coverflex:\n\n"
        for doc in relevant_docs[:3]:
            context += f"{doc.page_content[:600]}\n\n"

    chat_history = ""
    for msg in conversation_history[-6:]:
        role = "Utilizador" if msg["role"] == "user" else "Assistente"
        chat_history += f"{role}: {msg['content']}\n"

    system_prompt = """És o assistente de conhecimento interno da Coverflex — uma fintech portuguesa líder em gestão de benefícios flexíveis, fundada em 2021, presente em Portugal, Espanha e Itália.

Respondes SEMPRE em Português de Portugal, de forma clara, direta e útil. Tom profissional mas acessível — como um colega sénior bem informado.

Regras:
1. Responde com base na informação disponível
2. Se não tiveres informação suficiente, indica qual equipa contactar (RH, Suporte, IT)
3. Sê específico e concreto — evita generalidades
4. Valores Coverflex: Extreme Employee Obsession, transparência, simplicidade"""

    user_content = ""
    if context:
        user_content += f"{context}\n\n"
    if chat_history:
        user_content += f"Conversa anterior:\n{chat_history}\n\n"
    user_content += f"Pergunta: {user_message}"

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Erro: {str(e)}"

# ============================================
# INICIALIZAÇÃO — always start fresh on page load
# ============================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()

# Always open to welcome screen on fresh page load
if "app_initialized" not in st.session_state:
    st.session_state.app_initialized = True
    st.session_state.current_conversation = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.messages = []
    st.session_state.show_welcome = True

if "quick_question" not in st.session_state:
    st.session_state.quick_question = None

if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    # Logo header
    st.markdown("""
    <div style="padding:24px 20px 16px;border-bottom:1px solid #EAE7E2;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:36px;height:36px;background:#F4886C;border-radius:10px;
                        display:flex;align-items:center;justify-content:center;
                        box-shadow:0 4px 12px rgba(244,136,108,0.3);">
                <span style="color:white;font-size:17px;font-weight:800;
                             font-family:'DM Sans',sans-serif;letter-spacing:-1px;">CF</span>
            </div>
            <div>
                <div style="font-size:15px;font-weight:700;color:#1A1A2E;">Coverflex</div>
                <div style="font-size:10px;color:#F4886C;font-weight:600;
                            letter-spacing:0.08em;text-transform:uppercase;">Knowledge Assistant</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # New Chat
    st.markdown('<div style="padding:14px 16px 8px;">', unsafe_allow_html=True)
    if st.button("＋  Nova Conversa", use_container_width=True,
                 help="Iniciar uma nova conversa"):
        create_new_conversation()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Knowledge Base status
    st.markdown("""
    <div style="padding:4px 20px 8px;">
        <div style="font-size:10px;font-weight:600;letter-spacing:0.1em;
                    color:#B5B0A8;text-transform:uppercase;margin-bottom:6px;">
            Knowledge Base
        </div>
    </div>
    """, unsafe_allow_html=True)

    kb_ok = st.session_state.vectorstore is not None
    st.markdown(f"""
    <div style="margin:0 16px 12px;padding:10px 14px;
                background:{'#F0FBF6' if kb_ok else '#FFF8F0'};
                border:1px solid {'#A8DFC3' if kb_ok else '#F7D4A8'};
                border-radius:10px;font-size:12px;
                color:{'#1E7A4A' if kb_ok else '#B85C00'};font-weight:500;">
        {'✅  Base de conhecimento ativa' if kb_ok else '⚠️  Sem documentos carregados'}
    </div>
    """, unsafe_allow_html=True)

    # Quick topics
    st.markdown("""
    <div style="padding:8px 20px 6px;">
        <div style="font-size:10px;font-weight:600;letter-spacing:0.1em;
                    color:#B5B0A8;text-transform:uppercase;">Tópicos Rápidos</div>
    </div>
    """, unsafe_allow_html=True)

    topics = [
        ("💳", "Cartão Coverflex", "Como ativo o cartão Coverflex?"),
        ("🏖️", "Férias & Tempo Livre", "Quantos dias de férias tenho na Coverflex?"),
        ("🏠", "Remote Work Budget", "O que é o Remote Work Budget?"),
        ("📚", "Personal Growth", "O que é o Personal Growth Budget?"),
        ("👶", "Licença Parental", "Qual é a licença parental na Coverflex?"),
        ("❤️", "Caring Days", "O que são os Caring Days?"),
        ("📈", "Stock Options", "Como funcionam as stock options?"),
        ("🏥", "Seguro de Saúde", "Como funciona o seguro de saúde?"),
        ("🔄", "Reembolsos", "Como faço um pedido de reembolso?"),
        ("👋", "Onboarding", "Quais são os primeiros passos no onboarding?"),
        ("📞", "Contactos", "Quais os contactos do suporte Coverflex?"),
    ]

    for icon, label, question in topics:
        if st.button(f"{icon}  {label}", key=f"topic_{label}", use_container_width=True):
            st.session_state.quick_question = question
            st.session_state.show_welcome = False
            if not st.session_state.messages:
                st.session_state.current_conversation = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.rerun()

    # Divider
    st.markdown('<div style="height:1px;background:#EAE7E2;margin:12px 16px;"></div>', unsafe_allow_html=True)

    # Recent conversations
    conversations = get_conversation_files()
    if conversations:
        st.markdown("""
        <div style="padding:8px 20px 6px;">
            <div style="font-size:10px;font-weight:600;letter-spacing:0.1em;
                        color:#B5B0A8;text-transform:uppercase;">Conversas Recentes</div>
        </div>
        """, unsafe_allow_html=True)

        for conv in conversations[:8]:
            conv_id = conv.replace('.pkl', '')
            try:
                date_obj = datetime.strptime(conv_id, "%Y%m%d_%H%M%S")
                display_name = date_obj.strftime("%d/%m · %H:%M")
            except:
                display_name = conv_id[:12]

            is_active = conv_id == st.session_state.current_conversation
            if st.button(
                f"{'🟠' if is_active else '💬'}  {display_name}",
                key=f"hist_{conv_id}",
                use_container_width=True
            ):
                st.session_state.current_conversation = conv_id
                st.session_state.messages = load_conversation(conv_id)
                st.session_state.show_welcome = False
                st.rerun()

    # Train section
    st.markdown('<div style="height:1px;background:#EAE7E2;margin:12px 16px;"></div>', unsafe_allow_html=True)
    with st.expander("📁  Treinar com documentos"):
        uploaded_files = st.file_uploader(
            "PDF ou TXT", type=["pdf", "txt"],
            accept_multiple_files=True
        )
        if uploaded_files and st.button("🚀 Treinar agora", use_container_width=True):
            with st.spinner("A processar documentos..."):
                documents = []
                for file in uploaded_files:
                    temp_path = f"temp_{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    loader = PyPDFLoader(temp_path) if file.name.endswith('.pdf') \
                        else TextLoader(temp_path, encoding="utf-8")
                    documents.extend(loader.load())
                    os.remove(temp_path)
                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=50
                ).split_documents(documents)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vs = Chroma.from_documents(chunks, embeddings, persist_directory=VECTORSTORE_DIR)
                vs.persist()
                st.session_state.vectorstore = vs
                st.success(f"✅ {len(chunks)} fragmentos carregados!")
                st.rerun()

# ============================================
# MAIN AREA
# ============================================

# WELCOME SCREEN
if st.session_state.show_welcome and not st.session_state.messages:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;min-height:70vh;padding:40px 20px;text-align:center;">
        <div style="width:80px;height:80px;background:#F4886C;border-radius:22px;
                    display:flex;align-items:center;justify-content:center;
                    margin-bottom:24px;box-shadow:0 12px 32px rgba(244,136,108,0.3);">
            <span style="color:white;font-size:32px;font-weight:800;
                         font-family:'DM Sans',sans-serif;letter-spacing:-2px;">CF</span>
        </div>
        <h1 style="font-size:2.2rem;font-weight:700;color:#1A1A2E;
                   margin-bottom:10px;font-family:'DM Sans',sans-serif;">
            Bem-vindo ao Coverflex AI
        </h1>
        <p style="font-size:15px;color:#6B6560;margin-bottom:8px;
                  max-width:500px;line-height:1.7;">
            O teu assistente interno para benefícios, compensação e cultura Coverflex.
            Pergunta qualquer coisa — estou aqui para ajudar.
        </p>
        <p style="font-size:12px;color:#B5B0A8;margin-bottom:40px;">
            Powered by Groq · Llama 3.3 70B
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Suggestion cards in 2x2 grid
    suggestions = [
        ("🍽️", "Cartão de refeição", "Qual é o limite diário do cartão de refeição Coverflex?"),
        ("🎁", "Benefícios flexíveis", "Como funcionam os benefícios flexíveis da Coverflex?"),
        ("🏥", "Seguro de saúde", "Como funciona o seguro de saúde na Coverflex?"),
        ("🏠", "Remote Work", "O que é o Remote Work Budget da Coverflex?"),
        ("📈", "Stock Options", "Como funcionam as stock options na Coverflex?"),
        ("👋", "Novo colaborador", "Quais são os primeiros passos como novo colaborador Coverflex?"),
    ]

    col1, col2 = st.columns(2)
    for i, (icon, label, question) in enumerate(suggestions):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div class="welcome-card">
                <span style="font-size:24px;">{icon}</span>
                <div>
                    <div style="font-size:13px;font-weight:600;color:#2E2B28;">{label}</div>
                    <div style="font-size:11px;color:#B5B0A8;margin-top:2px;">
                        Clica para perguntar →
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("→", key=f"welcome_btn_{label}", use_container_width=False):
                st.session_state.quick_question = question
                st.session_state.show_welcome = False
                if not st.session_state.messages:
                    st.session_state.current_conversation = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.rerun()

# CHAT VIEW
else:
    st.markdown("""
    <div style="max-width:760px;margin:0 auto;padding:24px 24px 100px;">
    """, unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.markdown('</div>', unsafe_allow_html=True)

# CHAT INPUT — always visible
if prompt := st.chat_input("Pergunta sobre a Coverflex..."):
    st.session_state.show_welcome = False
    if not st.session_state.messages:
        st.session_state.current_conversation = datetime.now().strftime("%Y%m%d_%H%M%S")

    st.session_state.messages.append({"role": "user", "content": prompt})
    save_conversation(st.session_state.current_conversation, st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("A pensar..."):
            docs = st.session_state.vectorstore.similarity_search(prompt, k=3) \
                if st.session_state.vectorstore else []
            resposta = generate_response(prompt, st.session_state.messages[:-1], docs)
        st.markdown(resposta)
        st.session_state.messages.append({"role": "assistant", "content": resposta})
        save_conversation(st.session_state.current_conversation, st.session_state.messages)

# ============================================
# QUICK QUESTION HANDLER
# ============================================
if st.session_state.quick_question:
    quick_q = st.session_state.quick_question
    st.session_state.quick_question = None
    st.session_state.show_welcome = False

    st.session_state.messages.append({"role": "user", "content": quick_q})
    save_conversation(st.session_state.current_conversation, st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(quick_q)

    with st.chat_message("assistant"):
        with st.spinner("A pensar..."):
            docs = st.session_state.vectorstore.similarity_search(quick_q, k=3) \
                if st.session_state.vectorstore else []
            resposta = generate_response(quick_q, st.session_state.messages[:-1], docs)
        st.markdown(resposta)
        st.session_state.messages.append({"role": "assistant", "content": resposta})
        save_conversation(st.session_state.current_conversation, st.session_state.messages)

    st.rerun()
