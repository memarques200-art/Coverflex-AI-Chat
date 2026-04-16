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
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS — COVERFLEX BRAND REDESIGN
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

    /* ── RESET ── */
    #MainMenu, footer, header { visibility: hidden; }
    * { font-family: 'DM Sans', sans-serif; }

    /* ── COVERFLEX TOKENS ── */
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
        --radius-xl: 28px;
    }

    /* ── APP BACKGROUND ── */
    .stApp {
        background: var(--cf-gray-50);
    }

    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: var(--cf-white) !important;
        border-right: 1px solid var(--cf-gray-200);
        width: 280px !important;
    }

    [data-testid="stSidebar"] > div {
        padding: 0 !important;
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
    }

    .stButton > button:hover {
        background: var(--cf-peach-light) !important;
        border-color: var(--cf-peach-mid) !important;
        color: var(--cf-peach-dark) !important;
    }

    /* New Chat button */
    .stButton > button[kind="primary"] {
        background: var(--cf-peach) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: var(--radius-md) !important;
    }

    /* ── CHAT INPUT ── */
    .stChatInput {
        padding: 16px 24px !important;
        background: var(--cf-white) !important;
        border-top: 1px solid var(--cf-gray-200) !important;
    }

    .stChatInput textarea {
        border-radius: 40px !important;
        border: 1.5px solid var(--cf-gray-200) !important;
        background: var(--cf-gray-50) !important;
        font-size: 14px !important;
        padding: 12px 20px !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .stChatInput textarea:focus {
        border-color: var(--cf-peach) !important;
        box-shadow: 0 0 0 3px rgba(244,136,108,0.15) !important;
    }

    /* ── CHAT MESSAGES ── */
    .stChatMessage {
        background: transparent !important;
        padding: 4px 24px !important;
        max-width: 800px !important;
        margin: 0 auto !important;
    }

    [data-testid="stChatMessageContent"] {
        font-size: 14px !important;
        line-height: 1.7 !important;
        color: var(--cf-gray-800) !important;
    }

    /* ── SPINNER ── */
    .stSpinner > div {
        border-top-color: var(--cf-peach) !important;
    }

    /* ── EXPANDER ── */
    .streamlit-expanderHeader {
        font-size: 13px !important;
        color: var(--cf-gray-600) !important;
        border-radius: var(--radius-sm) !important;
    }

    /* ── FILE UPLOADER ── */
    [data-testid="stFileUploader"] {
        font-size: 13px !important;
    }

    /* ── SUCCESS/WARNING ── */
    .stSuccess, .stWarning {
        font-size: 12px !important;
        border-radius: var(--radius-sm) !important;
    }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--cf-gray-200); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ============================================
# FUNÇÕES PRINCIPAIS
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

Respondes SEMPRE em Português de Portugal, de forma clara, direta e útil. Tom profissional mas acessível.

Regras:
1. Responde com base na informação disponível
2. Se não tiveres informação, indica qual equipa contactar (RH, Suporte, IT)
3. Sê específico e concreto
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
# INICIALIZAÇÃO
# ============================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()

if "current_conversation" not in st.session_state:
    existing = get_conversation_files()
    if existing:
        latest = existing[0].replace('.pkl', '')
        st.session_state.current_conversation = latest
        st.session_state.messages = load_conversation(latest)
    else:
        st.session_state.current_conversation = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.messages = []

if "quick_question" not in st.session_state:
    st.session_state.quick_question = None

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding: 24px 20px 16px; border-bottom: 1px solid #EAE7E2;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:36px;height:36px;background:#F4886C;border-radius:10px;
                        display:flex;align-items:center;justify-content:center;">
                <span style="color:white;font-size:18px;font-weight:700;font-family:'DM Sans';">C</span>
            </div>
            <div>
                <div style="font-size:15px;font-weight:700;color:#1A1A2E;line-height:1.2;">Coverflex</div>
                <div style="font-size:11px;color:#F4886C;font-weight:500;letter-spacing:0.05em;">AI ASSISTANT</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # New Chat button
    st.markdown('<div style="padding: 16px 16px 8px;">', unsafe_allow_html=True)
    if st.button("＋  Nova Conversa", use_container_width=True, type="primary"):
        create_new_conversation()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Knowledge Base section
    st.markdown("""
    <div style="padding: 8px 20px 4px;">
        <div style="font-size:10px;font-weight:600;letter-spacing:0.1em;
                    color:#B5B0A8;text-transform:uppercase;">Knowledge Base</div>
    </div>
    """, unsafe_allow_html=True)

    kb_status = "✅ Ativa" if st.session_state.vectorstore else "⚠️ Sem documentos"
    kb_color = "#2D9E6B" if st.session_state.vectorstore else "#E8A020"
    st.markdown(f"""
    <div style="margin:4px 16px 8px;padding:10px 14px;background:#F5F3F0;
                border-radius:10px;font-size:12px;color:{kb_color};font-weight:500;">
        {kb_status}
    </div>
    """, unsafe_allow_html=True)

    # Quick topics
    st.markdown("""
    <div style="padding: 12px 20px 4px;">
        <div style="font-size:10px;font-weight:600;letter-spacing:0.1em;
                    color:#B5B0A8;text-transform:uppercase;">Tópicos Rápidos</div>
    </div>
    """, unsafe_allow_html=True)

    topics = [
        ("💳", "Cartão Coverflex", "Como ativo o cartão Coverflex?"),
        ("🏖️", "Férias", "Quantos dias de férias tenho na Coverflex?"),
        ("🏠", "Remote Work Budget", "O que é o Remote Work Budget?"),
        ("📚", "Personal Growth", "O que é o Personal Growth Budget?"),
        ("👶", "Licença Parental", "Qual é a licença parental na Coverflex?"),
        ("❤️", "Caring Days", "O que são os Caring Days?"),
        ("📈", "Stock Options", "Como funcionam as stock options?"),
        ("🏥", "Seguro de Saúde", "Como funciona o seguro de saúde?"),
        ("📧", "Contactos", "Quais os contactos do suporte Coverflex?"),
    ]

    for icon, label, question in topics:
        if st.button(f"{icon}  {label}", key=f"topic_{label}", use_container_width=True):
            st.session_state.quick_question = question
            st.rerun()

    # Recent chats
    conversations = get_conversation_files()
    if conversations:
        st.markdown("""
        <div style="padding: 16px 20px 4px; margin-top: 8px;">
            <div style="font-size:10px;font-weight:600;letter-spacing:0.1em;
                        color:#B5B0A8;text-transform:uppercase;">Conversas Recentes</div>
        </div>
        """, unsafe_allow_html=True)

        for conv in conversations[:6]:
            conv_id = conv.replace('.pkl', '')
            try:
                date_obj = datetime.strptime(conv_id, "%Y%m%d_%H%M%S")
                display_name = date_obj.strftime("%d/%m · %H:%M")
            except:
                display_name = conv_id[:12]

            is_active = conv_id == st.session_state.current_conversation
            bg = "#FDF1ED" if is_active else "transparent"
            color = "#D4613F" if is_active else "#6B6560"

            if st.button(f"💬  {display_name}", key=f"hist_{conv_id}", use_container_width=True):
                st.session_state.current_conversation = conv_id
                st.session_state.messages = load_conversation(conv_id)
                st.rerun()

    # Train section
    st.markdown('<div style="padding: 8px 16px 16px; margin-top: auto;">', unsafe_allow_html=True)
    with st.expander("📁 Treinar com documentos"):
        uploaded_files = st.file_uploader("PDF ou TXT", type=["pdf", "txt"], accept_multiple_files=True)
        if uploaded_files and st.button("🚀 Treinar", use_container_width=True):
            with st.spinner("A processar..."):
                documents = []
                for file in uploaded_files:
                    temp_path = f"temp_{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    loader = PyPDFLoader(temp_path) if file.name.endswith('.pdf') else TextLoader(temp_path, encoding="utf-8")
                    documents.extend(loader.load())
                    os.remove(temp_path)
                chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vs = Chroma.from_documents(chunks, embeddings, persist_directory=VECTORSTORE_DIR)
                vs.persist()
                st.session_state.vectorstore = vs
                st.success(f"✅ {len(chunks)} fragmentos carregados!")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# MAIN CHAT AREA
# ============================================

# Welcome screen if no messages
if not st.session_state.messages:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                min-height:60vh;padding:40px 20px;text-align:center;">
        <div style="width:72px;height:72px;background:#F4886C;border-radius:20px;
                    display:flex;align-items:center;justify-content:center;margin-bottom:20px;
                    box-shadow:0 8px 24px rgba(244,136,108,0.3);">
            <span style="color:white;font-size:36px;font-weight:700;">C</span>
        </div>
        <h1 style="font-size:2rem;font-weight:700;color:#1A1A2E;margin-bottom:8px;
                   font-family:'DM Sans',sans-serif;">
            Bem-vindo ao Coverflex AI
        </h1>
        <p style="font-size:15px;color:#6B6560;margin-bottom:36px;max-width:480px;line-height:1.6;">
            O teu assistente inteligente para benefícios e compensação flexível.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Suggestion cards
    suggestions = [
        ("🍽️", "Limite diário do cartão refeição?", "Qual é o limite diário do cartão de refeição Coverflex?"),
        ("🎁", "Como funcionam os benefícios?", "Como funcionam os benefícios flexíveis da Coverflex?"),
        ("🏥", "Seguro de saúde", "Como funciona o seguro de saúde na Coverflex?"),
        ("🏠", "Remote Work Budget", "O que é o Remote Work Budget da Coverflex?"),
    ]

    cols = st.columns(2)
    for i, (icon, label, question) in enumerate(suggestions):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:white;border:1px solid #EAE7E2;border-radius:14px;
                        padding:16px 18px;margin-bottom:10px;cursor:pointer;
                        transition:all 0.2s;display:flex;align-items:center;gap:12px;">
                <span style="font-size:22px;">{icon}</span>
                <div>
                    <div style="font-size:13px;font-weight:500;color:#2E2B28;">{label}</div>
                    <div style="font-size:11px;color:#B5B0A8;margin-top:2px;">Clica para perguntar →</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Perguntar", key=f"welcome_{label}", use_container_width=True):
                st.session_state.quick_question = question
                st.rerun()

else:
    # Chat messages
    st.markdown('<div style="max-width:800px;margin:0 auto;padding:24px 24px 0;">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input — always at bottom
st.markdown('<div style="max-width:800px;margin:0 auto;">', unsafe_allow_html=True)
if prompt := st.chat_input("Pergunta sobre a Coverflex..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_conversation(st.session_state.current_conversation, st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("A pensar..."):
            docs = st.session_state.vectorstore.similarity_search(prompt, k=3) if st.session_state.vectorstore else []
            resposta = generate_response(prompt, st.session_state.messages[:-1], docs)
        st.markdown(resposta)
        st.session_state.messages.append({"role": "assistant", "content": resposta})
        save_conversation(st.session_state.current_conversation, st.session_state.messages)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# QUICK QUESTION HANDLER
# ============================================
if st.session_state.quick_question:
    quick_q = st.session_state.quick_question
    st.session_state.quick_question = None

    st.session_state.messages.append({"role": "user", "content": quick_q})
    save_conversation(st.session_state.current_conversation, st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(quick_q)

    with st.chat_message("assistant"):
        with st.spinner("A pensar..."):
            docs = st.session_state.vectorstore.similarity_search(quick_q, k=3) if st.session_state.vectorstore else []
            resposta = generate_response(quick_q, st.session_state.messages[:-1], docs)
        st.markdown(resposta)
        st.session_state.messages.append({"role": "assistant", "content": resposta})
        save_conversation(st.session_state.current_conversation, st.session_state.messages)

    st.rerun()