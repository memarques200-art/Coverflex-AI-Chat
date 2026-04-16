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
# TRANSLATIONS
# ============================================
TRANSLATIONS = {
    "PT": {
        "title": "Coverflex AI",
        "subtitle": "O teu assistente para benefícios e compensação flexível",
        "welcome_title": "Bem-vindo ao Coverflex AI",
        "welcome_sub": "O teu assistente inteligente para benefícios, onboarding, cultura e muito mais.",
        "new_chat": "＋  Nova Conversa",
        "kb_section": "Base de Conhecimento",
        "kb_active": "Base de conhecimento ativa",
        "kb_inactive": "Sem documentos carregados",
        "topics_section": "Tópicos Rápidos",
        "recent_section": "Conversas Recentes",
        "no_convs": "Sem conversas ainda",
        "train_section": "Treinar com documentos",
        "train_btn": "🚀 Treinar",
        "train_success": "fragmentos carregados!",
        "chat_input": "Pergunta sobre a Coverflex...",
        "thinking": "A pensar...",
        "contacts_section": "Contactos",
        "ask_btn": "Perguntar →",
        "stats_questions": "Perguntas hoje",
        "stats_convs": "Conversas",
        "stats_docs": "Documentos",
        "topics": [
            ("💳", "Cartão Coverflex", "Como ativo o cartão Coverflex?"),
            ("🏖️", "Férias", "Quantos dias de férias tenho na Coverflex?"),
            ("🏠", "Remote Work", "O que é o Remote Work Budget?"),
            ("📚", "Crescimento Pessoal", "O que é o Personal Growth Budget?"),
            ("👶", "Licença Parental", "Qual é a licença parental na Coverflex?"),
            ("❤️", "Caring Days", "O que são os Caring Days?"),
            ("📈", "Stock Options", "Como funcionam as stock options?"),
            ("🏥", "Seguro de Saúde", "Como funciona o seguro de saúde?"),
            ("💰", "Salários", "Quais são as faixas salariais na Coverflex?"),
            ("🎯", "Recrutamento", "Como é o processo de recrutamento?"),
            ("📧", "Contactos", "Quais os contactos do suporte?"),
        ],
        "suggestions": [
            ("🍽️", "Cartão refeição", "Qual é o limite do cartão de refeição Coverflex?"),
            ("🎁", "Benefícios flexíveis", "Como funcionam os benefícios flexíveis da Coverflex?"),
            ("🏥", "Seguro de saúde", "Como funciona o seguro de saúde na Coverflex?"),
            ("🏠", "Remote Work", "O que inclui o Remote Work Budget?"),
            ("📈", "Stock Options", "Como funcionam as stock options na Coverflex?"),
            ("👋", "Onboarding", "Como é o processo de onboarding na Coverflex?"),
        ],
        "system_prompt": """És o assistente de conhecimento interno da Coverflex — uma fintech portuguesa líder em gestão de benefícios flexíveis, fundada em 2021, presente em Portugal, Espanha e Itália.

Respondes SEMPRE em Português de Portugal, de forma clara, direta e útil. O teu tom é profissional mas caloroso e acessível — como um colega sénior bem informado que genuinamente quer ajudar.

Regras:
1. Responde com base na informação disponível na base de conhecimento
2. Se não tiveres informação suficiente, indica qual equipa contactar (RH: rh@coverflex.com, Suporte: help@coverflex.com, IT: it-support@coverflex.com)
3. Sê específico e concreto — evita generalidades
4. Usa formatação clara (listas, negrito) quando ajuda a compreensão
5. Mantém os valores da Coverflex: Extreme Employee Obsession, transparência, simplicidade"""
    },
    "EN": {
        "title": "Coverflex AI",
        "subtitle": "Your assistant for flexible benefits and compensation",
        "welcome_title": "Welcome to Coverflex AI",
        "welcome_sub": "Your intelligent assistant for benefits, onboarding, culture and much more.",
        "new_chat": "＋  New Chat",
        "kb_section": "Knowledge Base",
        "kb_active": "Knowledge base active",
        "kb_inactive": "No documents loaded",
        "topics_section": "Quick Topics",
        "recent_section": "Recent Chats",
        "no_convs": "No conversations yet",
        "train_section": "Train with documents",
        "train_btn": "🚀 Train",
        "train_success": "chunks loaded!",
        "chat_input": "Ask about Coverflex...",
        "thinking": "Thinking...",
        "contacts_section": "Contacts",
        "ask_btn": "Ask →",
        "stats_questions": "Questions today",
        "stats_convs": "Conversations",
        "stats_docs": "Documents",
        "topics": [
            ("💳", "Coverflex Card", "How do I activate the Coverflex card?"),
            ("🏖️", "Holiday", "How many days off do I get at Coverflex?"),
            ("🏠", "Remote Work", "What is the Remote Work Budget?"),
            ("📚", "Personal Growth", "What is the Personal Growth Budget?"),
            ("👶", "Parental Leave", "What is the parental leave policy?"),
            ("❤️", "Caring Days", "What are Caring Days?"),
            ("📈", "Stock Options", "How do stock options work?"),
            ("🏥", "Health Insurance", "How does health insurance work?"),
            ("💰", "Salaries", "What are the salary ranges at Coverflex?"),
            ("🎯", "Recruitment", "What is the recruitment process?"),
            ("📧", "Contacts", "What are the support contacts?"),
        ],
        "suggestions": [
            ("🍽️", "Meal card", "What is the daily limit of the Coverflex meal card?"),
            ("🎁", "Flexible benefits", "How do flexible benefits work at Coverflex?"),
            ("🏥", "Health insurance", "How does health insurance work at Coverflex?"),
            ("🏠", "Remote Work", "What does the Remote Work Budget include?"),
            ("📈", "Stock Options", "How do stock options work at Coverflex?"),
            ("👋", "Onboarding", "What is the onboarding process at Coverflex?"),
        ],
        "system_prompt": """You are the internal knowledge assistant for Coverflex — a leading Portuguese fintech for flexible benefits management, founded in 2021, operating in Portugal, Spain and Italy.

You ALWAYS respond in English, clearly, directly and helpfully. Your tone is professional but warm and accessible — like a knowledgeable senior colleague who genuinely wants to help.

Rules:
1. Answer based on the available knowledge base information
2. If you don't have enough information, indicate which team to contact (HR: rh@coverflex.com, Support: help@coverflex.com, IT: it-support@coverflex.com)
3. Be specific and concrete — avoid generalities
4. Use clear formatting (lists, bold) when it aids understanding
5. Maintain Coverflex values: Extreme Employee Obsession, transparency, simplicity"""
    }
}

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Coverflex AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS — COVERFLEX BRAND
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

    #MainMenu, footer, header { visibility: hidden; }

    :root {
        --cf-peach: #F4886C;
        --cf-peach-light: #FDF1ED;
        --cf-peach-mid: #F7B49E;
        --cf-peach-dark: #C8563A;
        --cf-peach-bg: #FEF7F4;
        --cf-navy: #1A1A2E;
        --cf-white: #FFFFFF;
        --cf-gray-50: #FAFAF8;
        --cf-gray-100: #F5F3F0;
        --cf-gray-200: #EAE7E2;
        --cf-gray-300: #D5D0C8;
        --cf-gray-400: #B5B0A8;
        --cf-gray-500: #8A8580;
        --cf-gray-600: #6B6560;
        --cf-gray-700: #4A4540;
        --cf-gray-800: #2E2B28;
        --cf-green: #2D9E6B;
        --cf-green-light: #E8F7F0;
        --cf-amber: #E8A020;
        --cf-amber-light: #FEF5E4;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 24px;
        --shadow-sm: 0 1px 3px rgba(26,26,46,0.06);
        --shadow-md: 0 4px 16px rgba(26,26,46,0.08);
        --shadow-lg: 0 8px 32px rgba(244,136,108,0.15);
    }

    * { font-family: 'DM Sans', -apple-system, sans-serif !important; }

    .stApp { background: var(--cf-gray-50) !important; }

    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: var(--cf-white) !important;
        border-right: 1px solid var(--cf-gray-200) !important;
    }
    [data-testid="stSidebar"] > div { padding: 0 !important; }

    /* ── BUTTONS ── */
    .stButton > button {
        background: transparent !important;
        border: 1px solid var(--cf-gray-200) !important;
        border-radius: var(--radius-md) !important;
        color: var(--cf-gray-700) !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        padding: 8px 12px !important;
        transition: all 0.15s ease !important;
        text-align: left !important;
        width: 100% !important;
        box-shadow: none !important;
    }
    .stButton > button:hover {
        background: var(--cf-peach-light) !important;
        border-color: var(--cf-peach-mid) !important;
        color: var(--cf-peach-dark) !important;
        transform: translateX(2px) !important;
    }
    .stButton > button:focus {
        box-shadow: 0 0 0 2px rgba(244,136,108,0.2) !important;
    }

    /* ── CHAT INPUT ── */
    .stChatInput {
        background: var(--cf-white) !important;
        border-top: 1px solid var(--cf-gray-200) !important;
        padding: 16px 32px !important;
    }
    .stChatInput textarea {
        border-radius: 40px !important;
        border: 1.5px solid var(--cf-gray-200) !important;
        background: var(--cf-gray-50) !important;
        font-size: 14px !important;
        padding: 12px 20px !important;
        transition: all 0.2s !important;
        resize: none !important;
    }
    .stChatInput textarea:focus {
        border-color: var(--cf-peach) !important;
        background: var(--cf-white) !important;
        box-shadow: 0 0 0 3px rgba(244,136,108,0.12) !important;
    }

    /* ── CHAT MESSAGES ── */
    .stChatMessage {
        background: transparent !important;
        padding: 6px 0 !important;
    }
    [data-testid="stChatMessageContent"] {
        font-size: 14px !important;
        line-height: 1.75 !important;
        color: var(--cf-gray-800) !important;
    }

    /* User message bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        background: var(--cf-peach-light) !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 12px 16px !important;
        border: 1px solid var(--cf-peach-mid) !important;
    }

    /* Assistant message bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
        background: var(--cf-white) !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 12px 16px !important;
        border: 1px solid var(--cf-gray-200) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    /* ── EXPANDER ── */
    .streamlit-expanderHeader {
        font-size: 13px !important;
        color: var(--cf-gray-600) !important;
        background: transparent !important;
        border-radius: var(--radius-sm) !important;
    }
    .streamlit-expanderContent {
        border: none !important;
        padding: 8px 0 0 !important;
    }

    /* ── SUCCESS / WARNING / ERROR ── */
    .stSuccess {
        background: var(--cf-green-light) !important;
        border: 1px solid #A8DFC5 !important;
        border-radius: var(--radius-md) !important;
        font-size: 12px !important;
        color: var(--cf-green) !important;
    }
    .stWarning {
        background: var(--cf-amber-light) !important;
        border: 1px solid #F5D28A !important;
        border-radius: var(--radius-md) !important;
        font-size: 12px !important;
        color: var(--cf-amber) !important;
    }

    /* ── METRIC CARDS ── */
    [data-testid="stMetric"] {
        background: var(--cf-white) !important;
        border: 1px solid var(--cf-gray-200) !important;
        border-radius: var(--radius-md) !important;
        padding: 12px 16px !important;
        box-shadow: var(--shadow-sm) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 11px !important;
        color: var(--cf-gray-500) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 700 !important;
        color: var(--cf-navy) !important;
    }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--cf-gray-200); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--cf-gray-300); }

    /* ── SPINNER ── */
    .stSpinner > div { border-top-color: var(--cf-peach) !important; }

    /* ── SELECT BOX ── */
    .stSelectbox > div > div {
        border-radius: var(--radius-md) !important;
        border-color: var(--cf-gray-200) !important;
        font-size: 13px !important;
    }

    /* ── FILE UPLOADER ── */
    [data-testid="stFileUploader"] section {
        border: 1.5px dashed var(--cf-gray-300) !important;
        border-radius: var(--radius-md) !important;
        background: var(--cf-gray-50) !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: var(--cf-peach-mid) !important;
        background: var(--cf-peach-light) !important;
    }

    /* ── DIVIDER ── */
    hr {
        border: none !important;
        border-top: 1px solid var(--cf-gray-200) !important;
        margin: 8px 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# STATE INIT
# ============================================
MEMORY_DIR = "chat_histories"
VECTORSTORE_DIR = "coverflex_vectorstore"
if not os.path.exists(MEMORY_DIR):
    os.makedirs(MEMORY_DIR)

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

def create_new_conversation():
    cid = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.current_conversation = cid
    st.session_state.messages = []
    save_conversation(cid, [])

def load_vectorstore():
    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        try:
            emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            return Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=emb)
        except:
            return None
    return None

def count_today_questions():
    today = datetime.now().strftime("%Y%m%d")
    count = 0
    for f in get_conversation_files():
        if f.startswith(today):
            msgs = load_conversation(f.replace('.pkl',''))
            count += sum(1 for m in msgs if m["role"] == "user")
    return count

def generate_response(user_message, conversation_history, relevant_docs, lang="PT"):
    T = TRANSLATIONS[lang]
    context = ""
    if relevant_docs:
        context = "Informação oficial da Coverflex:\n\n" if lang == "PT" else "Official Coverflex information:\n\n"
        for doc in relevant_docs[:4]:
            context += f"{doc.page_content[:700]}\n\n"

    chat_history = ""
    for msg in conversation_history[-8:]:
        role = "Utilizador" if msg["role"] == "user" else "Assistente"
        chat_history += f"{role}: {msg['content']}\n"

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
                {"role": "system", "content": T["system_prompt"]},
                {"role": "user", "content": user_content}
            ],
            max_tokens=1200,
            temperature=0.6
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Erro: {str(e)}"

# ── Init session state ──
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()
if "lang" not in st.session_state:
    st.session_state.lang = "PT"
if "quick_question" not in st.session_state:
    st.session_state.quick_question = None
if "current_conversation" not in st.session_state:
    existing = get_conversation_files()
    if existing:
        latest = existing[0].replace('.pkl', '')
        st.session_state.current_conversation = latest
        st.session_state.messages = load_conversation(latest)
    else:
        st.session_state.current_conversation = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.messages = []

T = TRANSLATIONS[st.session_state.lang]

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:

    # ── Logo + Language Toggle ──
    st.markdown(f"""
    <div style="padding:20px 18px 16px;border-bottom:1px solid #EAE7E2;">
        <div style="display:flex;align-items:center;justify-content:space-between;">
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="width:38px;height:38px;background:linear-gradient(135deg,#F4886C,#E05C3A);
                            border-radius:11px;display:flex;align-items:center;justify-content:center;
                            box-shadow:0 4px 12px rgba(244,136,108,0.35);">
                    <span style="color:white;font-size:19px;font-weight:800;letter-spacing:-1px;">C</span>
                </div>
                <div>
                    <div style="font-size:15px;font-weight:700;color:#1A1A2E;line-height:1.1;">Coverflex</div>
                    <div style="font-size:10px;color:#F4886C;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;">AI Assistant</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Language selector
    st.markdown('<div style="padding:12px 18px 0;">', unsafe_allow_html=True)
    lang_choice = st.selectbox("🌐", ["PT 🇵🇹", "EN 🇬🇧"], index=0 if st.session_state.lang == "PT" else 1, label_visibility="collapsed")
    new_lang = "PT" if lang_choice.startswith("PT") else "EN"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        T = TRANSLATIONS[new_lang]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── New Chat ──
    st.markdown('<div style="padding:8px 18px 12px;">', unsafe_allow_html=True)
    if st.button(T["new_chat"], use_container_width=True):
        create_new_conversation()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr style="margin:0 18px;">', unsafe_allow_html=True)

    # ── Stats row ──
    conversations = get_conversation_files()
    doc_count = len(os.listdir(VECTORSTORE_DIR)) if os.path.exists(VECTORSTORE_DIR) else 0
    today_q = count_today_questions()

    st.markdown('<div style="padding:12px 18px;">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(T["stats_questions"], today_q)
    with c2:
        st.metric(T["stats_convs"], len(conversations))
    with c3:
        kb_icon = "✅" if st.session_state.vectorstore else "⚠️"
        st.metric("KB", kb_icon)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr style="margin:0 18px;">', unsafe_allow_html=True)

    # ── Quick Topics ──
    st.markdown(f"""
    <div style="padding:12px 18px 6px;">
        <div style="font-size:10px;font-weight:600;letter-spacing:0.1em;
                    color:#B5B0A8;text-transform:uppercase;">{T["topics_section"]}</div>
    </div>
    """, unsafe_allow_html=True)

    for icon, label, question in T["topics"]:
        if st.button(f"{icon}  {label}", key=f"topic_{label}_{st.session_state.lang}", use_container_width=True):
            st.session_state.quick_question = question
            st.rerun()

    st.markdown('<hr style="margin:8px 18px;">', unsafe_allow_html=True)

    # ── Recent Chats ──
    if conversations:
        st.markdown(f"""
        <div style="padding:8px 18px 4px;">
            <div style="font-size:10px;font-weight:600;letter-spacing:0.1em;
                        color:#B5B0A8;text-transform:uppercase;">{T["recent_section"]}</div>
        </div>
        """, unsafe_allow_html=True)
        for conv in conversations[:5]:
            cid = conv.replace('.pkl', '')
            try:
                d = datetime.strptime(cid, "%Y%m%d_%H%M%S")
                display = d.strftime("%d/%m · %H:%M")
            except:
                display = cid[:12]
            is_active = cid == st.session_state.current_conversation
            msgs = load_conversation(cid)
            msg_count = len([m for m in msgs if m["role"] == "user"])
            label_text = f"💬  {display} · {msg_count}q"
            if st.button(label_text, key=f"hist_{cid}", use_container_width=True):
                st.session_state.current_conversation = cid
                st.session_state.messages = load_conversation(cid)
                st.rerun()

    st.markdown('<hr style="margin:8px 18px;">', unsafe_allow_html=True)

    # ── Train Documents ──
    with st.expander(f"📁 {T['train_section']}"):
        uploaded_files = st.file_uploader("PDF / TXT", type=["pdf", "txt"], accept_multiple_files=True)
        if uploaded_files and st.button(T["train_btn"], use_container_width=True):
            with st.spinner("⏳"):
                docs = []
                for file in uploaded_files:
                    tmp = f"temp_{file.name}"
                    with open(tmp, "wb") as f:
                        f.write(file.getbuffer())
                    loader = PyPDFLoader(tmp) if file.name.endswith('.pdf') else TextLoader(tmp, encoding="utf-8")
                    docs.extend(loader.load())
                    os.remove(tmp)
                chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80).split_documents(docs)
                emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vs = Chroma.from_documents(chunks, emb, persist_directory=VECTORSTORE_DIR)
                vs.persist()
                st.session_state.vectorstore = vs
                st.success(f"✅ {len(chunks)} {T['train_success']}")
                st.rerun()

    # ── Contacts ──
    st.markdown(f"""
    <div style="padding:12px 18px 4px;">
        <div style="font-size:10px;font-weight:600;letter-spacing:0.1em;
                    color:#B5B0A8;text-transform:uppercase;">{T["contacts_section"]}</div>
    </div>
    <div style="padding:4px 18px 20px;">
        <div style="font-size:12px;color:#6B6560;line-height:2;">
            📧 <a href="mailto:help@coverflex.com" style="color:#F4886C;text-decoration:none;">help@coverflex.com</a><br>
            👥 <a href="mailto:rh@coverflex.com" style="color:#F4886C;text-decoration:none;">rh@coverflex.com</a><br>
            💻 <a href="mailto:it-support@coverflex.com" style="color:#F4886C;text-decoration:none;">it-support@coverflex.com</a><br>
            💬 <span style="color:#B5B0A8;">#suporte no Slack</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# MAIN AREA
# ============================================

if not st.session_state.messages:
    # ── WELCOME SCREEN ──
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;
                padding:60px 24px 32px;text-align:center;">
        <div style="width:80px;height:80px;background:linear-gradient(135deg,#F4886C,#E05C3A);
                    border-radius:24px;display:flex;align-items:center;justify-content:center;
                    margin-bottom:24px;box-shadow:0 12px 32px rgba(244,136,108,0.3);">
            <span style="color:white;font-size:42px;font-weight:800;">C</span>
        </div>
        <h1 style="font-size:2.2rem;font-weight:700;color:#1A1A2E;margin:0 0 10px;
                   letter-spacing:-0.02em;">{T["welcome_title"]}</h1>
        <p style="font-size:16px;color:#8A8580;margin:0 0 48px;max-width:500px;line-height:1.6;">
            {T["welcome_sub"]}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Suggestion cards
    suggestions = T["suggestions"]
    row1 = suggestions[:3]
    row2 = suggestions[3:]

    for row in [row1, row2]:
        cols = st.columns(3)
        for i, (icon, label, question) in enumerate(row):
            with cols[i]:
                st.markdown(f"""
                <div style="background:white;border:1px solid #EAE7E2;border-radius:16px;
                            padding:20px;margin-bottom:12px;cursor:pointer;
                            box-shadow:0 2px 8px rgba(26,26,46,0.04);
                            transition:all 0.2s;">
                    <div style="font-size:28px;margin-bottom:10px;">{icon}</div>
                    <div style="font-size:13px;font-weight:600;color:#2E2B28;margin-bottom:4px;">{label}</div>
                    <div style="font-size:12px;color:#B5B0A8;">{T["ask_btn"]}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(label, key=f"welcome_{label}_{st.session_state.lang}", use_container_width=True):
                    st.session_state.quick_question = question
                    st.rerun()

else:
    # ── CHAT MESSAGES ──
    st.markdown('<div style="max-width:760px;margin:0 auto;padding:28px 24px 8px;">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.markdown(f'<div style="font-size:10px;color:#B5B0A8;margin-top:4px;">{message["timestamp"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── CHAT INPUT ──
st.markdown('<div style="max-width:760px;margin:0 auto;">', unsafe_allow_html=True)
if prompt := st.chat_input(T["chat_input"]):
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
    save_conversation(st.session_state.current_conversation, st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(T["thinking"]):
            docs = st.session_state.vectorstore.similarity_search(prompt, k=4) if st.session_state.vectorstore else []
            resposta = generate_response(prompt, st.session_state.messages[:-1], docs, st.session_state.lang)
        st.markdown(resposta)
        ts = datetime.now().strftime("%H:%M")
        st.markdown(f'<div style="font-size:10px;color:#B5B0A8;margin-top:4px;">{ts}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": resposta, "timestamp": ts})
        save_conversation(st.session_state.current_conversation, st.session_state.messages)
st.markdown('</div>', unsafe_allow_html=True)

# ── QUICK QUESTION HANDLER ──
if st.session_state.quick_question:
    quick_q = st.session_state.quick_question
    st.session_state.quick_question = None
    timestamp = datetime.now().strftime("%H:%M")

    st.session_state.messages.append({"role": "user", "content": quick_q, "timestamp": timestamp})
    save_conversation(st.session_state.current_conversation, st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(quick_q)

    with st.chat_message("assistant"):
        with st.spinner(T["thinking"]):
            docs = st.session_state.vectorstore.similarity_search(quick_q, k=4) if st.session_state.vectorstore else []
            resposta = generate_response(quick_q, st.session_state.messages[:-1], docs, st.session_state.lang)
        st.markdown(resposta)
        ts = datetime.now().strftime("%H:%M")
        st.markdown(f'<div style="font-size:10px;color:#B5B0A8;margin-top:4px;">{ts}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": resposta, "timestamp": ts})
        save_conversation(st.session_state.current_conversation, st.session_state.messages)

    st.rerun()
