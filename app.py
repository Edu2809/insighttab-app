import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
import warnings
import re

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# 🔑 CONFIGURAÇÃO DA API
# ═══════════════════════════════════════════════════════════════
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    st.error("❌ API Key GEMINI_API_KEY não encontrada!")
    st.error("👉 Render Dashboard > Environment > Adicione: GEMINI_API_KEY = sua_chave_real")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

MODEL_TIMEOUT = 60
MODEL_RETRIES = 2
RETRY_BACKOFF = 2.0
SAMPLE_SIZE = 500
MAX_OUTPUT_TOKENS = 1024

st.set_page_config(
    page_title="InsightTab - Analista Inteligente", 
    page_icon="📊", 
    layout="wide"
)

# ========== ESTADO ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}
if "processing" not in st.session_state:
    st.session_state.processing = False

# ========== CSS MODO ESCURO ==========
st.markdown("""
<style>
:root {
    --app-bg: #0b1116;
    --panel-bg: #1a1f26;
    --card-bg: #0f1720;
    --text-color: #ffffff;
    --muted-color: #9aa6b2;
    --accent: #667eea;
    --font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}
body, .stApp, .main, .block-container {
    background-color: var(--app-bg) !important;
    color: var(--text-color) !important;
    font-family: var(--font-family) !important;
}
* {
    color: var(--text-color) !important;
    font-family: var(--font-family) !important;
}

/* Corrige texto branco sobre branco */
.stMarkdown, .stTextInput, .stTextInput * {
    color: var(--text-color) !important;
}

/* Placeholder visível */
.stTextInput>div>div>input::placeholder {
    color: rgba(255,255,255,0.6) !important;
}

/* Força contraste em outputs */
div[data-testid="stMarkdownContainer"] {
    background-color: transparent !important;
    color: var(--text-color) !important;
}

/* Fonte uniforme */
.uniform-font {
    font-family: var(--font-family) !important;
    font-size: 1.1em;
    line-height: 1.6em;
}

/* Header */
.main-header {
    font-size: 2.8rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 1rem;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Chat */
.chat-message {
    padding: 15px;
    border-radius: 10px;
    margin: 12px 0;
    color: var(--text-color);
    word-wrap: break-word;
}
.user-message {
    background: rgba(66,153,225,0.12);
    border-left: 4px solid rgba(66,153,225,1);
}
.bot-message {
    background: rgba(156,39,176,0.1);
    border-left: 4px solid rgba(156,39,176,1);
}
</style>
""", unsafe_allow_html=True)

# ========== FUNÇÕES ==========
def read_uploaded_file_to_df(uploaded_file):
    if uploaded_file is None:
        raise ValueError("Nenhum arquivo fornecido")
    uploaded_file.seek(0)
    content = uploaded_file.getvalue()
    bio = BytesIO(content)
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(bio)
    else:
        return pd.read_excel(bio, engine="openpyxl")

def build_prompt_with_data(question, dataframes, sample_size=SAMPLE_SIZE):
    if not dataframes:
        return f"Nenhuma planilha carregada.\nPergunta: {question}"

    all_data = ""
    total_rows = 0
    for filename, df in dataframes.items():
        if isinstance(df, dict):
            for sheet_name, sheet_df in df.items():
                preview = sheet_df.head(5).to_string(index=False)
                all_data += f"\n--- Planilha: {filename} | Aba: {sheet_name} ({len(sheet_df)} linhas) ---\n{preview}\n"
                total_rows += len(sheet_df)
        else:
            preview = df.head(5).to_string(index=False)
            all_data += f"\n--- Planilha: {filename} ({len(df)} linhas) ---\n{preview}\n"
            total_rows += len(df)
    
    prompt = f"""Você é um analista de dados especializado.
DADOS CARREGADOS ({total_rows} linhas no total):
{all_data}

PERGUNTA DO USUÁRIO: {question}

INSTRUÇÕES:
1. Analise APENAS os dados fornecidos acima.
2. Responda em português, com clareza e precisão.
3. Se valores forem monetários, use reais (R$).
4. Mantenha uma formatação consistente de fonte e cor.
Responda agora:"""
    return prompt

def _call_model_sync(prompt, max_output_tokens=MAX_OUTPUT_TOKENS):
    resp = model.generate_content(prompt, max_output_tokens=max_output_tokens)
    return getattr(resp, "text", str(resp))

def call_model_with_timeout(prompt, timeout=MODEL_TIMEOUT):
    last_exc = None
    for attempt in range(1, MODEL_RETRIES + 1):
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_call_model_sync, prompt)
            try:
                return future.result(timeout=timeout)
            except TimeoutError as te:
                future.cancel()
                last_exc = te
            except Exception as e:
                last_exc = e
        time.sleep(RETRY_BACKOFF ** (attempt - 1))
    raise last_exc

def format_currency_in_text(text):
    """Converte números monetários em formato R$."""
    text = re.sub(r'(\b\d{1,3}(?:\.\d{3})*,\d{2}\b)', r'R$ \1', text)
    text = re.sub(r'(?<!R\$ )(\b\d{1,3}(?:\.\d{3})*\b)', lambda m: f"R$ {m.group(0)}" if len(m.group(0)) > 3 else m.group(0), text)
    return text

# ========== MODELO ==========
try:
    model = genai.GenerativeModel("gemini-2.5-flash")
except Exception:
    model = None

# ========== HEADER ==========
st.markdown('<h1 class="main-header">InsightTab - Analista Inteligente</h1>', unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("### 📂 Upload de Planilhas")
    uploaded_files = st.file_uploader("Envie arquivos (Excel ou CSV)", type=["xlsx", "xls", "csv"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.dataframes:
                st.session_state.dataframes[file.name] = read_uploaded_file_to_df(file)
        st.success(f"{len(uploaded_files)} arquivo(s) carregado(s)!")
    if st.button("🗑️ Limpar Planilhas"):
        st.session_state.dataframes = {}
        st.session_state.chat_history = []
        st.rerun()

# ========== CHAT ==========
if st.session_state.dataframes:
    for chat in st.session_state.chat_history:
        st.markdown(f'<div class="chat-message user-message"><b>👤 Você:</b><br>{chat["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-message bot-message uniform-font"><b>🤖 InsightTab:</b><br>{chat["answer"]}</div>', unsafe_allow_html=True)

    user_question = st.text_input("💭 Pergunte sobre os dados:", placeholder="Ex: Qual produto vendeu mais em abril?")
    if st.button("📤 Enviar") and user_question:
        prompt = build_prompt_with_data(user_question, st.session_state.dataframes)
        with st.spinner("🤖 Analisando..."):
            try:
                answer = call_model_with_timeout(prompt)
                answer = format_currency_in_text(answer)
            except Exception as e:
                answer = f"❌ Erro: {e}"
        st.session_state.chat_history.append({"question": user_question, "answer": answer})
        st.rerun()

    if st.button("🧹 Limpar Conversa"):
        st.session_state.chat_history = []
        st.rerun()

else:
    st.info("Envie planilhas na barra lateral para começar.")
