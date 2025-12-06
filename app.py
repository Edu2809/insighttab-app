import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import gspread
import json
from io import BytesIO, StringIO
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from google.oauth2.service_account import Credentials
import time
import warnings
warnings.filterwarnings('ignore')
# ═══════════════════════════════════════════════════════════════
# 🔑 CONFIGURAÇÃO DAS APIs
# ═══════════════════════════════════════════════════════════════
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GOOGLE_SHEETS_CREDENTIALS = os.getenv('GOOGLE_SHEETS_CREDENTIALS')
# IDs das planilhas no Google Sheets
SHEET_IDS = {
    "Janeiro 2024": os.getenv('GOOGLE_SHEET_ID_JANEIRO'),
    "Fevereiro 2024": os.getenv('GOOGLE_SHEET_ID_FEVEREIRO'),
    "Março 2024": os.getenv('GOOGLE_SHEET_ID_MARCO'),
    "Abril 2024": os.getenv('GOOGLE_SHEET_ID_ABRIL'),
    "Maio 2024": os.getenv('GOOGLE_SHEET_ID_MAIO'),
    "Junho 2024": os.getenv('GOOGLE_SHEET_ID_JUNHO'),
    "Julho 2024": os.getenv('GOOGLE_SHEET_ID_JULHO'),
    "Agosto 2024": os.getenv('GOOGLE_SHEET_ID_AGOSTO'),
    "Setembro 2024": os.getenv('GOOGLE_SHEET_ID_SETEMBRO'),
    "Outubro 2024": os.getenv('GOOGLE_SHEET_ID_OUTUBRO'),
    "Novembro 2024": os.getenv('GOOGLE_SHEET_ID_NOVEMBRO'),
    "Dezembro 2024": os.getenv('GOOGLE_SHEET_ID_DEZEMBRO'),
}
# Lista de nomes das planilhas do Google Sheets (para identificação)
GOOGLE_SHEETS_NAMES = list(SHEET_IDS.keys())
# Validar API Keys
if not GEMINI_API_KEY:
    st.error("❌ API Key GEMINI_API_KEY não encontrada!")
    st.error("👉 Render Dashboard > Environment > Adicione: GEMINI_API_KEY = sua_chave")
    st.stop()
if not GOOGLE_SHEETS_CREDENTIALS:
    st.warning("⚠️ Google Sheets não configurado. Modo upload manual ativado.")
    GOOGLE_SHEETS_ENABLED = False
else:
    GOOGLE_SHEETS_ENABLED = True
# Configurar Gemini
genai.configure(api_key=GEMINI_API_KEY)
# ═══════════════════════════════════════════════════════════════
# 📊 FUNÇÃO PARA CARREGAR GOOGLE SHEETS
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def carregar_google_sheets():
    """Carrega dados das planilhas do Google Sheets"""
    if not GOOGLE_SHEETS_ENABLED:
        return {}
    try:
        creds_dict = json.loads(GOOGLE_SHEETS_CREDENTIALS)
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(credentials)
        dataframes = {}
  
        for nome, sheet_id in SHEET_IDS.items():
            if sheet_id:
                try:
                    spreadsheet = client.open_by_key(sheet_id)
                    for worksheet in spreadsheet.worksheets():
                        data = worksheet.get_all_values()
                        if len(data) > 1:
                            df = pd.DataFrame(data[1:], columns=data[0])
                            for col in df.columns:
                                try:
                                    df[col] = pd.to_numeric(df[col])
                                except:
                                    pass
                            aba_nome = worksheet.title
                            key = f"{nome} - {aba_nome}" if aba_nome != "Sheet1" else nome
                            dataframes[key] = df
                except Exception as e:
                    st.warning(f"⚠️ Erro ao carregar {nome}: {str(e)}")
                    continue
        return dataframes
    except json.JSONDecodeError:
        st.error("❌ Erro: Credenciais do Google Sheets inválidas!")
        return {}
    except Exception as e:
        st.error(f"❌ Erro ao conectar Google Sheets: {str(e)}")
        return {}
# ═══════════════════════════════════════════════════════════════
# ⚙️ CONFIGURAÇÕES OTIMIZADAS
# ═══════════════════════════════════════════════════════════════
MODEL_TIMEOUT = 180 # Aumentado para 3 minutos
MODEL_RETRIES = 3 # Mais tentativas
RETRY_BACKOFF = 1.5
MAX_OUTPUT_TOKENS = 8192 # Aumentado para o limite máximo suportado pelos modelos Gemini para permitir respostas mais longas e completas
st.set_page_config(
    page_title="InsightTab - Analista Inteligente",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# ========== ESTADO INICIAL ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dataframes" not in st.session_state:
    loading_placeholder = st.empty()
    loading_placeholder.markdown('<div style="text-align: center; color: black; font-size: 24px;">Carregando o site...</div>', unsafe_allow_html=True)
    st.session_state.dataframes = carregar_google_sheets()
    if st.session_state.dataframes:
        st.success(f"✅ {len(st.session_state.dataframes)} planilha(s) carregada(s) do Google Sheets!")
    loading_placeholder.empty()
if "processing" not in st.session_state:
    st.session_state.processing = False
if "uploaded_file_keys" not in st.session_state:
    st.session_state.uploaded_file_keys = []
if "last_question" not in st.session_state:
    st.session_state.last_question = None
# ========== CSS MODO ESCURO FIXO ==========
st.markdown(
    """
    <style>
    /* Forçar modo escuro global */
    :root {
        --app-bg: #0b1116;
        --panel-bg: #1a1f26;
        --card-bg: #0f1720;
        --text-color: #ffffff;
        --muted-color: #9aa6b2;
        --accent: #667eea;
    }
    /* Background principal - FORÇAR ESCURO EM TUDO */
    .stApp, body, .main, .block-container {
        background-color: var(--app-bg) !important;
        color: var(--text-color) !important;
    }
    /* Forçar fundo escuro na área do conteúdo */
    .main .block-container {
        background-color: var(--app-bg) !important;
    }
    /* Remover qualquer fundo branco */
    .element-container, .stMarkdown, div[data-testid="stVerticalBlock"] {
        background-color: transparent !important;
    }
    /* Sidebar - melhorar contraste */
    [data-testid="stSidebar"] {
        background-color: var(--panel-bg) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--text-color) !important;
    }
    /* Títulos da sidebar mais visíveis */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--text-color) !important;
        font-weight: 600 !important;
    }
    /* Labels da sidebar */
    [data-testid="stSidebar"] label {
        color: var(--text-color) !important;
        font-weight: 500 !important;
    }
    /* Texto da sidebar */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div {
        color: var(--text-color) !important;
    }
    /* BOTÃO CUSTOMIZADO PARA ABRIR SIDEBAR */
    .sidebar-toggle-btn {
        position: fixed;
        top: 20px;
        left: 20px;
        z-index: 999999;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border: none;
        border-radius: 10px;
        width: 50px;
        height: 50px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    .sidebar-toggle-btn:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    .sidebar-toggle-btn svg {
        fill: white;
        width: 24px;
        height: 24px;
    }
    /* Esconder botão customizado quando sidebar está aberta */
    [data-testid="stSidebar"][aria-expanded="true"] ~ div .sidebar-toggle-btn {
        display: none !important;
    }
    /* Mostrar botão customizado quando sidebar está fechada */
    [data-testid="stSidebar"][aria-expanded="false"] ~ div .sidebar-toggle-btn {
        display: flex !important;
    }
    /* Botão de abrir/fechar sidebar - EXTERNO */
    button[kind="header"] {
        color: white !important;
    }
    button[kind="header"] svg {
        fill: white !important;
        stroke: white !important;
    }
    /* Ícone do botão hamburguer - EXTERNO (quando sidebar está fechada) */
   [data-testid="collapsedControl"] {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 10px !important;
    width: 50px !important;
    height: 50px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="collapsedControl"] svg {
    fill: white !important;
    width: 24px !important;
    height: 24px !important;
}
    [data-testid="collapsedControl"]:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 6px 20px rgba(255, 111, 97, 0.8) !important;
    }
    [data-testid="collapsedControl"] > div {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="collapsedControl"] svg {
        fill: white !important;
        width: 24px !important;
        height: 24px !important;
    }
    /* Botão do menu superior (fora da sidebar) */
    header[data-testid="stHeader"] button {
        color: white !important;
    }
    header[data-testid="stHeader"] button svg {
        fill: white !important;
        stroke: white !important;
    }
    /* Forçar todos os botões do header */
    [data-testid="stHeader"] button[kind="header"], [data-testid="stHeader"] button {
        color: white !important;
    }
    [data-testid="stHeader"] button[kind="header"] svg, [data-testid="stHeader"] button svg {
        fill: white !important;
        stroke: white !important;
        color: white !important;
    }
    /* Header principal com ícone de estrela */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        color: var(--text-color);
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header::before {
        content: "";
        display: inline-block;
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 8px;
        margin-right: 15px;
        vertical-align: middle;
        position: relative;
        top: -3px;
    }
    /* Stats boxes */
    .stat-box {
        background: linear-gradient(135deg, #1a1f26, #2d3748);
        padding: 18px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 10px 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    /* Painéis */
    .panel {
        background: var(--panel-bg);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.05);
    }
    /* Mensagens do chat */
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 12px 0;
        color: var(--text-color) !important;
        max-width: 100%;
        word-wrap: break-word;
        position: relative;
    }
    .user-message {
        background: rgba(66,153,225,0.12);
        border-left: 4px solid rgba(66,153,225,1);
    }
    .bot-message {
        background: rgba(156,39,176,0.1);
        border-left: 4px solid rgba(156,39,176,1);
    }
    .chat-message, .chat-message * {
        color: var(--text-color) !important;
    }
    /* ÍCONE DE PLANILHA NO BOT */
    .bot-icon {
        display: inline-block;
        width: 28px;
        height: 28px;
        margin-right: 8px;
        vertical-align: middle;
        position: relative;
        top: -2px;
    }
    /* Cor do elemento ID/Nome (código inline) */
    .chat-message code {
        color: var(--accent) !important;
        background-color: rgba(102, 126, 234, 0.15) !important;
        padding: 2px 4px;
        border-radius: 4px;
        font-family: inherit !important;
        font-size: 0.9em;
    }
    /* MENSAGEM DE PROCESSAMENTO */
    .processing-message {
        padding: 15px;
        border-radius: 10px;
        margin: 12px 0;
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid rgba(102, 126, 234, 1);
        color: var(--text-color) !important;
        animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    /* Inputs e formulários */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 8px;
        caret-color: white !important;
    }
    .stTextInput>label, .stTextArea>label {
        color: var(--text-color) !important;
        font-weight: 500;
    }
    /* Placeholder branco */
    .stTextInput>div>div>input::placeholder, .stTextArea>div>div>textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
        opacity: 1 !important;
    }
    /* Foco nos inputs */
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }
    /* Botões PADRÃO (roxo com gradiente) */
    .stButton>button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    /* BOTÃO DE ENVIAR (azul escuro com texto branco) - FORÇADO */
    div[data-testid="stForm"] button[kind="formSubmit"],
    button[kind="formSubmit"],
    .stFormSubmitButton button,
    .stFormSubmitButton > button,
    div[data-testid="stForm"] > div > div > button {
        background: #1a3a5c !important;
        background-color: #1a3a5c !important;
        background-image: none !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        transition: all 0.3s !important;
        display: inline-block !important;
        min-width: 120px !important;
    }
    div[data-testid="stForm"] button[kind="formSubmit"]:hover,
    button[kind="formSubmit"]:hover,
    .stFormSubmitButton button:hover,
    .stFormSubmitButton > button:hover,
    div[data-testid="stForm"] > div > div > button:hover {
        background: #2d5a8a !important;
        background-color: #2d5a8a !important;
        background-image: none !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(26, 58, 92, 0.6) !important;
    }
    /* Forçar cor do texto do botão e todos os elementos internos */
    div[data-testid="stForm"] button[kind="formSubmit"] *,
    button[kind="formSubmit"] *,
    .stFormSubmitButton button *,
    .stFormSubmitButton button p,
    div[data-testid="stForm"] button p {
        color: white !important;
    }
    /* File uploader - MODO ESCURO */
    [data-testid="stFileUploader"] {
        background-color: var(--card-bg) !important;
        border: 2px dashed rgba(102, 126, 234, 0.5) !important;
        border-radius: 10px !important;
        padding: 25px !important;
    }
    [data-testid="stFileUploader"] * {
        color: var(--text-color) !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: var(--card-bg) !important;
        border: none !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: rgba(102, 126, 234, 0.2) !important;
        color: var(--text-color) !important;
        border: 1px solid rgba(102, 126, 234, 0.4) !important;
    }
    [data-testid="stFileUploader"] small {
        color: var(--muted-color) !important;
    }
    /* Tradução do texto do file uploader */
    [data-testid="stFileUploader"] span[data-testid="stMarkdownContainer"] p {
        color: var(--text-color) !important;
    }
    /* DataFrames */
    .stDataFrame {
        background-color: var(--panel-bg) !important;
    }
    .stDataFrame * {
        color: var(--text-color) !important;
    }
    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--accent) !important;
    }
    /* Esconder footer padrão do Streamlit */
    footer {
        visibility: hidden;
    }
    footer:after {
        content: '';
        visibility: hidden;
    }
    /* Remover "Made with Streamlit" */
    .viewerBadge_container__1QSob {
        display: none !important;
    }
    /* Markdown */
    .stMarkdown {
        color: var(--text-color) !important;
    }
    /* Selectbox */
    .stSelectbox>div>div>div {
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
    }
    /* Success/Error/Warning messages */
    .stAlert {
        background-color: var(--panel-bg) !important;
        color: var(--text-color) !important;
        border-radius: 8px;
    }
    /* ESCONDER TABS (remover visualizar dados) */
    .stTabs {
        display: none !important;
    }
    /* FORÇAR REMOÇÃO DE QUALQUER FUNDO BRANCO */
    section[data-testid="stAppViewContainer"] {
        background-color: var(--app-bg) !important;
    }
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    /* Container principal */
    .main {
        background-color: var(--app-bg) !important;
    }
    /* Todos os elementos precisam ser escuros */
    * {
        scrollbar-color: var(--muted-color) var(--app-bg);
    }
    /* Scrollbar customizada */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: var(--app-bg);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--muted-color);
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent);
    }
    /* Forçar ícones ou textos com a classe .texto-branco a ficarem brancos */
    .texto-branco,
    .texto-branco svg,
    .texto-branco path {
        color: white !important;
        fill: white !important;
        stroke: white !important;
    }
    /* Forçar cor branca em todos os ícones padrão do Streamlit keyboard_double_arrow_right */
    span[data-testid="stIconMaterial"] {
        color: white !important;
        fill: white !important;
        stroke: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Ícone de planilha com fundo roxo-azulado (quadrado)
TABLE_ICON_SVG = """
<svg class="bot-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <!-- Fundo roxo-azulado com bordas arredondadas -->
    <rect x="2" y="2" width="20" height="20" rx="3" fill="url(#gradient)" />
    <defs>
        <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
        </linearGradient>
    </defs>
    <!-- Ícone de planilha branco -->
    <path d="M7 6 L7 18 L14 18 L17 15 L17 6 Z" fill="white" stroke="white" stroke-width="0.3"/>
    <path d="M14 6 L14 15 L17 15" fill="none" stroke="white" stroke-width="0.3"/>
    <!-- Grade da planilha -->
    <line x1="8" y1="9" x2="13" y2="9" stroke="#667eea" stroke-width="0.4"/>
    <line x1="8" y1="11" x2="13" y2="11" stroke="#667eea" stroke-width="0.4"/>
    <line x1="8" y1="13" x2="13" y2="13" stroke="#667eea" stroke-width="0.4"/>
    <line x1="8" y1="15" x2="13" y2="15" stroke="#667eea" stroke-width="0.4"/>
    <line x1="10" y1="7.5" x2="10" y2="16" stroke="#667eea" stroke-width="0.4"/>
    <line x1="11.5" y1="7.5" x2="11.5" y2="16" stroke="#667eea" stroke-width="0.4"/>
</svg>
"""
# ========== CONFIGURAR GEMINI COM PARÂMETROS OTIMIZADOS ==========
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  },
]
try:
    generation_config = {
        "temperature": 0.4, # Reduzido para respostas mais focadas
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
except Exception:
    try:
        model = genai.GenerativeModel("gemini-1.5-pro", safety_settings=safety_settings)
    except:
        model = genai.GenerativeModel("gemini-1.5-flash", safety_settings=safety_settings)
# ========== FUNÇÕES AUXILIARES ==========
def read_uploaded_file_to_df(uploaded_file):
    """Lê arquivo Excel ou CSV e retorna DataFrame"""
    if uploaded_file is None:
        raise ValueError("Nenhum arquivo fornecido")
    try:
        uploaded_file.seek(0)
        content = uploaded_file.getvalue()
        bio = BytesIO(content)
        name = uploaded_file.name.lower()
  
        if name.endswith(".csv"):
            df = pd.read_csv(bio)
            bio.close()
            return df
  
        try:
            df = pd.read_excel(bio, engine="openpyxl")
            bio.close()
            return df
        except Exception:
            bio.seek(0)
            xl = pd.ExcelFile(bio, engine="openpyxl")
            sheets = xl.sheet_names
            if len(sheets) > 1:
                dfs = {}
                for sheet in sheets:
                    bio.seek(0)
                    dfs[f"{uploaded_file.name} - {sheet}"] = pd.read_excel(bio, sheet_name=sheet, engine="openpyxl")
                xl.close()
                bio.close()
                return dfs
            else:
                bio.seek(0)
                df = pd.read_excel(bio, engine="openpyxl")
                xl.close()
                bio.close()
                return df
    except Exception as e:
        raise Exception(f"Erro ao ler arquivo: {str(e)}")
def is_google_sheets_data(filename):
    """Verifica se uma planilha veio do Google Sheets"""
    for sheet_name in GOOGLE_SHEETS_NAMES:
        if filename.startswith(sheet_name):
            return True
    return False
def build_prompt_for_code_gen(question, dataframes):
    """Constrói prompt para gerar código Python que responde à pergunta usando os dataframes completos"""
    if not dataframes:
        return f"Nenhuma planilha foi carregada ainda.\n\nPergunta: {question}"
    
    descriptions = ""
    for filename, df in dataframes.items():
        if isinstance(df, dict):
            for sheet_name, sheet_df in df.items():
                cols = sheet_df.columns.tolist()
                dtypes = sheet_df.dtypes.to_dict()
                descriptions += f"\n--- DataFrame: '{sheet_name}' ---\n"
                descriptions += f"Colunas: {cols}\n"
                descriptions += f"Tipos: {dtypes}\n"
                descriptions += f"Número de linhas: {len(sheet_df)}\n"
        else:
            cols = df.columns.tolist()
            dtypes = df.dtypes.to_dict()
            descriptions += f"\n--- DataFrame: '{filename}' ---\n"
            descriptions += f"Colunas: {cols}\n"
            descriptions += f"Tipos: {dtypes}\n"
            descriptions += f"Número de linhas: {len(df)}\n"
    
    prompt = f"""Você é um analista de dados especializado em pandas.
Os dataframes estão disponíveis em um dicionário chamado 'dfs', onde as chaves são os nomes das planilhas (ex: 'Janeiro 2024', 'Julho 2024 - Aba1', etc.).
Descrições dos dataframes:
{descriptions}

PERGUNTA DO USUÁRIO: {question}

INSTRUÇÕES:
- Escreva APENAS o código Python usando pandas para calcular a resposta exata usando os dataframes completos em 'dfs'.
- NÃO carregue dados, assuma que 'dfs' já existe.
- Use import pandas as pd se necessário, mas como já está importado, não precisa.
- Para perguntas sobre meses específicos, use o dataframe correspondente (ex: dfs['Julho 2024']).
- Calcule agregações exatas como somas, contagens, máximos, etc.
- Formate a saída final em português brasileiro, com formatação monetária R$ X.XXX,XX se aplicável.
- Imprima SOMENTE a resposta final usando print(), sem explicações adicionais.
- Se os dados não forem suficientes para responder, print('Dados insuficientes para responder.').
- Coloque o código dentro de ```python ... ```

Exemplo de código:
```python
df = dfs['Julho 2024']
vendas = df.groupby('Produto')['Quantidade'].sum()
max_produto = vendas.idxmax()
max_quant = vendas.max()
print(f'O produto que mais vendeu foi `{max_produto}` com `{max_quant}` unidades.')
