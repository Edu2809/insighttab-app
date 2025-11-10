import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import gspread
import json
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from google.oauth2.service_account import Credentials
import time
import warnings
import hashlib
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# 🔑 CONFIGURAÇÃO DAS APIs COM ROTAÇÃO
# ═══════════════════════════════════════════════════════════════
GOOGLE_SHEETS_CREDENTIALS = os.getenv('GOOGLE_SHEETS_CREDENTIALS')

# Sistema de múltiplas chaves API (Rotação automática)
GEMINI_API_KEYS = []
for i in range(1, 11):  # Suporta até 10 chaves (GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.)
    key = os.getenv(f'GEMINI_API_KEY_{i}') or os.getenv(f'GEMINI_API_KEY{i}')
    if key:
        GEMINI_API_KEYS.append(key)

# Fallback para chave única
if not GEMINI_API_KEYS:
    single_key = os.getenv('GEMINI_API_KEY')
    if single_key:
        GEMINI_API_KEYS.append(single_key)

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

GOOGLE_SHEETS_NAMES = list(SHEET_IDS.keys())

# Validar API Keys
if not GEMINI_API_KEYS:
    st.error("❌ Nenhuma API Key do Gemini encontrada!")
    st.error("👉 Render Dashboard > Environment > Adicione:")
    st.error("   - GEMINI_API_KEY_1 = primeira_chave")
    st.error("   - GEMINI_API_KEY_2 = segunda_chave (opcional)")
    st.error("   - GEMINI_API_KEY_3 = terceira_chave (opcional)")
    st.stop()

if not GOOGLE_SHEETS_CREDENTIALS:
    st.warning("⚠️ Google Sheets não configurado. Modo upload manual ativado.")
    GOOGLE_SHEETS_ENABLED = False
else:
    GOOGLE_SHEETS_ENABLED = True

# Estado para controle de rotação de chaves
# MOVIDO PARA DEPOIS DE set_page_config E ANTES DE USAR
# (já está corrigido acima no código)

# ═══════════════════════════════════════════════════════════════
# 📊 FUNÇÃO PARA CARREGAR GOOGLE SHEETS
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)  # AUMENTADO PARA 1 HORA
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
# ⚙️ CONFIGURAÇÕES OTIMIZADAS - REDUZIDAS PARA EVITAR LIMITE
# ═══════════════════════════════════════════════════════════════
MODEL_TIMEOUT = 120  # 2 minutos
MODEL_RETRIES = 1  # APENAS 1 TENTATIVA - CRÍTICO!
RETRY_BACKOFF = 2
SAMPLE_SIZE = 1500
MAX_OUTPUT_TOKENS = 2048

st.set_page_config(
    page_title="InsightTab - Analista Inteligente",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== FUNÇÕES PARA GERAR HASH (EVITAR DUPLICAÇÃO) ==========
def generate_question_hash(question):
    """Gera hash único para cada pergunta"""
    return hashlib.md5(question.encode()).hexdigest()

# ========== ESTADO INICIAL - DEVE VIR ANTES DE QUALQUER USO ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "uploaded_file_keys" not in st.session_state:
    st.session_state.uploaded_file_keys = []
if "processed_questions" not in st.session_state:
    st.session_state.processed_questions = set()
if "last_submission_time" not in st.session_state:
    st.session_state.last_submission_time = 0

# INICIALIZAR VARIÁVEIS DE CONTROLE DE API - CRÍTICO!
if "current_api_key_index" not in st.session_state:
    st.session_state.current_api_key_index = 0
if "api_key_failures" not in st.session_state:
    st.session_state.api_key_failures = {i: 0 for i in range(len(GEMINI_API_KEYS))}
if "last_api_call_time" not in st.session_state:
    st.session_state.last_api_call_time = {}

# Carregar dataframes após inicializar session_state
if "dataframes" not in st.session_state:
    loading_placeholder = st.empty()
    loading_placeholder.markdown('<div style="text-align: center; color: white; font-size: 24px;">Carregando o site...</div>', unsafe_allow_html=True)
    st.session_state.dataframes = carregar_google_sheets()
    if st.session_state.dataframes:
        st.success(f"✅ {len(st.session_state.dataframes)} planilha(s) carregada(s) do Google Sheets!")
    loading_placeholder.empty()

def get_next_available_api_key():
    """Retorna a próxima chave API disponível com menos falhas"""
    # Ordenar chaves por número de falhas (menor primeiro)
    sorted_keys = sorted(st.session_state.api_key_failures.items(), key=lambda x: x[1])
    
    # Tentar encontrar uma chave que não falhou recentemente
    for key_index, failures in sorted_keys:
        # Se a chave tem menos de 3 falhas, usar ela
        if failures < 3:
            st.session_state.current_api_key_index = key_index
            return GEMINI_API_KEYS[key_index], key_index
    
    # Se todas falharam muito, resetar contadores e usar a primeira
    st.session_state.api_key_failures = {i: 0 for i in range(len(GEMINI_API_KEYS))}
    st.session_state.current_api_key_index = 0
    return GEMINI_API_KEYS[0], 0

def mark_api_key_failed(key_index):
    """Marca uma chave como falha"""
    st.session_state.api_key_failures[key_index] += 1

def check_rate_limit(key_index):
    """Verifica se passou tempo suficiente desde a última chamada"""
    current_time = time.time()
    last_call = st.session_state.last_api_call_time.get(key_index, 0)
    
    # Exigir 3 segundos entre chamadas da mesma chave
    if current_time - last_call < 3:
        return False
    
    st.session_state.last_api_call_time[key_index] = current_time
    return True

# ========== CSS (MANTIDO IGUAL) ==========
st.markdown(
    """
    <style>
    :root {
        --app-bg: #0b1116;
        --panel-bg: #1a1f26;
        --card-bg: #0f1720;
        --text-color: #ffffff;
        --muted-color: #9aa6b2;
        --accent: #667eea;
    }
    .stApp, body, .main, .block-container {
        background-color: var(--app-bg) !important;
        color: var(--text-color) !important;
    }
    .main .block-container {
        background-color: var(--app-bg) !important;
    }
    .element-container, .stMarkdown, div[data-testid="stVerticalBlock"] {
        background-color: transparent !important;
    }
    [data-testid="stSidebar"] {
        background-color: var(--panel-bg) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--text-color) !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--text-color) !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] label {
        color: var(--text-color) !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div {
        color: var(--text-color) !important;
    }
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
    .stat-box {
        background: linear-gradient(135deg, #1a1f26, #2d3748);
        padding: 18px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 10px 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    .panel {
        background: var(--panel-bg);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.05);
    }
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
    .bot-icon {
        display: inline-block;
        width: 28px;
        height: 28px;
        margin-right: 8px;
        vertical-align: middle;
        position: relative;
        top: -2px;
    }
    .chat-message code {
        color: var(--accent) !important;
        background-color: rgba(102, 126, 234, 0.15) !important;
        padding: 2px 4px;
        border-radius: 4px;
        font-family: inherit !important;
        font-size: 0.9em;
    }
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
    .stTextInput>div>div>input::placeholder, .stTextArea>div>div>textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
        opacity: 1 !important;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }
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
    div[data-testid="stForm"] button[kind="formSubmit"] *,
    button[kind="formSubmit"] *,
    .stFormSubmitButton button *,
    .stFormSubmitButton button p,
    div[data-testid="stForm"] button p {
        color: white !important;
    }
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
    .stDataFrame {
        background-color: var(--panel-bg) !important;
    }
    .stDataFrame * {
        color: var(--text-color) !important;
    }
    .stSpinner > div {
        border-top-color: var(--accent) !important;
    }
    footer {
        visibility: hidden;
    }
    footer:after {
        content: '';
        visibility: hidden;
    }
    .viewerBadge_container__1QSob {
        display: none !important;
    }
    .stMarkdown {
        color: var(--text-color) !important;
    }
    .stSelectbox>div>div>div {
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
    }
    .stAlert {
        background-color: var(--panel-bg) !important;
        color: var(--text-color) !important;
        border-radius: 8px;
    }
    .stTabs {
        display: none !important;
    }
    section[data-testid="stAppViewContainer"] {
        background-color: var(--app-bg) !important;
    }
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    .main {
        background-color: var(--app-bg) !important;
    }
    * {
        scrollbar-color: var(--muted-color) var(--app-bg);
    }
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
    </style>
    """,
    unsafe_allow_html=True,
)

TABLE_ICON_SVG = """
<svg class="bot-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="2" y="2" width="20" height="20" rx="3" fill="url(#gradient)" />
    <defs>
        <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
        </linearGradient>
    </defs>
    <path d="M7 6 L7 18 L14 18 L17 15 L17 6 Z" fill="white" stroke="white" stroke-width="0.3"/>
    <path d="M14 6 L14 15 L17 15" fill="none" stroke="white" stroke-width="0.3"/>
    <line x1="8" y1="9" x2="13" y2="9" stroke="#667eea" stroke-width="0.4"/>
    <line x1="8" y1="11" x2="13" y2="11" stroke="#667eea" stroke-width="0.4"/>
    <line x1="8" y1="13" x2="13" y2="13" stroke="#667eea" stroke-width="0.4"/>
    <line x1="8" y1="15" x2="13" y2="15" stroke="#667eea" stroke-width="0.4"/>
    <line x1="10" y1="7.5" x2="10" y2="16" stroke="#667eea" stroke-width="0.4"/>
    <line x1="11.5" y1="7.5" x2="11.5" y2="16" stroke="#667eea" stroke-width="0.4"/>
</svg>
"""

# ========== CONFIGURAR GEMINI ==========
# Remover configuração única - agora usa rotação dinâmica
# A configuração é feita dentro de _call_model_sync()

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

def build_prompt_with_data(question, dataframes, sample_size=SAMPLE_SIZE):
    """Constrói prompt otimizado com dados das planilhas"""
    if not dataframes:
        return f"Nenhuma planilha foi carregada ainda.\n\nPergunta: {question}"
  
    summary_data = ""
    detailed_data = ""
    total_rows = 0
  
    for filename, df in dataframes.items():
        if isinstance(df, dict):
            for sheet_name, sheet_df in df.items():
                total_rows += len(sheet_df)
                summary_data += f"\n--- Resumo: {sheet_name} ({len(sheet_df)} linhas, {len(sheet_df.columns)} colunas) ---\n"
                summary_data += f"Colunas: {', '.join(sheet_df.columns.tolist())}\n"
                numeric_cols = sheet_df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    summary_data += f"Colunas numéricas: {', '.join(numeric_cols)}\n"
                    for col in numeric_cols[:5]:
                        try:
                            summary_data += f"  {col}: min={sheet_df[col].min()}, max={sheet_df[col].max()}, média={sheet_df[col].mean():.2f}\n"
                        except:
                            pass
                if len(sheet_df) <= sample_size:
                    detailed_data += f"\n--- Dados completos: {sheet_name} ---\n"
                    detailed_data += sheet_df.to_string(index=False, max_rows=sample_size)
                else:
                    detailed_data += f"\n--- Amostra: {sheet_name} (primeiras {sample_size} linhas) ---\n"
                    detailed_data += sheet_df.head(sample_size).to_string(index=False)
                detailed_data += "\n"
        else:
            total_rows += len(df)
            summary_data += f"\n--- Resumo: {filename} ({len(df)} linhas, {len(df.columns)} colunas) ---\n"
            summary_data += f"Colunas: {', '.join(df.columns.tolist())}\n"
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                summary_data += f"Colunas numéricas: {', '.join(numeric_cols)}\n"
                for col in numeric_cols[:5]:
                    try:
                        summary_data += f"  {col}: min={df[col].min()}, max={df[col].max()}, média={df[col].mean():.2f}\n"
                    except:
                        pass
            if len(df) <= sample_size:
                detailed_data += f"\n--- Dados completos: {filename} ---\n"
                detailed_data += df.to_string(index=False, max_rows=sample_size)
            else:
                detailed_data += f"\n--- Amostra: {filename} (primeiras {sample_size} linhas) ---\n"
                detailed_data += df.head(sample_size).to_string(index=False)
            detailed_data += "\n"
  
    prompt = f"""Você é um analista de dados especializado em análise de planilhas.

RESUMO DOS DADOS DISPONÍVEIS (Total: {total_rows} linhas):
{summary_data}

AMOSTRA DOS DADOS:
{detailed_data}

PERGUNTA DO USUÁRIO: {question}

INSTRUÇÕES IMPORTANTES:
1. Você tem acesso a TODAS as {total_rows} linhas de dados através do resumo estatístico acima
2. Use os dados estatísticos (min, max, média) para responder perguntas sobre totais e agregações
3. Se precisar de cálculos específicos, você pode inferir a partir das estatísticas fornecidas
4. Responda em português brasileiro de forma clara, objetiva e profissional
5. Use números EXATOS e formatação monetária brasileira: R$ X.XXX,XX (ex: R$ 42.173,01)
6. NUNCA use negrito, itálico ou formatação de fonte
7. Use APENAS código inline do Markdown (crases) para destacar: `nomes de produtos`, `IDs` e `valores monetários`
8. Se os dados não forem suficientes para responder, informe isso claramente
9. Para perguntas complexas, forneça análise detalhada com base nas estatísticas disponíveis

Responda de forma direta e completa:"""
  
    return prompt

def _call_model_sync(prompt, max_output_tokens=MAX_OUTPUT_TOKENS):
    """Chamada síncrona ao modelo com rotação de chaves API"""
    max_attempts = len(GEMINI_API_KEYS)
    last_error = None
    
    for attempt in range(max_attempts):
        # Obter próxima chave disponível
        api_key, key_index = get_next_available_api_key()
        
        # Verificar rate limit
        if not check_rate_limit(key_index):
            time.sleep(3)  # Aguardar se necessário
        
        try:
            # Reconfigurar Gemini com a nova chave
            genai.configure(api_key=api_key)
            
            # Tentar criar o modelo
            try:
                model = genai.GenerativeModel(
                    "gemini-2.0-flash-exp",
                    generation_config={
                        "temperature": 0.4,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": max_output_tokens,
                    }
                )
            except:
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                except:
                    model = genai.GenerativeModel("gemini-pro")
            
            # Fazer a chamada
            resp = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_output_tokens,
                    "temperature": 0.4,
                }
            )
            
            # Se chegou aqui, sucesso! Resetar falhas desta chave
            st.session_state.api_key_failures[key_index] = 0
            return getattr(resp, "text", str(resp))
            
        except Exception as e:
            error_str = str(e).lower()
            last_error = e
            
            # Se for erro de quota/limite, marcar chave como falha e tentar próxima
            if any(x in error_str for x in ["429", "quota", "resource_exhausted", "rate limit"]):
                mark_api_key_failed(key_index)
                
                # Se não há mais chaves para tentar, retornar erro
                if attempt >= max_attempts - 1:
                    raise Exception("❌ Todas as chaves API atingiram o limite. Aguarde alguns minutos e tente novamente.")
                
                # Tentar próxima chave
                continue
            else:
                # Para outros erros, tentar imediatamente com fallback
                try:
                    resp = model.generate_content(prompt)
                    return getattr(resp, "text", str(resp))
                except:
                    raise e
    
    # Se todas as tentativas falharam
    raise last_error if last_error else Exception("Erro desconhecido")

def call_model_with_timeout(prompt, timeout=MODEL_TIMEOUT):
    """Chama modelo com timeout - APENAS 1 TENTATIVA"""
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_call_model_sync, prompt)
        try:
            result = future.result(timeout=timeout)
            return result
        except TimeoutError as te:
            future.cancel()
            raise TimeoutError("⏱️ A análise está demorando mais que o esperado. Por favor, tente reformular sua pergunta de forma mais específica.")
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "resource_exhausted" in str(e).lower():
                raise Exception("❌ Limite de requisições da API Gemini atingido. Aguarde alguns minutos e tente novamente.")
            raise e

# ========== HEADER ==========
st.markdown('<h1 class="main-header">InsightTab - Analista Inteligente</h1>', unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("### 📂 Gerenciar Dados")
    
    # Mostrar status das chaves API
    st.markdown("---")
    st.markdown("### 🔑 Status das APIs")
    st.success(f"✅ {len(GEMINI_API_KEYS)} chave(s) Gemini configurada(s)")
    
    # Mostrar qual chave está ativa
    active_key_num = st.session_state.current_api_key_index + 1
    st.info(f"🔄 Usando chave #{active_key_num}")
    
    # Mostrar falhas (se houver)
    total_failures = sum(st.session_state.api_key_failures.values())
    if total_failures > 0:
        st.warning(f"⚠️ {total_failures} falha(s) registrada(s)")
        if st.button("🔄 Resetar Contadores", use_container_width=True):
            st.session_state.api_key_failures = {i: 0 for i in range(len(GEMINI_API_KEYS))}
            st.success("✅ Contadores resetados!")
            st.rerun()
    
    st.markdown("---")
  
    if GOOGLE_SHEETS_ENABLED:
        st.success("✅ Google Sheets conectado!")
        if st.button("🔄 Recarregar Google Sheets", use_container_width=True):
            st.cache_data.clear()
            st.session_state.dataframes = carregar_google_sheets()
            st.rerun()
    else:
        st.info("ℹ️ Google Sheets não configurado. Use upload manual.")
  
    st.markdown("---")
    st.markdown("### ➕ Upload Manual (Opcional)")
  
    file_uploader_key = f"file_uploader_{len(st.session_state.uploaded_file_keys)}"
    uploaded_files = st.file_uploader(
        "Adicionar planilhas extras",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        key=file_uploader_key
    )
  
    if uploaded_files:
        new_files = []
        for file in uploaded_files:
            if file.name not in st.session_state.dataframes:
                new_files.append(file)
      
        if new_files:
            with st.spinner(f"📊 Carregando {len(new_files)} arquivo(s)..."):
                for file in new_files:
                    try:
                        df_result = read_uploaded_file_to_df(file)
                        if isinstance(df_result, dict):
                            for sheet_name, sheet_df in df_result.items():
                                st.session_state.dataframes[sheet_name] = sheet_df
                        else:
                            st.session_state.dataframes[file.name] = df_result
                    except Exception as e:
                        st.error(f"❌ {file.name}: {str(e)}")
            st.rerun()
  
    if st.session_state.dataframes:
        st.markdown("---")
        st.markdown("### ✅ Dados Disponíveis")
        total_rows = 0
        for filename, df in st.session_state.dataframes.items():
            rows = len(df)
            if is_google_sheets_data(filename):
                badge = "☁️"
            else:
                badge = "📄"
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"{badge} **{filename}**<br><small>{rows} linhas</small>", unsafe_allow_html=True)
            if not is_google_sheets_data(filename):
                with col2:
                    if st.button("X", key=f"delete_{filename}"):
                        del st.session_state.dataframes[filename]
                        st.session_state.uploaded_file_keys.append(time.time())
                        st.success(f"✅ Planilha {filename} excluída!")
                        st.rerun()
            total_rows += rows
      
        st.markdown(f'<div style="margin-top: 10px; padding: 10px; background: var(--card-bg); border-radius: 8px; text-align: center;"><b>Total: {total_rows:,} linhas</b></div>', unsafe_allow_html=True)
  
    has_manual_sheets = any(
        not is_google_sheets_data(filename) for filename in st.session_state.dataframes.keys()
    )
  
    if has_manual_sheets:
        if st.button("🗑️ Limpar Planilhas Manuais", use_container_width=True):
            google_sheets_data = {
                k: v for k, v in st.session_state.dataframes.items() if is_google_sheets_data(k)
            }
            st.session_state.dataframes = google_sheets_data
            st.session_state.uploaded_file_keys.append(time.time())
            st.success("✅ Planilhas manuais removidas!")
            st.rerun()

# ========== FUNÇÃO PARA EXIBIR CHAT ==========
def render_chat_history():
    """Renderiza o histórico do chat"""
    for chat in st.session_state.chat_history:
        st.markdown(
            f'<div class="chat-message user-message"><b>👤 Você:</b><br>{chat["question"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="chat-message bot-message"><b>{TABLE_ICON_SVG} InsightTab:</b><br>{chat["answer"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")

# ========== ÁREA PRINCIPAL - CHAT ==========
if st.session_state.dataframes:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
  
    render_chat_history()
   
    if st.session_state.processing:
        st.markdown(
            '<div class="processing-message"><b>🤖 Analisando dados... Por favor, aguarde.</b></div>',
            unsafe_allow_html=True
        )
  
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input(
            "💭 Faça sua pergunta:",
            placeholder="Ex: Qual produto vendeu mais? Qual região tem maior receita?",
            key="chat_input",
            label_visibility="collapsed"
        )
        col1, col2 = st.columns([4, 1])
        with col2:
            submit = st.form_submit_button("📤 Enviar", use_container_width=True)
  
    # CRÍTICO: DEBOUNCING E PREVENÇÃO DE DUPLICAÇÃO
    if submit and user_question and not st.session_state.processing:
        current_time = time.time()
        question_hash = generate_question_hash(user_question)
        
        # Verificar debouncing (mínimo 2 segundos entre envios)
        if current_time - st.session_state.last_submission_time < 2:
            st.warning("⚠️ Aguarde um momento antes de enviar outra pergunta.")
        # Verificar se pergunta já foi processada
        elif question_hash in st.session_state.processed_questions:
            st.info("ℹ️ Esta pergunta já foi respondida. Veja o histórico acima.")
        else:
            # Marcar como processando
            st.session_state.processing = True
            st.session_state.last_submission_time = current_time
            st.session_state.processed_questions.add(question_hash)
            
            # Construir prompt
            prompt = build_prompt_with_data(user_question, st.session_state.dataframes)
            
            # Chamar modelo
            try:
                answer = call_model_with_timeout(prompt)
            except TimeoutError as e:
                answer = str(e)
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower() or "resource_exhausted" in error_msg.lower():
                    answer = "❌ Limite de requisições da API Gemini atingido. Aguarde alguns minutos antes de fazer outra pergunta."
                else:
                    answer = f"❌ Erro ao processar: {error_msg[:200]}"
            
            # Adicionar ao histórico
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": answer
            })
            
            # Finalizar processamento
            st.session_state.processing = False
            st.rerun()
  
    if st.button("🧹 Limpar Conversa"):
        st.session_state.chat_history = []
        st.session_state.processed_questions = set()
        st.session_state.processing = False
        st.rerun()
  
    st.markdown('</div>', unsafe_allow_html=True)
  
    st.markdown("---")
    total_files = len(st.session_state.dataframes)
    total_rows = sum(len(df) for df in st.session_state.dataframes.values())
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stat-box"><h2 style="margin:0;">{total_files}</h2><p style="margin:0;">Tabelas</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-box"><h2 style="margin:0;">{total_rows:,}</h2><p style="margin:0;">Linhas</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-box"><h2 style="margin:0;">{len(st.session_state.chat_history)}</h2><p style="margin:0;">Perguntas</p></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
  
    render_chat_history()
   
    if st.session_state.processing:
        st.markdown(
            '<div class="processing-message"><b>🤖 Analisando sua pergunta...</b></div>',
            unsafe_allow_html=True
        )
  
    with st.form(key="nodata_form", clear_on_submit=True):
        user_question = st.text_input(
            "💭 Faça sua pergunta:",
            placeholder="Ex: Como funciona este app? O que posso fazer aqui?",
            key="input_question_no_data",
            label_visibility="collapsed"
        )
        col_btn1, col_btn2 = st.columns([4, 1])
        with col_btn2:
            submit_btn = st.form_submit_button("📤 Enviar", use_container_width=True)
  
    if submit_btn and user_question and not st.session_state.processing:
        current_time = time.time()
        question_hash = generate_question_hash(user_question)
        
        if current_time - st.session_state.last_submission_time < 2:
            st.warning("⚠️ Aguarde um momento antes de enviar outra pergunta.")
        elif question_hash in st.session_state.processed_questions:
            st.info("ℹ️ Esta pergunta já foi respondida. Veja o histórico acima.")
        else:
            st.session_state.processing = True
            st.session_state.last_submission_time = current_time
            st.session_state.processed_questions.add(question_hash)
            
            prompt = build_prompt_with_data(user_question, None)
            try:
                answer = call_model_with_timeout(prompt, timeout=60)
            except TimeoutError:
                answer = "⏱️ Tempo limite atingido. Tente novamente."
            except Exception as e:
                answer = f"❌ Erro: {str(e)[:200]}"
            
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": answer
            })
            
            st.session_state.processing = False
            st.rerun()
  
    if st.session_state.chat_history:
        if st.button("🧹 Limpar Conversa"):
            st.session_state.chat_history = []
            st.session_state.processed_questions = set()
            st.session_state.processing = False
            st.rerun()
  
    st.markdown('</div>', unsafe_allow_html=True)
  
    st.markdown("---")
  
    st.markdown(
        """
        <div class="panel" style="text-align:center; padding: 60px 20px;">
            <img src='https://img.icons8.com/fluency/150/data-configuration.png' width='150' style="margin-bottom:20px;"/>
            <h2 style="margin:10px 0; color: var(--text-color);">Bem-vindo ao InsightTab! 👋</h2>
            <p style="color:var(--muted-color); font-size: 1.1em; margin-bottom: 30px;">
                Carregue suas planilhas na barra lateral e comece a fazer perguntas inteligentes
            </p>
            <div style="background: var(--card-bg); padding: 20px; border-radius: 10px; max-width: 600px; margin: 0 auto;">
                <h3 style="color: var(--text-color); margin-top: 0;">🚀 Como usar:</h3>
                <ol style="text-align: left; color: var(--text-color)">
                <li>Clique no botão <span style='font-size: 1.2em; font-weight: bold;'>&lt;</span> no canto superior esquerdo para abrir a barra lateral.</li>
                <li>Conecte-se ao Google Sheets ou faça upload de arquivos Excel/CSV.</li>
                <li>Digite sua pergunta na caixa de chat (ex: 'Qual a receita total em Janeiro?').</li>
                <li>O InsightTab analisa seus dados e fornece a resposta.</li>
                </ol>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
