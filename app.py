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
@st.cache_data(ttl=300) # Cache por 5 minutos
def carregar_google_sheets():
    """Carrega dados das planilhas do Google Sheets"""
    if not GOOGLE_SHEETS_ENABLED:
        return {}
   
    try:
        # Parsear credenciais JSON
        creds_dict = json.loads(GOOGLE_SHEETS_CREDENTIALS)
       
        # Criar credenciais
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
       
        # Conectar ao Google Sheets
        client = gspread.authorize(credentials)
       
        dataframes = {}
       
        # Carregar cada planilha configurada
        for nome, sheet_id in SHEET_IDS.items():
            if sheet_id: # Só carregar se o ID foi configurado
                try:
                    spreadsheet = client.open_by_key(sheet_id)
                   
                    # Pegar todas as abas da planilha
                    for worksheet in spreadsheet.worksheets():
                        # Converter para DataFrame
                        data = worksheet.get_all_values()
                        if len(data) > 1: # Tem header e dados
                            df = pd.DataFrame(data[1:], columns=data[0])
                           
                            # Tentar converter colunas numéricas
                            for col in df.columns:
                                try:
                                    df[col] = pd.to_numeric(df[col])
                                except:
                                    pass
                           
                            # Nome da aba
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
MODEL_TIMEOUT = 60
MODEL_RETRIES = 2
RETRY_BACKOFF = 2.0
SAMPLE_SIZE = 500
MAX_OUTPUT_TOKENS = 1024
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
    # Tentar carregar do Google Sheets automaticamente
    st.session_state.dataframes = carregar_google_sheets()
    if st.session_state.dataframes:
        st.success(f"✅ {len(st.session_state.dataframes)} planilha(s) carregada(s) do Google Sheets!")
if "processing" not in st.session_state:
    st.session_state.processing = False
if "uploaded_file_keys" not in st.session_state:
    st.session_state.uploaded_file_keys = []
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

}
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
        content: "⭐";
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 8px 12px;
        border-radius: 8px;
        margin-right: 10px;
        font-size: 1.8rem;
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
    /* Cor do elemento ID/Nome (código inline) */
    .chat-message code {
        color: var(--accent) !important;
        background-color: rgba(102, 126, 234, 0.15) !important;
        padding: 2px 4px;
        border-radius: 4px;
        font-family: inherit !important;
        font-size: 0.9em;
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
    </style>
    """,
    unsafe_allow_html=True,
)
# ========== CONFIGURAR GEMINI ==========
try:
    model = genai.GenerativeModel("gemini-2.5-flash")
except Exception:
    model = None
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
    # Verifica se o nome começa com algum dos nomes das planilhas do Google Sheets
    for sheet_name in GOOGLE_SHEETS_NAMES:
        if filename.startswith(sheet_name):
            return True
    return False
def build_prompt_with_data(question, dataframes, sample_size=SAMPLE_SIZE):
    """Constrói prompt com dados das planilhas"""
    if not dataframes:
        return f"Nenhuma planilha foi carregada ainda.\n\nPergunta: {question}"
   
    all_data = ""
    total_rows = 0
   
    for filename, df in dataframes.items():
        if isinstance(df, dict):
            for sheet_name, sheet_df in df.items():
                preview = sheet_df.head(10).to_string(index=False)
                all_data += f"\n--- Planilha: {sheet_name} ({len(sheet_df)} linhas) ---\n{preview}\n"
                total_rows += len(sheet_df)
        else:
            preview = df.head(10).to_string(index=False)
            all_data += f"\n--- Planilha: {filename} ({len(df)} linhas) ---\n{preview}\n"
            total_rows += len(df)
   
    prompt = f"""Você é um analista de dados especializado.
DADOS CARREGADOS ({total_rows} linhas no total): {all_data}
PERGUNTA DO USUÁRIO: {question}
INSTRUÇÕES:
1. Analise APENAS os dados fornecidos acima
2. Responda em português brasileiro de forma clara e profissional
3. Use estatísticas e números EXATOS dos dados
4. Se a pergunta for sobre dados não presentes, informe isso
5. Seja objetivo e direto na resposta
6. Formate todos os valores monetários em Reais, usando o formato R$ X.XXX,XX (ex: R$ 42.173,01). O símbolo R$ deve ser colado ao valor.
7. Ao responder, NUNCA use negrito, itálico ou formatação de fonte que possa alterar o tipo de fonte do texto. Use APENAS a formatação de código inline do Markdown (texto entre crases, ex: `Monitor 4K`) para destacar nomes de itens, IDs de produtos e valores monetários.
Responda agora:"""
   
    return prompt
def _call_model_sync(prompt, max_output_tokens=MAX_OUTPUT_TOKENS):
    """Chamada síncrona ao modelo"""
    if model is None:
        raise RuntimeError("Modelo não configurado.")
   
    try:
        resp = model.generate_content(prompt, max_output_tokens=max_output_tokens)
    except TypeError:
        resp = model.generate_content(prompt)
   
    return getattr(resp, "text", str(resp))
def call_model_with_timeout(prompt, timeout=MODEL_TIMEOUT):
    """Chama modelo com timeout e retry"""
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
# ========== HEADER ==========
st.markdown('<h1 class="main-header">InsightTab - Analista Inteligente</h1>', unsafe_allow_html=True)
# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("### 📂 Gerenciar Dados")
   
    # Status do Google Sheets
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
   
    # Mostrar dados carregados
    if st.session_state.dataframes:
        st.markdown("---")
        st.markdown("### ✅ Dados Disponíveis")
        total_rows = 0
        for filename, df in st.session_state.dataframes.items():
            rows = len(df)
            # Identificar origem usando a nova função
            if is_google_sheets_data(filename):
                badge = "☁️" # Google Sheets
            else:
                badge = "📄" # Upload manual
            st.markdown(f"{badge} **{filename}**<br><small>{rows} linhas</small>", unsafe_allow_html=True)
            total_rows += rows
       
        st.markdown(f'<div style="margin-top: 10px; padding: 10px; background: var(--card-bg); border-radius: 8px; text-align: center;"><b>Total: {total_rows:,} linhas</b></div>', unsafe_allow_html=True)
   
    # Verificar se existem planilhas manuais usando a nova função
    has_manual_sheets = any(
        not is_google_sheets_data(filename) for filename in st.session_state.dataframes.keys()
    )
   
    if has_manual_sheets:
        if st.button("🗑️ Limpar Planilhas Manuais", use_container_width=True):
            # Manter apenas planilhas do Google Sheets usando a nova função
            google_sheets_data = {
                k: v for k, v in st.session_state.dataframes.items() if is_google_sheets_data(k)
            }
            st.session_state.dataframes = google_sheets_data
            st.session_state.uploaded_file_keys.append(time.time())
            st.success("✅ Planilhas manuais removidas!")
            st.rerun()
# ========== ÁREA PRINCIPAL - CHAT ==========
if st.session_state.dataframes:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
   
    # Histórico (exibido UMA ÚNICA VEZ)
    for chat in st.session_state.chat_history:
        st.markdown(
            f'<div class="chat-message user-message"><b>👤 Você:</b><br>{chat["question"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="chat-message bot-message"><b>🤖 InsightTab:</b><br>{chat["answer"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")
   
    # Input
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
   
    if submit and user_question and not st.session_state.processing:
        st.session_state.processing = True
        # Adicionar a pergunta ao histórico ANTES de processar
        st.session_state.chat_history.append({"question": user_question, "answer": "🤖 Analisando..."})
       
        prompt = build_prompt_with_data(user_question, st.session_state.dataframes)
        try:
            answer = call_model_with_timeout(prompt)
        except TimeoutError:
            answer = "⏱️ Tempo limite atingido. Tente uma pergunta mais simples."
        except Exception as e:
            answer = f"❌ Erro: {str(e)}"
       
        # Atualizar a última resposta no histórico
        st.session_state.chat_history[-1]["answer"] = answer
        st.session_state.processing = False
        st.rerun()
   
    if st.button("🧹 Limpar Conversa"):
        st.session_state.chat_history = []
        st.rerun()
   
    st.markdown('</div>', unsafe_allow_html=True)
   
    # Stats
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
    # Tela inicial (sem dados)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
   
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            st.markdown(
                f'<div class="chat-message user-message"><b>👤 Você:</b><br>{chat["question"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="chat-message bot-message"><b>🤖 InsightTab:</b><br>{chat["answer"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown("---")
   
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
        st.session_state.processing = True
        # Adicionar a pergunta ao histórico ANTES de processar
        st.session_state.chat_history.append({"question": user_question, "answer": "🤖 Analisando..."})
       
        prompt = build_prompt_with_data(user_question, None)
        try:
            answer = call_model_with_timeout(prompt, timeout=MODEL_TIMEOUT)
        except TimeoutError:
            answer = f"⏱️ Tempo limite atingido ({MODEL_TIMEOUT}s). Tente novamente."
        except Exception as e:
            answer = f"❌ Erro: {str(e)}"
       
        # Atualizar a última resposta no histórico
        st.session_state.chat_history[-1]["answer"] = answer
        st.session_state.processing = False
        st.rerun()
   
    if st.session_state.chat_history:
        if st.button("🧹 Limpar Conversa"):
            st.session_state.chat_history = []
            st.rerun()
   
    st.markdown('</div>', unsafe_allow_html=True)
   
    st.markdown("---")
   
    # Mensagem de boas-vindas
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
                <ol style="text-align: left; color: var(--text-color">
        """,
        unsafe_allow_html=True
    )
    # A última tag não está fechada, continuando a lista em Python...
    st.markdown(
        """
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
