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
}

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
@st.cache_data(ttl=300)  # Cache por 5 minutos
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
            if sheet_id:  # Só carregar se o ID foi configurado
                try:
                    spreadsheet = client.open_by_key(sheet_id)
                    
                    # Pegar todas as abas da planilha
                    for worksheet in spreadsheet.worksheets():
                        # Converter para DataFrame
                        data = worksheet.get_all_values()
                        if len(data) > 1:  # Tem header e dados
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
    layout="wide"
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

# ========== CSS (mantido igual ao original) ==========
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
        .main-header {
            font-size: 2.8rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 1rem;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
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
        }
        .user-message {
            background: rgba(66,153,225,0.12);
            border-left: 4px solid rgba(66,153,225,1);
        }
        .bot-message {
            background: rgba(156,39,176,0.1);
            border-left: 4px solid rgba(156,39,176,1);
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
        .stButton>button {
            background: linear-gradient(90deg, #667eea, #764ba2);
            color: white !important;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
        }
        div[data-testid="stForm"] button[kind="formSubmit"] {
            background: #1a3a5c !important;
            color: white !important;
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

DADOS CARREGADOS ({total_rows} linhas no total):
{all_data}

PERGUNTA DO USUÁRIO: {question}

INSTRUÇÕES:
1. Analise APENAS os dados fornecidos acima
2. Responda em português brasileiro de forma clara e profissional
3. Use estatísticas e números EXATOS dos dados
4. Se a pergunta for sobre dados não presentes, informe isso
5. Seja objetivo e direto na resposta
6. Formate valores monetários em Reais: R$ X.XXX,XX
7. Use código inline markdown para destacar nomes e valores

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
st.markdown('<h1 class="main-header">📊 InsightTab - Google Sheets Edition</h1>', unsafe_allow_html=True)

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
            # Identificar origem
            if any(month in filename for month in ["Janeiro", "Fevereiro", "Março"]):
                badge = "☁️"  # Google Sheets
            else:
                badge = "📄"  # Upload manual
            
            st.markdown(f"{badge} **{filename}**<br><small>{rows} linhas</small>", unsafe_allow_html=True)
            total_rows += rows
        
        st.markdown(f'<div style="margin-top: 10px; padding: 10px; background: var(--card-bg); border-radius: 8px; text-align: center;"><b>Total: {total_rows:,} linhas</b></div>', unsafe_allow_html=True)

# ========== ÁREA PRINCIPAL - CHAT ==========
st.markdown('<div class="panel">', unsafe_allow_html=True)

# Histórico
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
    
    prompt = build_prompt_with_data(user_question, st.session_state.dataframes)
    
    with st.spinner("🤖 Analisando..."):
        try:
            answer = call_model_with_timeout(prompt)
        except TimeoutError:
            answer = "⏱️ Tempo limite atingido. Tente uma pergunta mais simples."
        except Exception as e:
            answer = f"❌ Erro: {str(e)}"
    
    st.session_state.chat_history.append({"question": user_question, "answer": answer})
    st.session_state.processing = False
    st.rerun()

if st.button("🧹 Limpar Conversa"):
    st.session_state.chat_history = []
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Stats
if st.session_state.dataframes:
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
