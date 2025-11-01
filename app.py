import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# 🔑 CONFIGURAÇÃO DA API - COLOQUE SUA API KEY AQUI:
# ═══════════════════════════════════════════════════════════════
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') # Pega do Render AUTOMATICAMENTE!

if not GEMINI_API_KEY:
    st.error("❌ API Key GEMINI_API_KEY não encontrada!")
    st.error("👉 Render Dashboard > Environment > Adicione: GEMINI_API_KEY = sua_chave_real")
    st.stop() # Para o app

# Configura Gemini (vai funcionar!)
genai.configure(api_key=GEMINI_API_KEY)

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
    st.session_state.dataframes = {} # {filename: dataframe}
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
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--text-color) !important;
            font-weight: 600 !important;
        }
        
        /* Labels da sidebar */
        [data-testid="stSidebar"] label {
            color: var(--text-color) !important;
            font-weight: 500 !important;
        }
        
        /* Texto da sidebar */
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: var(--text-color) !important;
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
        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea {
            background-color: var(--card-bg) !important;
            color: var(--text-color) !important;
            border: 1px solid rgba(102, 126, 234, 0.3) !important;
            border-radius: 8px;
        }
        
        .stTextInput>label,
        .stTextArea>label {
            color: var(--text-color) !important;
            font-weight: 500;
        }
        
        /* Placeholder branco */
        .stTextInput>div>div>input::placeholder,
        .stTextArea>div>div>textarea::placeholder {
            color: rgba(255, 255, 255, 0.5) !important;
            opacity: 1 !important;
        }
        
        /* Foco nos inputs */
        .stTextInput>div>div>input:focus,
        .stTextArea>div>div>textarea:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
        }
        
        /* Botões */
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

        /* Botão de Envio do formulário */
        div[data-testid="stForm"] .stButton button {
            background: #4a9eff !important;
            color: white !important;
            border: none !important;
            transition: all 0.3s;
        }

        div[data-testid="stForm"] .stButton button:hover {
            background: #3d8de6 !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(74, 158, 255, 0.5);
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
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: var(--panel-bg) !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: var(--text-color) !important;
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
    genai.configure(api_key=GEMINI_API_KEY)
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
                # Carregar todas as sheets
                dfs = {}
                for sheet in sheets:
                    bio.seek(0)
                    dfs[sheet] = pd.read_excel(bio, sheet_name=sheet, engine="openpyxl")
                xl.close()
                bio.close()
                return dfs # Retorna dict de DataFrames
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
    
    # Combinar dados de todas as planilhas
    all_data = ""
    total_rows = 0
    
    for filename, df in dataframes.items():
        if isinstance(df, dict): # Multi-sheet
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
1. Analise APENAS os dados fornecidos acima
2. Responda em português brasileiro de forma clara e profissional
3. Use estatísticas e números EXATOS dos dados
4. Se a pergunta for sobre dados não presentes, informe isso
5. Seja objetivo e direto na resposta
6. **FIX: Formate todos os valores monetários em Reais, usando o formato R$ X.XXX,XX (ex: R$ 42.173,01). O símbolo R$ deve ser colado ao valor.**
7. **FIX: Ao responder, NUNCA use negrito, itálico ou formatação de fonte que possa alterar o tipo de fonte do texto. Use APENAS a formatação de código inline do Markdown (texto entre crases, ex: `Monitor 4K`) para destacar nomes de itens, IDs de produtos e valores monetários.**

Responda agora:"""
    
    return prompt

def _call_model_sync(prompt, max_output_tokens=MAX_OUTPUT_TOKENS):
    """Chamada síncrona ao modelo"""
    if model is None:
        raise RuntimeError("Modelo não configurado. Verifique a API Key na linha 14.")
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

# ========== SIDEBAR - UPLOAD MÚLTIPLO ==========
with st.sidebar:
    st.markdown("### 📂 Upload de Planilhas")
    st.markdown('<p style="color: var(--muted-color); font-size: 0.9em;">Carregue uma ou várias planilhas simultaneamente</p>', unsafe_allow_html=True)
    
    # Upload múltiplo com key única
    file_uploader_key = f"file_uploader_{len(st.session_state.uploaded_file_keys)}"
    uploaded_files = st.file_uploader(
        "Arraste e solte os arquivos aqui",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        label_visibility="visible",
        key=file_uploader_key
    )
    
    # Processar uploads automaticamente
    if uploaded_files:
        new_files = []
        # Adicionar apenas arquivos que ainda não foram carregados
        for file in uploaded_files:
            if file.name not in st.session_state.dataframes:
                new_files.append(file)
        
        if new_files:
            with st.spinner(f"📊 Carregando {len(new_files)} arquivo(s)..."):
                for file in new_files:
                    try:
                        df = read_uploaded_file_to_df(file)
                        st.session_state.dataframes[file.name] = df
                    except Exception as e:
                        st.error(f"❌ Erro em {file.name}: {str(e)}")
            st.rerun()

    
    # Mostrar arquivos carregados
    if st.session_state.dataframes:
        st.markdown("---")
        st.markdown("### ✅ Planilhas Carregadas")
        
        total_rows = 0
        for filename, df in st.session_state.dataframes.items():
            if isinstance(df, dict): # Multi-sheet
                sheets_info = ", ".join([f"{name} ({len(sheet_df)} linhas)" for name, sheet_df in df.items()])
                st.markdown(f"**{filename}**<br><small>{sheets_info}</small>", unsafe_allow_html=True)
                total_rows += sum(len(sheet_df) for sheet_df in df.values())
            else:
                st.markdown(f"**{filename}**<br><small>{len(df)} linhas</small>", unsafe_allow_html=True)
                total_rows += len(df)
        
        st.markdown(f'<div style="margin-top: 10px; padding: 10px; background: var(--card-bg); border-radius: 8px; text-align: center;"><b>Total: {total_rows:,} linhas</b></div>', unsafe_allow_html=True)
        
        # Botão limpar - AGORA FUNCIONA
        if st.button("🗑️ Limpar Todas as Planilhas", use_container_width=True):
            st.session_state.dataframes = {} # Limpa os DataFrames
            st.session_state.chat_history = [] # Limpa o histórico de chat
            st.session_state.uploaded_file_keys.append(time.time()) # Força nova key para o uploader
            st.rerun()

# ========== ÁREA PRINCIPAL ==========
if st.session_state.dataframes:
    # Área do chat PRIMEIRO
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    
    # Histórico de chat
    chat_container = st.container()
    with chat_container:
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
    
    # Input de pergunta com form (evita duplicação)
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input(
            "💭 Faça sua pergunta sobre os dados:",
            placeholder="Ex: Qual é a média de vendas? Qual produto tem maior lucro?",
            key="chat_input",
            label_visibility="collapsed"
        )
        
        col_btn1, col_btn2 = st.columns([4, 1])
        with col_btn2:
            submit_btn = st.form_submit_button("📤 Enviar", use_container_width=True)
    
    # Processar pergunta APENAS quando submit
    if submit_btn and user_question and not st.session_state.processing:
        st.session_state.processing = True
        
        prompt = build_prompt_with_data(user_question, st.session_state.dataframes, SAMPLE_SIZE)
        
        with st.spinner("🤖 Analisando seus dados..."):
            try:
                answer = call_model_with_timeout(prompt, timeout=MODEL_TIMEOUT)
            except TimeoutError:
                answer = f"⏱️ Tempo limite atingido ({MODEL_TIMEOUT}s). Tente uma pergunta mais simples."
            except Exception as e:
                answer = f"❌ Erro: {str(e)}"
        
        # Adicionar ao histórico
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": answer
        })
        
        st.session_state.processing = False
        st.rerun()
    
    # Botão limpar chat
    if st.button("🧹 Limpar Conversa"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Stats boxes DEPOIS do chat
    total_files = len(st.session_state.dataframes)
    total_rows = sum(len(df) if not isinstance(df, dict) else sum(len(s) for s in df.values()) for df in st.session_state.dataframes.values())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stat-box"><h2 style="margin:0;">{total_files}</h2><p style="margin:0; opacity:0.8;">Planilhas</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-box"><h2 style="margin:0;">{total_rows:,}</h2><p style="margin:0; opacity:0.8;">Linhas Totais</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-box"><h2 style="margin:0;">{len(st.session_state.chat_history)}</h2><p style="margin:0; opacity:0.8;">Perguntas</p></div>', unsafe_allow_html=True)

else:
    # Tela inicial (sem dados) - MAS COM CHAT
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    
    # Histórico de chat (se houver)
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
    
    # Form para enviar perguntas mesmo sem dados
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
        prompt = build_prompt_with_data(user_question, None)
        
        with st.spinner("🤖 Analisando..."):
            try:
                answer = call_model_with_timeout(prompt, timeout=MODEL_TIMEOUT)
            except TimeoutError:
                answer = f"⏱️ Tempo limite atingido ({MODEL_TIMEOUT}s). Tente novamente."
            except Exception as e:
                answer = f"❌ Erro: {str(e)}"
        
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": answer
        })
        st.session_state.processing = False
        st.rerun()
    
    # Botão limpar chat
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
                <ol style="text-align: left; color: var(--text-color); line-height: 1.8;">
                    <li>Faça upload de uma ou mais planilhas (Excel/CSV)</li>
                    <li>Os dados serão carregados automaticamente</li>
                    <li>Digite sua pergunta no chat acima</li>
                    <li>A IA analisa e responde baseado nos seus dados</li>
                </ol>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
