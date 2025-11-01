# app.py
import os
import time
import re
import json
import threading
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from functools import wraps

from flask import Flask, request, render_template_string, redirect, url_for, send_file, flash
import pandas as pd
import google.generativeai as genai

# Opcional Google Sheets
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSHEET_AVAILABLE = True
except Exception:
    GSHEET_AVAILABLE = False

# ========== CONFIGURAÇÃO ==========
# Carregar chave Gemini da variável de ambiente
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise RuntimeError("Variável de ambiente GEMINI_API_KEY não definida. Coloque sua chave e reexecute.")

genai.configure(api_key=GEMINI_API_KEY)

# Modelo - tentativa de configurar
try:
    MODEL = genai.GenerativeModel("gemini-2.5-flash")
except Exception:
    MODEL = None

# Parâmetros
MODEL_TIMEOUT = 60            # segundos
MODEL_RETRIES = 2
RETRY_BACKOFF = 2.0
SAMPLE_SIZE = 500
MAX_OUTPUT_TOKENS = 1024

# Flask
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "troque_esta_chave_em_producao")

# Storage em memória (simples)
UPLOADED_DFS = {}   # {filename: DataFrame or {sheet_name: DataFrame}}
CHAT_HISTORY = []   # lista de dicts {"question":..., "answer":...}

# ========== UTILITÁRIOS ==========
def read_file_to_df(file_storage):
    """
    Lê arquivo enviado (CSV/XLSX/XLS).
    Se o Excel tiver múltiplas sheets, retorna dict {sheet_name: df}.
    """
    name = file_storage.filename.lower()
    file_storage.stream.seek(0)
    content = file_storage.read()
    bio = BytesIO(content)

    if name.endswith(".csv"):
        df = pd.read_csv(bio)
        return df
    else:
        # tenta com openpyxl (xlsx/xls)
        try:
            xls = pd.ExcelFile(bio, engine="openpyxl")
        except Exception as e:
            # fallback: pandas read_excel direta
            bio.seek(0)
            df = pd.read_excel(bio, engine="openpyxl")
            return df

        sheets = xls.sheet_names
        if len(sheets) == 1:
            df = xls.parse(sheets[0])
            xls.close()
            return df
        else:
            dfs = {}
            for s in sheets:
                dfs[s] = xls.parse(s)
            xls.close()
            return dfs

def read_gsheet(spreadsheet_id=None, creds_json_path=None, creds_json_string=None):
    """
    Lê uma planilha Google Sheets por ID (todas as abas).
    Retorna dict {sheet_name: df}
    Necessita `gspread` e credenciais de service account.
    """
    if not GSHEET_AVAILABLE:
        raise RuntimeError("gspread não disponível. Instale gspread e google-auth para usar Google Sheets.")
    if creds_json_string:
        creds_info = json.loads(creds_json_string)
        creds = Credentials.from_service_account_info(creds_info, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])
    elif creds_json_path and os.path.exists(creds_json_path):
        creds = Credentials.from_service_account_file(creds_json_path, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])
    else:
        raise RuntimeError("Credenciais do Google Sheets não fornecidas (path ou JSON string).")

    client = gspread.authorize(creds)
    sh = client.open_by_key(spreadsheet_id)
    worksheets = sh.worksheets()
    result = {}
    for ws in worksheets:
        values = ws.get_all_records()
        df = pd.DataFrame(values)
        result[ws.title] = df
    return result

def build_prompt_with_data(question, dataframes, sample_size=SAMPLE_SIZE):
    """
    Constrói prompt a partir dos dataframes carregados.
    dataframes: dict {filename: df_or_dict}
    """
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
2. Responda em português brasileiro de forma clara e profissional.
3. Use estatísticas e números EXATOS dos dados quando possível.
4. Formate valores monetários em reais (R$) quando apropriado.
5. Mantenha uma frase de destaque com fonte uniforme: "Comparando ambas, a maior venda de Abril".
6. Se a pergunta exigir dados que não estão nas planilhas, diga explicitamente.
Responda agora:"""
    return prompt

def _call_model_sync(prompt, max_output_tokens=MAX_OUTPUT_TOKENS):
    if MODEL is None:
        raise RuntimeError("Modelo não configurado. Verifique GEMINI_API_KEY.")
    # A API do genai pode variar; tentamos usar generate_content
    resp = MODEL.generate_content(prompt, max_output_tokens=max_output_tokens)
    return getattr(resp, "text", str(resp))

def call_model_with_timeout(prompt, timeout=MODEL_TIMEOUT):
    last_exc = None
    for attempt in range(1, MODEL_RETRIES + 1):
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_call_model_sync, prompt)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeout as te:
                future.cancel()
                last_exc = te
            except Exception as e:
                last_exc = e
        time.sleep(RETRY_BACKOFF ** (attempt - 1))
    raise last_exc

# Formatação monetária robusta (transforma padrões numéricos em R$ X.XXX,YY)
def format_currency_in_text(text):
    """
    Detecta números no texto e aplica prefixo R$ conforme heurísticas.
    Não é perfeito para todos os casos, mas cobre os formatos mais comuns:
      - 1.234,56  (BR)
      - 1234.56  (EN) -> será convertido para R$ 1.234,56
      - 1234    -> R$ 1.234 se for plausível
    Mantemos números pequenos (1, 2, 3) sem prefixo para evitar ruído.
    """
    if not isinstance(text, str):
        return text

    # Primeiro, já no formato BR (ex: 1.234,56)
    text = re.sub(r'(?<!R\$)\s(\d{1,3}(?:\.\d{3})*,\d{2})\b', r' R$ \1', text)

    # Converter números com ponto decimal (ex: 1234.56) para R$ 1.234,56
    def repl_en_decimal(m):
        val = m.group(0)
        try:
            f = float(val)
            # formata para BR
            br = f"{f:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            return f"R$ {br}"
        except Exception:
            return val

    text = re.sub(r'(?<!R\$)\b\d+\.\d{1,2}\b', repl_en_decimal, text)

    # Números inteiros grandes: 1000 -> R$ 1.000
    def repl_int(m):
        val = m.group(0)
        # se for pequeno (até 3 dígitos) não coloco R$
        if len(val) <= 3:
            return val
        try:
            i = int(val)
            br = f"{i:,}".replace(",", ".")
            return f"R$ {br}"
        except:
            return val

    text = re.sub(r'(?<!R\$)\b\d{4,}\b', repl_int, text)

    # Garante não duplicar R$
    text = re.sub(r'R\$\s+R\$\s*', 'R$ ', text)
    return text

# Segurança simples: limitar upload size (opcional)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))  # 5MB padrão

# ========== TEMPLATE HTML ==========
# Template com CSS que força modo escuro, corrige texto branco/branco em inputs/tables, e define fonte uniforme.
BASE_HTML = """
<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>InsightTab - Analista Inteligente (Flask)</title>
  <style>
    :root{
      --app-bg: #0b1116;
      --panel-bg: #12161a;
      --card-bg: #0f1720;
      --text-color: #e6eef3;
      --muted: #9aa6b2;
      --accent: #667eea;
      --font-family: "Inter", "Segoe UI", Roboto, Arial, sans-serif;
    }
    html,body {
      height:100%;
      margin:0;
      background:var(--app-bg);
      color:var(--text-color);
      font-family:var(--font-family);
    }
    .container{
      max-width:1100px;
      margin:20px auto;
      padding:20px;
    }
    header{
      text-align:center;
      margin-bottom:10px;
    }
    h1{
      font-size:28px;
      margin:0;
      background:linear-gradient(90deg,#667eea,#764ba2);
      -webkit-background-clip:text;
      -webkit-text-fill-color:transparent;
    }
    .panel{
      background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border-radius:12px;
      padding:18px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.5);
      border: 1px solid rgba(255,255,255,0.03);
      margin-bottom:16px;
    }
    .row{display:flex; gap:16px;}
    .col{flex:1;}
    .actions{display:flex; gap:10px; align-items:center;}
    input[type="file"]{background:var(--card-bg); color:var(--text-color); padding:10px; border-radius:8px; border:1px solid rgba(255,255,255,0.03)}
    input[type="text"], textarea{
      width:100%; padding:10px; border-radius:8px; background:var(--card-bg); color:var(--text-color); border:1px solid rgba(255,255,255,0.04)
    }
    button{
      background: linear-gradient(90deg, #667eea, #764ba2);
      color:white; border:none; padding:10px 14px; border-radius:8px; cursor:pointer; font-weight:600;
    }
    .file-list{margin-top:10px; color:var(--muted);}
    table{width:100%; border-collapse:collapse; margin-top:10px; font-size:0.95em;}
    th, td{padding:8px 10px; border-bottom:1px solid rgba(255,255,255,0.03); text-align:left; color:var(--text-color);}
    /* Corrige texto branco/branco em células com background claro (garante contraste) */
    td, th, caption, label, p, span, a, li {
      color: var(--text-color) !important;
      font-family: var(--font-family) !important;
    }
    .muted { color: var(--muted); font-size:0.9em; }
    .uniform-font { font-family: var(--font-family); font-size:1.05em; }
    .chat { max-height:320px; overflow:auto; padding:10px; background: rgba(255,255,255,0.01); border-radius:8px; }
    .user { background: rgba(66,153,225,0.07); padding:8px; border-left:4px solid rgba(66,153,225,1); margin-bottom:8px; border-radius:6px;}
    .bot { background: rgba(156,39,176,0.06); padding:8px; border-left:4px solid rgba(156,39,176,1); margin-bottom:8px; border-radius:6px;}
    .highlight { padding:8px; background: linear-gradient(90deg,#222533,#2a2f3b); border-radius:8px; margin-top:10px; }
    footer { text-align:center; color:var(--muted); margin-top:18px; font-size:0.9em; }
    .danger { color:#ff7b7b; font-weight:600; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>InsightTab - Analista Inteligente ⭐</h1>
      <p class="muted">Faça upload de suas planilhas e faça perguntas em linguagem natural.</p>
    </header>

    <div class="panel">
      <form method="post" action="/upload" enctype="multipart/form-data">
        <div class="row">
          <div class="col">
            <label for="files">Enviar arquivos (Excel/CSV)</label><br>
            <input type="file" id="files" name="files" multiple>
            <div class="file-list">
              {% if uploaded %}
                <strong>Planilhas carregadas:</strong>
                <ul>
                {% for name, info in uploaded.items() %}
                  <li>{{name}} <span class="muted">({{info.rows}} linhas{% if info.sheets %}, abas: {{info.sheets}}{% endif %})</span></li>
                {% endfor %}
                </ul>
              {% else %}
                Nenhuma planilha enviada ainda.
              {% endif %}
            </div>
          </div>
          <div class="col actions">
            <button type="submit">📥 Carregar</button>
            <a href="/clear" style="text-decoration:none"><button type="button">🗑️ Limpar</button></a>
          </div>
        </div>
      </form>
    </div>

    <div class="panel">
      <form method="post" action="/ask">
        <label for="question">💭 Faça sua pergunta sobre os dados:</label><br>
        <input type="text" id="question" name="question" placeholder="Ex: Qual produto vendeu mais em Abril?" required>
        <div style="margin-top:8px;">
          <button type="submit">📤 Perguntar</button>
          <a href="/history" style="margin-left:10px; color:var(--muted); text-decoration:none;">Ver histórico</a>
        </div>
      </form>
      <div style="margin-top:12px;">
        <div class="chat">
          {% for h in history %}
            <div class="user"><b>Você:</b> {{h.question}}</div>
            <div class="bot uniform-font"><b>InsightTab:</b><br> {{h.answer|safe}}</div>
          {% else %}
            <div class="muted">Nenhuma interação ainda. Pergunte algo usando o formulário acima.</div>
          {% endfor %}
        </div>
      </div>
    </div>

    <div class="panel">
      <h3 class="uniform-font">Visualizar amostra das planilhas</h3>
      {% if uploaded %}
        {% for name, info in uploaded.items() %}
          <div style="margin-top:12px;">
            <strong>{{name}}</strong> <span class="muted">({{info.rows}} linhas)</span>
            {% if info.preview_html %}
              <div class="highlight">{{info.preview_html|safe}}</div>
            {% endif %}
          </div>
        {% endfor %}
      {% else %}
        <div class="muted">Não há planilhas para visualizar.</div>
      {% endif %}
    </div>

    <footer>
      InsightTab — Exemplo integrado com Gemini. Configure GEMINI_API_KEY no ambiente. 
    </footer>
  </div>
</body>
</html>
"""

# ========== ROTAS ==========
@app.route("/", methods=["GET"])
def index():
    uploaded = {}
    for k, v in UPLOADED_DFS.items():
        if isinstance(v, dict):  # multi-sheet
            total_rows = sum(len(df) for df in v.values())
            sheets = ", ".join(v.keys())
            # preview: first sheet head
            first = next(iter(v.values()))
            preview_html = first.head(5).to_html(classes="table", index=False)
            uploaded[k] = {"rows": total_rows, "sheets": sheets, "preview_html": preview_html}
        else:
            uploaded[k] = {"rows": len(v), "sheets": None, "preview_html": v.head(5).to_html(classes="table", index=False)}
    # reverse history for display (most recent last)
    return render_template_string(BASE_HTML, uploaded=uploaded, history=CHAT_HISTORY)

@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    if not files:
        flash("Nenhum arquivo enviado.", "danger")
        return redirect(url_for("index"))
    for f in files:
        f.stream.seek(0)
        data = f.read()
        if len(data) > MAX_UPLOAD_BYTES:
            flash(f"Arquivo {f.filename} excede o limite de {MAX_UPLOAD_BYTES} bytes.", "danger")
            continue
        f.stream.seek(0)
        try:
            # reconstruct FileStorage-like object to pass to reader
            f.stream.seek(0)
            # pandas accepts BytesIO
            df_or_dict = read_file_to_df(f)
            UPLOADED_DFS[f.filename] = df_or_dict
        except Exception as e:
            flash(f"Erro ao ler {f.filename}: {e}", "danger")
    return redirect(url_for("index"))

@app.route("/clear", methods=["GET"])
def clear():
    UPLOADED_DFS.clear()
    CHAT_HISTORY.clear()
    return redirect(url_for("index"))

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()
    if not question:
        flash("Pergunta vazia.", "danger")
        return redirect(url_for("index"))

    # contruir prompt
    prompt = build_prompt_with_data(question, UPLOADED_DFS, SAMPLE_SIZE)

    # chamar modelo com timeout / retries
    try:
        raw_answer = call_model_with_timeout(prompt, timeout=MODEL_TIMEOUT)
    except Exception as e:
        raw_answer = f"❌ Erro ao chamar o modelo: {str(e)}"

    # formatar moeda
    answer_formatted = format_currency_in_text(raw_answer)

    # garantir que frase "Comparando ambas, a maior venda de Abril" tenha fonte uniforme no HTML
    # substituimos a frase (se aparecer) por um bloco com a classe uniform-font
    phrase = "Comparando ambas, a maior venda de Abril"
    if phrase in answer_formatted:
        answer_formatted = answer_formatted.replace(phrase, f'<span class="uniform-font">{phrase}</span>')

    CHAT_HISTORY.append({"question": question, "answer": answer_formatted})
    # limite histórico em memória (p.ex. 80)
    if len(CHAT_HISTORY) > 200:
        CHAT_HISTORY.pop(0)

    return redirect(url_for("index"))

@app.route("/history", methods=["GET"])
def history():
    # simples exibição do histórico em texto plano
    out = "<h2>Histórico</h2><ul>"
    for h in CHAT_HISTORY:
        out += f"<li><b>Q:</b> {h['question']}<br><b>A:</b> {h['answer']}</li><hr>"
    out += "</ul><a href='/'>Voltar</a>"
    return out

# ========== EXECUÇÃO ==========
if __name__ == "__main__":
    # roda em 0.0.0.0:5000 por padrão
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=debug)
