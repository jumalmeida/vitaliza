"""
Vitaliza — Aplicação Completa (API + Frontend)
================================================
Artefato Tech - Modulo 2 - MBA Inteli - Grupo 02

Execucao:
    python vitaliza_app_completo.py

Acesse:
    http://localhost:8000          -> Dashboard completo
    http://localhost:8000/docs     -> Swagger (API)
    http://localhost:8000/predict  -> Endpoint de predicao (POST)
"""

import os, json, io, warnings
import numpy as np
import pandas as pd
import joblib
import shap as shap_lib
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# ── artefatos serializados ─────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "modelo")

modelo    = joblib.load(os.path.join(MODEL_DIR, "vitaliza_modelo.joblib"))
scaler    = joblib.load(os.path.join(MODEL_DIR, "vitaliza_scaler.joblib"))
metadata  = joblib.load(os.path.join(MODEL_DIR, "vitaliza_metadata.joblib"))
explainer = joblib.load(os.path.join(MODEL_DIR, "vitaliza_explainer.joblib"))

FEATURES       = metadata["features"]
FEATURE_LABELS = metadata["feature_labels"]

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vitaliza — Sistema de Retenção Preditiva",
    version="2.0.0",
    description="MBA Inteli · Módulo 2 · Grupo 02",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── schemas ────────────────────────────────────────────────────────────────────
class EntradaPredicao(BaseModel):
    genero:              int   = Field(1,   ge=0, le=1)
    proximo_unidade:     int   = Field(1,   ge=0, le=1)
    plano_duplo:         int   = Field(0,   ge=0, le=1)
    indicado_amigo:      int   = Field(0,   ge=0, le=1)
    telefone_cadastrado: int   = Field(1,   ge=0, le=1)
    desafios_grupo:      int   = Field(0,   ge=0, le=1)
    idade:               int   = Field(28,  ge=18, le=70)
    gasto_adicional:     float = Field(45., ge=0)
    meses_renovacao:     float = Field(1.,  ge=0)
    tempo_assinatura:    int   = Field(1,   ge=0)
    freq_historica:      float = Field(1.2, ge=0)
    freq_mes_atual:      float = Field(0.0, ge=0)
    duracao_contrato:    int   = Field(1,   ge=1)

# ── helpers ────────────────────────────────────────────────────────────────────
def preparar_X(e: EntradaPredicao) -> np.ndarray:
    delta = e.freq_historica - e.freq_mes_atual
    v = [e.genero, e.proximo_unidade, e.plano_duplo, e.indicado_amigo,
         e.telefone_cadastrado, e.desafios_grupo, e.idade, e.gasto_adicional,
         e.meses_renovacao, e.tempo_assinatura, e.freq_historica,
         e.freq_mes_atual, e.duracao_contrato, delta]
    return np.array(v).reshape(1, -1)

def nivel_risco(p: float) -> str:
    return "Crítico" if p > .85 else "Alto" if p > .65 else "Moderado" if p > .35 else "Baixo"

def cor_risco(p: float) -> str:
    return "#c0392b" if p > .85 else "#e67e22" if p > .65 else "#f39c12" if p > .35 else "#27ae60"

def cluster_info(e: EntradaPredicao) -> tuple:
    f_a, f_h = e.freq_mes_atual, e.freq_historica
    delta = f_h - f_a
    social = e.desafios_grupo or e.indicado_amigo
    if f_a >= 2.4 or f_h >= 2.4:
        return "C2", "Entusiasta Frequente", "Uso intenso — alvo de upsell para plano anual"
    if e.duracao_contrato >= 6 and social and f_a >= 1.5:
        return "C1", "Comprometido Consistente", "Baixo risco — não acionar intervenção ativa"
    if e.duracao_contrato == 1 and f_a < 1.3 and not social:
        return "C0", "Usuário Invisível", "Risco alto — onboarding social e conversão contratual urgentes"
    if delta > 0.3 or (not social and f_a < 1.8):
        return "C3", "Remoto Desconectado", "Queda de uso — criar âncora social antes da renovação"
    return "C1", "Comprometido Consistente", "Estável — monitorar passivamente"

def recomendacoes(cluster: str, e: EntradaPredicao, prob: float) -> list:
    delta = e.freq_historica - e.freq_mes_atual
    recs = []
    if prob > 0.75:
        recs.append(f"⚠️ URGENTE: score {prob*100:.0f}% — acionar CS hoje")
    if cluster == "C0":
        recs += ["Onboarding Social (Dia 1-7): desafio em grupo obrigatório",
                 "Conversão Contratual (Dia 25): oferta de plano semestral"]
        if e.freq_mes_atual == 0:
            recs.append("CS Proativo 48h: usuário inativo — contato WhatsApp")
    elif cluster == "C1":
        recs += ["Sleeping Dog — não disparar intervenção ativa",
                 "Programa de Indicações: ativar como embaixador (CAC zero)"]
    elif cluster == "C2":
        recs += ["Upsell Plano Anual: desconto de até 15% na renovação",
                 "Gamificação: badges e ranking de consistência"]
    else:
        if delta >= 0.3:
            recs.append(f"Alerta Queda Frequência: delta={delta:.1f}x/sem — CS em 48h")
        recs += ["Âncora Social: emparelhar com peers em desafio",
                 "Survey JTBD Dia 1: personalizar trilha"]
    return recs

# ── endpoints REST ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "auc": metadata["auc_roc_teste"], "versao": "2.0.0"}

@app.get("/modelo/info")
def info():
    return {k: metadata[k] for k in [
        "algoritmo","data_treinamento","auc_roc_teste","avg_precision_teste",
        "brier_score_teste","cv_auc_media","cv_auc_std","gap_overfit",
        "n_registros_treino","feature_importances","shap_global","confusao"
    ]}

@app.post("/predict")
def predict(e: EntradaPredicao):
    X = preparar_X(e)
    X_s = scaler.transform(X)
    prob = float(modelo.predict_proba(X_s)[0, 1])
    sv   = explainer.shap_values(X_s)[0]
    delta = e.freq_historica - e.freq_mes_atual
    vals = [e.genero, e.proximo_unidade, e.plano_duplo, e.indicado_amigo,
            e.telefone_cadastrado, e.desafios_grupo, e.idade, e.gasto_adicional,
            e.meses_renovacao, e.tempo_assinatura, e.freq_historica,
            e.freq_mes_atual, e.duracao_contrato, delta]
    shap_items = sorted([
        {"variavel": f, "label": FEATURE_LABELS[f],
         "valor": float(v), "shap": float(s),
         "direcao": "aumenta_risco" if s > 0 else "reduz_risco"}
        for f, v, s in zip(FEATURES, vals, sv) if abs(s) > 1e-4
    ], key=lambda x: abs(x["shap"]), reverse=True)

    c_id, c_nome, c_desc = cluster_info(e)
    nivel = nivel_risco(prob)
    top3 = shap_items[:3]
    explicacao = (
        f"Nível de risco: {nivel} ({prob*100:.1f}%). "
        f"Perfil: {c_nome}. "
        "Principais fatores: " +
        "; ".join(f"{i['label']} = {i['valor']:.2f} → {'eleva' if i['direcao']=='aumenta_risco' else 'reduz'} o risco (SHAP {i['shap']:+.3f})"
                  for i in top3) + "."
    )
    return {
        "probabilidade_cancelamento": round(prob, 4),
        "nivel_risco": nivel,
        "cluster": c_id,
        "cluster_nome": c_nome,
        "cluster_descricao": c_desc,
        "recomendacoes": recomendacoes(c_id, e, prob),
        "shap_local": shap_items,
        "explicacao_linguagem_natural": explicacao,
        "metadata_modelo": {
            "algoritmo": metadata["algoritmo"],
            "auc_roc": metadata["auc_roc_teste"],
            "data_treino": metadata["data_treinamento"][:10],
            "versao": "2.0.0",
        }
    }

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    rename = {"gender":"genero","Near_Location":"proximo_unidade","Partner":"plano_duplo",
              "Promo_friends":"indicado_amigo","Phone":"telefone_cadastrado",
              "Contract_period":"duracao_contrato","Group_visits":"desafios_grupo",
              "Age":"idade","Avg_additional_charges_total":"gasto_adicional",
              "Month_to_end_contract":"meses_renovacao","Lifetime":"tempo_assinatura",
              "Avg_class_frequency_total":"freq_historica",
              "Avg_class_frequency_current_month":"freq_mes_atual"}
    df = df.rename(columns=rename)
    df["delta_freq"] = df["freq_historica"] - df["freq_mes_atual"]
    feats = [f for f in FEATURES if f in df.columns]
    X_s = scaler.transform(df[feats].fillna(0))
    probs = modelo.predict_proba(X_s)[:, 1]
    res = [{"linha": i+1, "prob": round(float(p),4), "risco": nivel_risco(p)} for i,p in enumerate(probs)]
    dist = {n: sum(1 for r in res if r["risco"]==n) for n in ["Crítico","Alto","Moderado","Baixo"]}
    return {"total": len(res), "distribuicao": dist, "predicoes": res}

class BatchJsonRequest(BaseModel):
    registros: list[EntradaPredicao]

@app.post("/predict/batch/json")
def predict_batch_json(req: BatchJsonRequest):
    resultados = []
    for e in req.registros:
        X = preparar_X(e)
        X_s = scaler.transform(X)
        prob = float(modelo.predict_proba(X_s)[0, 1])
        delta = e.freq_historica - e.freq_mes_atual
        c_id, c_nome, _ = cluster_info(e)
        resultados.append({
            "probabilidade_cancelamento": round(prob, 4),
            "nivel_risco": nivel_risco(prob),
            "cluster": c_id,
            "cluster_nome": c_nome,
            "delta_freq": round(delta, 3),
        })
    dist = {n: sum(1 for r in resultados if r["nivel_risco"]==n)
            for n in ["Crítico","Alto","Moderado","Baixo"]}
    return {"total": len(resultados), "distribuicao": dist, "resultados": resultados}

# ── FRONTEND HTML ─────────────────────────────────────────────────────────────
_SHAP_JSON = json.dumps(metadata["shap_global"])
_META_JSON = json.dumps({
    "auc": round(metadata["auc_roc_teste"], 4),
    "ap": round(metadata["avg_precision_teste"], 4),
    "brier": round(metadata["brier_score_teste"], 4),
    "cv_auc": round(metadata["cv_auc_media"], 4),
    "cv_std": round(metadata["cv_auc_std"], 4),
    "gap": round(metadata["gap_overfit"], 4),
    "n_treino": metadata["n_registros_treino"],
    "algoritmo": metadata["algoritmo"],
    "data": metadata["data_treinamento"][:10],
    "cm": metadata["confusao"],
})

def _load_html() -> str:
    html_path = os.path.join(BASE_DIR, "vitaliza_frontend.html")
    with open(html_path, encoding="utf-8") as f:
        return (f.read()
                .replace("{{SHAP_DATA}}", _SHAP_JSON)
                .replace("{{META_DATA}}", _META_JSON))

SHAP_GLOBAL_JSON = _SHAP_JSON
META_JSON        = _META_JSON

_HTML_LEGACY = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vitaliza — Sistema de Retenção Preditiva</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{--navy:#1a2644;--red:#c0392b;--teal:#0d6e6e;--bg:#f7f6f2;--white:#fff;--gray:#6b7280}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:#111;min-height:100vh}
nav{background:var(--navy);padding:14px 32px;display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:100}
.logo{width:38px;height:38px;background:var(--red);display:flex;align-items:center;justify-content:center;color:#fff;font-size:20px;font-weight:700;flex-shrink:0}
.logo-txt{color:#fff;font-size:17px;font-weight:600}
.logo-sub{color:rgba(255,255,255,.4);font-size:10px;letter-spacing:2px;text-transform:uppercase}
.tabs{display:flex;gap:4px;margin-left:auto}
.tab{background:rgba(255,255,255,.08);color:rgba(255,255,255,.7);border:none;padding:8px 16px;cursor:pointer;font-size:12px;font-weight:600;letter-spacing:.5px;text-transform:uppercase;border-radius:3px;transition:all .2s}
.tab.active,.tab:hover{background:var(--red);color:#fff}
.pane{display:none;padding:40px 40px 80px;max-width:1200px;margin:0 auto}
.pane.active{display:block}
.hero{background:var(--navy);padding:60px 48px;margin:-40px -40px 48px;color:#fff;position:relative;overflow:hidden}
.hero::after{content:'';position:absolute;top:-40px;right:-40px;width:240px;height:240px;border:48px solid rgba(192,57,43,.2);border-radius:50%;pointer-events:none}
.hero-label{color:rgba(255,255,255,.35);font-size:10px;letter-spacing:3px;text-transform:uppercase;margin-bottom:12px}
.hero h1{font-size:44px;line-height:1.15;margin-bottom:16px}
.hero h1 em{color:var(--red);font-style:normal}
.hero p{color:rgba(255,255,255,.55);font-size:16px;max-width:560px;line-height:1.7}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:40px}
.kpi{background:#fff;border:1px solid rgba(0,0,0,.06);padding:20px;border-radius:4px}
.kpi-label{font-size:11px;color:var(--gray);text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px}
.kpi-val{font-size:28px;font-weight:700}
.kpi-val.red{color:var(--red)} .kpi-val.teal{color:var(--teal)} .kpi-val.warn{color:#e67e22}
.kpi-sub{font-size:11px;color:var(--gray);margin-top:4px}
.section-title{font-size:24px;font-weight:700;margin-bottom:24px;padding-bottom:12px;border-bottom:2px solid var(--navy)}
.section-label{font-size:10px;font-weight:700;color:var(--red);letter-spacing:2px;text-transform:uppercase;margin-bottom:4px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:32px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-bottom:32px}
.card{background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:4px;padding:24px}
.card.navy{background:var(--navy);color:#fff}
.card.bordered-l{border-left:4px solid var(--navy)}
.card.bordered-red{border-left:4px solid var(--red)}
.card.bordered-teal{border-left:4px solid var(--teal)}
.form-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
.form-group label{display:block;font-size:12px;font-weight:600;color:var(--gray);margin-bottom:6px;text-transform:uppercase;letter-spacing:.3px}
.form-group input,.form-group select{width:100%;padding:10px 12px;border:1px solid #d1d5db;border-radius:4px;font-size:14px;background:#fff;color:#111}
.form-group input:focus,.form-group select:focus{outline:2px solid var(--teal);border-color:var(--teal)}
.btn-predict{background:var(--red);color:#fff;border:none;padding:14px 40px;font-size:14px;font-weight:700;letter-spacing:1px;text-transform:uppercase;cursor:pointer;border-radius:3px;margin-top:24px;transition:background .2s}
.btn-predict:hover{background:#a93226}
.btn-predict:disabled{background:#9ca3af;cursor:not-allowed}
.result-box{background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:4px;padding:28px;margin-top:28px;display:none}
.prob-bar-wrap{background:#f3f4f6;border-radius:999px;height:16px;overflow:hidden;margin:12px 0}
.prob-bar{height:100%;border-radius:999px;transition:width .6s ease}
.shap-row{display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:13px}
.shap-label{min-width:220px;color:#374151}
.shap-track{flex:1;background:#f3f4f6;border-radius:4px;height:10px;overflow:hidden}
.shap-fill-pos{background:#c0392b;height:100%;border-radius:4px}
.shap-fill-neg{background:#0d6e6e;height:100%;border-radius:4px}
.shap-val{min-width:56px;text-align:right;font-family:monospace;font-size:12px;font-weight:600}
.badge{display:inline-block;padding:4px 12px;border-radius:3px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px}
.badge.red{background:#fee2e2;color:var(--red)}
.badge.orange{background:#fef3c7;color:#92400e}
.badge.yellow{background:#fefce8;color:#713f12}
.badge.green{background:#d1fae5;color:#065f46}
.rec-item{display:flex;gap:10px;align-items:flex-start;padding:10px 0;border-bottom:1px solid #f3f4f6;font-size:14px}
.rec-dot{width:6px;height:6px;border-radius:50%;background:var(--red);flex-shrink:0;margin-top:6px}
.api-status{display:flex;align-items:center;gap:8px;font-size:12px;margin-top:8px}
.status-dot{width:8px;height:8px;border-radius:50%}
.status-dot.online{background:#22c55e} .status-dot.offline{background:#ef4444}
.metric-row{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #f3f4f6;font-size:14px}
.metric-row:last-child{border-bottom:none}
.metric-name{color:var(--gray)}
.metric-val{font-weight:700;font-family:monospace}
.metric-val.good{color:var(--teal)} .metric-val.warn{color:#e67e22}
.legend-item{display:flex;align-items:center;gap:6px;font-size:12px}
.legend-dot{width:10px;height:10px;border-radius:50%}
@media(max-width:768px){
  .kpis{grid-template-columns:1fr 1fr}.grid2,.grid3,.form-grid{grid-template-columns:1fr}
  .hero h1{font-size:28px}.pane{padding:20px}
}
</style>
</head>
<body>

<nav>
  <div class="logo">V</div>
  <div>
    <div class="logo-txt">Vitaliza</div>
    <div class="logo-sub">Sistema de Retenção Preditiva</div>
  </div>
  <div class="tabs">
    <button class="tab active" onclick="showTab('predicao')">Predição</button>
    <button class="tab" onclick="showTab('modelo')">Modelo & Métricas</button>
    <button class="tab" onclick="showTab('negocio')">Contexto de Negócio</button>
    <button class="tab" onclick="showTab('api')">API</button>
  </div>
</nav>

<!-- ═══════════ ABA 1: PREDICAO ═══════════ -->
<div id="pane-predicao" class="pane active">
  <div class="hero">
    <div class="hero-label">Vitaliza · Sistema de Retenção Preditiva · v2.0</div>
    <h1>Qual é o risco de <em>cancelamento</em><br>deste assinante?</h1>
    <p>Gradient Boosting treinado sobre 4.000 assinantes. AUC-ROC: <strong>0,9884</strong>. SHAP values para cada predição.</p>
    <div class="api-status">
      <div class="status-dot" id="status-dot"></div>
      <span id="status-text">Verificando API...</span>
    </div>
  </div>

  <div class="card" style="margin-bottom:28px">
    <div class="section-label">Dados do Assinante</div>
    <div class="form-grid" style="margin-top:16px">
      <div class="form-group"><label>Duração do Contrato</label>
        <select id="duracao_contrato"><option value="1">Mensal (1 mês)</option><option value="6">Semestral (6 meses)</option><option value="12">Anual (12 meses)</option></select></div>
      <div class="form-group"><label>Freq. Sessões — Mês Atual (x/sem)</label><input type="number" id="freq_mes_atual" value="0.0" step="0.1" min="0" max="7"></div>
      <div class="form-group"><label>Freq. Sessões — Histórico (x/sem)</label><input type="number" id="freq_historica" value="1.2" step="0.1" min="0" max="7"></div>
      <div class="form-group"><label>Tempo de Assinatura (meses)</label><input type="number" id="tempo_assinatura" value="1" min="0" max="100"></div>
      <div class="form-group"><label>Meses até Renovação</label><input type="number" id="meses_renovacao" value="1" step="0.5" min="0" max="12"></div>
      <div class="form-group"><label>Gasto Adicional Médio (R$)</label><input type="number" id="gasto_adicional" value="45" step="1" min="0"></div>
      <div class="form-group"><label>Idade</label><input type="number" id="idade" value="28" min="18" max="70"></div>
      <div class="form-group"><label>Desafios em Grupo</label>
        <select id="desafios_grupo"><option value="0">Não participa</option><option value="1">Participa</option></select></div>
      <div class="form-group"><label>Indicado por Amigo</label>
        <select id="indicado_amigo"><option value="0">Não</option><option value="1">Sim</option></select></div>
      <div class="form-group"><label>Próximo a uma Unidade</label>
        <select id="proximo_unidade"><option value="1">Sim</option><option value="0">Não</option></select></div>
      <div class="form-group"><label>Plano Duplo/Parceiro</label>
        <select id="plano_duplo"><option value="0">Não</option><option value="1">Sim</option></select></div>
      <div class="form-group"><label>Telefone Cadastrado</label>
        <select id="telefone_cadastrado"><option value="1">Sim</option><option value="0">Não</option></select></div>
    </div>
    <button class="btn-predict" id="btn-predict" onclick="predict()">Calcular Risco de Cancelamento</button>
  </div>

  <div class="result-box" id="result-box">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px;flex-wrap:wrap;gap:12px">
      <div>
        <div class="section-label">Resultado da Predição</div>
        <div style="font-size:42px;font-weight:800;margin-top:4px" id="res-prob">—</div>
        <div style="font-size:14px;color:var(--gray);margin-top:4px">probabilidade de cancelamento</div>
      </div>
      <div style="text-align:right">
        <div class="badge" id="res-badge">—</div>
        <div style="font-size:13px;color:var(--gray);margin-top:8px" id="res-cluster">—</div>
      </div>
    </div>
    <div class="prob-bar-wrap"><div class="prob-bar" id="res-bar"></div></div>
    <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--gray);margin-bottom:24px">
      <span>0%</span><span>Limiar 50%</span><span>100%</span>
    </div>

    <div class="grid2">
      <div>
        <div class="section-label" style="margin-bottom:12px">SHAP — Fatores Explicativos</div>
        <div id="shap-container"></div>
      </div>
      <div>
        <div class="section-label" style="margin-bottom:12px">Recomendações Operacionais</div>
        <div id="recs-container"></div>
      </div>
    </div>

    <div class="card" style="background:#f9fafb;margin-top:20px">
      <div class="section-label" style="margin-bottom:8px">Explicação em Linguagem Natural</div>
      <p id="res-explicacao" style="font-size:14px;line-height:1.7;color:#374151"></p>
      <div style="font-size:11px;color:var(--gray);margin-top:12px" id="res-meta"></div>
    </div>
  </div>
</div>

<!-- ═══════════ ABA 2: MODELO ═══════════ -->
<div id="pane-modelo" class="pane">
  <div class="hero">
    <div class="hero-label">Metodologia & Validação</div>
    <h1>Gradient Boosting.<br><em>AUC 0,9884.</em> Sem overfit.</h1>
    <p>Pipeline de treinamento com split estratificado, validação cruzada 5-folds e SHAP values para explicabilidade global e local.</p>
  </div>

  <div class="kpis">
    <div class="kpi"><div class="kpi-label">AUC-ROC (teste)</div><div class="kpi-val teal" id="m-auc">—</div><div class="kpi-sub">Ideal: próximo de 1,0</div></div>
    <div class="kpi"><div class="kpi-label">Avg Precision</div><div class="kpi-val teal" id="m-ap">—</div><div class="kpi-sub">Curva Precisão-Recall</div></div>
    <div class="kpi"><div class="kpi-label">Gap Overfit (CV)</div><div class="kpi-val green" id="m-gap">—</div><div class="kpi-sub">Sem overfitting (&lt; 0,05)</div></div>
    <div class="kpi"><div class="kpi-label">Brier Score</div><div class="kpi-val teal" id="m-brier">—</div><div class="kpi-sub">Calibração (0=perfeito)</div></div>
  </div>

  <div class="grid2">
    <div class="card">
      <div class="section-label">Métricas Detalhadas</div>
      <div id="metrics-detail" style="margin-top:12px"></div>
    </div>
    <div class="card">
      <div class="section-label">Matriz de Confusão (teste)</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:16px">
        <div style="text-align:center;padding:16px;background:#d1fae5;border-radius:4px">
          <div style="font-size:10px;color:var(--gray);text-transform:uppercase;margin-bottom:4px">Verdadeiro Negativo</div>
          <div style="font-size:32px;font-weight:700;color:#065f46" id="cm-tn">—</div></div>
        <div style="text-align:center;padding:16px;background:#fee2e2;border-radius:4px">
          <div style="font-size:10px;color:var(--gray);text-transform:uppercase;margin-bottom:4px">Falso Positivo</div>
          <div style="font-size:32px;font-weight:700;color:var(--red)" id="cm-fp">—</div></div>
        <div style="text-align:center;padding:16px;background:#fee2e2;border-radius:4px">
          <div style="font-size:10px;color:var(--gray);text-transform:uppercase;margin-bottom:4px">Falso Negativo</div>
          <div style="font-size:32px;font-weight:700;color:var(--red)" id="cm-fn">—</div></div>
        <div style="text-align:center;padding:16px;background:#d1fae5;border-radius:4px">
          <div style="font-size:10px;color:var(--gray);text-transform:uppercase;margin-bottom:4px">Verdadeiro Positivo</div>
          <div style="font-size:32px;font-weight:700;color:#065f46" id="cm-tp">—</div></div>
      </div>
    </div>
  </div>

  <div class="card" style="margin-bottom:28px">
    <div class="section-label" style="margin-bottom:16px">SHAP Global — Importância Média Absoluta</div>
    <canvas id="shap-chart" height="320"></canvas>
  </div>

  <div class="card">
    <div class="section-label" style="margin-bottom:12px">Governança Contra Data Leakage</div>
    <div class="grid3" style="margin-top:12px">
      <div class="card bordered-teal"><strong>Split Estratificado</strong><br><small style="color:var(--gray)">80% treino / 20% teste. Nenhum dado do teste visto durante o fit.</small></div>
      <div class="card bordered-teal"><strong>Scaler Separado</strong><br><small style="color:var(--gray)">StandardScaler fittado apenas no conjunto de treino. Aplicado no teste sem refit.</small></div>
      <div class="card bordered-teal"><strong>CV 5-Folds</strong><strong></strong><br><small style="color:var(--gray)">StratifiedKFold garante proporção de classes em cada fold. Gap treino-validação: 0,011.</small></div>
    </div>
  </div>
</div>

<!-- ═══════════ ABA 3: NEGOCIO ═══════════ -->
<div id="pane-negocio" class="pane">
  <div class="hero">
    <div class="hero-label">Business Case — Módulo 2 · Inteli MBA</div>
    <h1>Vitaliza: A Arquitetura<br>da <em>Retenção Preditiva</em></h1>
    <p>R$ 113.500/mês saindo sem resistência. Churn 10,2%. LTV/CAC em 2,02×. Série B em risco.</p>
  </div>

  <div class="kpis">
    <div class="kpi"><div class="kpi-label">Churn Mensal (set/2025)</div><div class="kpi-val red">10,2%</div><div class="kpi-sub">Meta: 6,0% até Q4/2026</div></div>
    <div class="kpi"><div class="kpi-label">MRR Perdido/Mês</div><div class="kpi-val red">R$ 113,5k</div><div class="kpi-sub">2.847 cancelamentos × R$ 39,90</div></div>
    <div class="kpi"><div class="kpi-label">LTV/CAC Atual</div><div class="kpi-val warn">2,02×</div><div class="kpi-sub">Piso Série B: 3,0×</div></div>
    <div class="kpi"><div class="kpi-label">Retenção Mês 6</div><div class="kpi-val red">14%</div><div class="kpi-sub">Meta Série A+: 32,5%</div></div>
  </div>

  <div class="grid2">
    <div class="card bordered-red">
      <div class="section-label">Caminho A — Win-back Reativo</div>
      <p style="margin:12px 0;font-size:14px;color:var(--gray)">MVP em 4 semanas. Recupera 15-25% dos cancelamentos no ponto de saída.</p>
      <div style="font-size:13px;line-height:2">
        <div>✅ Dados no PostgreSQL — disponíveis agora</div>
        <div>✅ ~R$ 500k receita preservada (26 semanas)</div>
        <div>❌ Cego para early churn (77% dos cancelamentos!)</div>
        <div>❌ Inferior ao Strava competitivamente</div>
      </div>
    </div>
    <div class="card bordered-teal">
      <div class="section-label">Caminho B — Engagement Forecasting</div>
      <p style="margin:12px 0;font-size:14px;color:var(--gray)">Modelo preditivo contínuo. Previne churn antes da decisão de cancelar.</p>
      <div style="font-size:13px;line-height:2">
        <div>✅ Pode atingir meta de 6,0% de churn</div>
        <div>✅ Paridade com Strava Premium Brasil (Q1/2026)</div>
        <div>❌ 6 semanas de infra antes do MVP</div>
        <div>❌ Risco de acordar sleeping dogs</div>
      </div>
    </div>
  </div>

  <div class="card" style="background:var(--navy);color:#fff;margin-bottom:28px;padding:32px">
    <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,.4);margin-bottom:12px">Recomendação do Grupo 02</div>
    <h2 style="font-size:28px;margin-bottom:12px">Plano 2 Balanceado: Híbrido A + B</h2>
    <p style="color:rgba(255,255,255,.6);font-size:15px;line-height:1.7">Semanas 1-4: Win-back reativo (MVP imediato, impacto financeiro antes do Board). Semanas 5-10: Pipeline BigQuery + treinamento de modelo preditivo. Semana 11+: Engagement Forecasting em produção.</p>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:20px">
      <div style="background:rgba(255,255,255,.08);padding:16px;border-radius:4px;text-align:center">
        <div style="font-size:24px;font-weight:700;color:var(--red)">2,9×</div><div style="font-size:11px;color:rgba(255,255,255,.5)">ROI 12 meses</div></div>
      <div style="background:rgba(255,255,255,.08);padding:16px;border-radius:4px;text-align:center">
        <div style="font-size:24px;font-weight:700;color:#22c55e">3,1×</div><div style="font-size:11px;color:rgba(255,255,255,.5)">LTV/CAC projetado</div></div>
      <div style="background:rgba(255,255,255,.08);padding:16px;border-radius:4px;text-align:center">
        <div style="font-size:24px;font-weight:700;color:#f59e0b">~7%</div><div style="font-size:11px;color:rgba(255,255,255,.5)">Churn projetado 12m</div></div>
    </div>
  </div>

  <div class="grid3">
    <div class="card bordered-red">
      <strong style="font-size:13px">Early Churn (77% dos cancelamentos)</strong>
      <p style="font-size:13px;color:var(--gray);margin-top:8px;line-height:1.6">Usuários com ≤ 1 mês saem via inércia/chargeback — invisíveis para qualquer sistema acionado por clique.</p></div>
    <div class="card bordered-red">
      <strong style="font-size:13px">Sleeping Dogs (ARR R$ 227k)</strong>
      <p style="font-size:13px;color:var(--gray);margin-top:8px;line-height:1.6">6+ meses, &lt; 0,5 sessão/sem, ainda ativos. Intervenção pode despertar cancelamento. Não acionar.</p></div>
    <div class="card bordered-teal">
      <strong style="font-size:13px">Alavanca Social (churn ÷2)</strong>
      <p style="font-size:13px;color:var(--gray);margin-top:8px;line-height:1.6">Desafios em grupo: 17% vs 33%. Indicação de amigo: 15,8% vs 31,3%. Ação mais barata e eficaz.</p></div>
  </div>
</div>

<!-- ═══════════ ABA 4: API ═══════════ -->
<div id="pane-api" class="pane">
  <div class="hero">
    <div class="hero-label">Documentação da API</div>
    <h1>Pipeline de <em>Inferência</em><br>em produção.</h1>
    <p>FastAPI + Gradient Boosting serializado (joblib). SHAP local por predição. Documentação interativa em /docs.</p>
  </div>
  <div class="grid2">
    <div class="card">
      <div class="section-label">Endpoints Disponíveis</div>
      <div style="margin-top:12px;font-family:monospace;font-size:13px">
        <div style="padding:10px 0;border-bottom:1px solid #f3f4f6;display:flex;gap:12px"><span style="background:#22c55e;color:#fff;padding:2px 8px;border-radius:3px;font-size:11px">GET</span><span>/health</span><span style="color:var(--gray);margin-left:auto">Health check</span></div>
        <div style="padding:10px 0;border-bottom:1px solid #f3f4f6;display:flex;gap:12px"><span style="background:#22c55e;color:#fff;padding:2px 8px;border-radius:3px;font-size:11px">GET</span><span>/modelo/info</span><span style="color:var(--gray);margin-left:auto">Metadados</span></div>
        <div style="padding:10px 0;border-bottom:1px solid #f3f4f6;display:flex;gap:12px"><span style="background:#3b82f6;color:#fff;padding:2px 8px;border-radius:3px;font-size:11px">POST</span><span>/predict</span><span style="color:var(--gray);margin-left:auto">Predição individual</span></div>
        <div style="padding:10px 0;display:flex;gap:12px"><span style="background:#3b82f6;color:#fff;padding:2px 8px;border-radius:3px;font-size:11px">POST</span><span>/predict/batch</span><span style="color:var(--gray);margin-left:auto">Lote CSV</span></div>
      </div>
    </div>
    <div class="card">
      <div class="section-label">Exemplo de Resposta POST /predict</div>
      <pre style="font-size:11px;background:#f9fafb;padding:12px;border-radius:4px;overflow:auto;margin-top:12px;line-height:1.6">{
  "probabilidade_cancelamento": 0.9982,
  "nivel_risco": "Crítico",
  "cluster": "C0",
  "cluster_nome": "Usuário Invisível",
  "recomendacoes": ["URGENTE: ..."],
  "shap_local": [
    {"label": "Queda de Frequência",
     "shap": 4.8013,
     "direcao": "aumenta_risco"}
  ],
  "explicacao_linguagem_natural": "..."
}</pre>
    </div>
  </div>
  <div class="card">
    <div class="section-label" style="margin-bottom:12px">Artefatos Serializados (joblib)</div>
    <div class="grid3" style="margin-top:12px">
      <div style="padding:12px;background:#f9fafb;border-radius:4px;font-family:monospace;font-size:12px">
        <div style="color:var(--teal);font-weight:700">vitaliza_modelo.joblib</div>
        <div style="color:var(--gray);margin-top:4px">GradientBoostingClassifier treinado</div></div>
      <div style="padding:12px;background:#f9fafb;border-radius:4px;font-family:monospace;font-size:12px">
        <div style="color:var(--teal);font-weight:700">vitaliza_scaler.joblib</div>
        <div style="color:var(--gray);margin-top:4px">StandardScaler (fit no treino)</div></div>
      <div style="padding:12px;background:#f9fafb;border-radius:4px;font-family:monospace;font-size:12px">
        <div style="color:var(--teal);font-weight:700">vitaliza_explainer.joblib</div>
        <div style="color:var(--gray);margin-top:4px">SHAP TreeExplainer</div></div>
    </div>
    <div style="margin-top:16px">
      <a href="/docs" target="_blank" style="display:inline-block;padding:10px 24px;background:var(--teal);color:#fff;text-decoration:none;font-size:13px;font-weight:600;border-radius:3px;margin-right:12px">Abrir Swagger UI →</a>
      <a href="/redoc" target="_blank" style="display:inline-block;padding:10px 24px;background:var(--navy);color:#fff;text-decoration:none;font-size:13px;font-weight:600;border-radius:3px">Abrir ReDoc →</a>
    </div>
  </div>
</div>

<script>
const SHAP_GLOBAL = """ + SHAP_GLOBAL_JSON + """;
const META = """ + META_JSON + """;

function showTab(id) {
  document.querySelectorAll('.pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('pane-' + id).classList.add('active');
  event.target.classList.add('active');
  if (id === 'modelo') renderModelo();
}

// ── Status da API ───────────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const r = await fetch('/health', {signal: AbortSignal.timeout(3000)});
    const d = await r.json();
    document.getElementById('status-dot').className = 'status-dot online';
    document.getElementById('status-text').textContent = `API online · AUC ${d.auc.toFixed(4)} · Modelo carregado`;
  } catch {
    document.getElementById('status-dot').className = 'status-dot offline';
    document.getElementById('status-text').textContent = 'API offline (usando fallback)';
  }
}
checkStatus();

// ── Predição ────────────────────────────────────────────────────────────────
async function predict() {
  const btn = document.getElementById('btn-predict');
  btn.disabled = true; btn.textContent = 'Calculando...';
  const payload = {
    genero: 1,
    proximo_unidade: +document.getElementById('proximo_unidade').value,
    plano_duplo: +document.getElementById('plano_duplo').value,
    indicado_amigo: +document.getElementById('indicado_amigo').value,
    telefone_cadastrado: +document.getElementById('telefone_cadastrado').value,
    desafios_grupo: +document.getElementById('desafios_grupo').value,
    idade: +document.getElementById('idade').value,
    gasto_adicional: +document.getElementById('gasto_adicional').value,
    meses_renovacao: +document.getElementById('meses_renovacao').value,
    tempo_assinatura: +document.getElementById('tempo_assinatura').value,
    freq_historica: +document.getElementById('freq_historica').value,
    freq_mes_atual: +document.getElementById('freq_mes_atual').value,
    duracao_contrato: +document.getElementById('duracao_contrato').value,
  };
  try {
    const res = await fetch('/predict', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const d = await res.json();
    renderResult(d);
  } catch(e) {
    alert('Erro ao chamar API: ' + e.message);
  } finally {
    btn.disabled = false; btn.textContent = 'Calcular Risco de Cancelamento';
  }
}

function riskClass(nivel) {
  return {Crítico:'red', Alto:'orange', Moderado:'yellow', Baixo:'green'}[nivel] || 'green';
}

function renderResult(d) {
  const box = document.getElementById('result-box');
  box.style.display = 'block';
  const pct = Math.round(d.probabilidade_cancelamento * 100);
  document.getElementById('res-prob').textContent = pct + '%';
  document.getElementById('res-prob').style.color = d.probabilidade_cancelamento > 0.65 ? 'var(--red)' : d.probabilidade_cancelamento > 0.35 ? '#e67e22' : 'var(--teal)';
  const badge = document.getElementById('res-badge');
  badge.textContent = d.nivel_risco; badge.className = 'badge ' + riskClass(d.nivel_risco);
  document.getElementById('res-cluster').textContent = d.cluster + ' — ' + d.cluster_nome;
  const bar = document.getElementById('res-bar');
  bar.style.width = pct + '%';
  bar.style.background = d.probabilidade_cancelamento > 0.65 ? 'var(--red)' : d.probabilidade_cancelamento > 0.35 ? '#e67e22' : 'var(--teal)';

  // SHAP
  const shapEl = document.getElementById('shap-container');
  const maxShap = Math.max(...d.shap_local.slice(0,8).map(s => Math.abs(s.shap)));
  shapEl.innerHTML = d.shap_local.slice(0, 8).map(s => {
    const pct = Math.round(Math.abs(s.shap) / maxShap * 100);
    const cor = s.direcao === 'aumenta_risco' ? 'shap-fill-pos' : 'shap-fill-neg';
    return `<div class="shap-row">
      <div class="shap-label">${s.label}</div>
      <div class="shap-track"><div class="${cor}" style="width:${pct}%"></div></div>
      <div class="shap-val" style="color:${s.direcao==='aumenta_risco'?'var(--red)':'var(--teal)'}">${s.shap>0?'+':''}${s.shap.toFixed(3)}</div>
    </div>`;
  }).join('');

  // Recomendações
  const recsEl = document.getElementById('recs-container');
  recsEl.innerHTML = d.recomendacoes.map(r => `<div class="rec-item"><div class="rec-dot"></div><div>${r}</div></div>`).join('');

  document.getElementById('res-explicacao').textContent = d.explicacao_linguagem_natural;
  document.getElementById('res-meta').textContent = `Algoritmo: ${d.metadata_modelo.algoritmo} · AUC ${d.metadata_modelo.auc_roc} · Treino: ${d.metadata_modelo.data_treino}`;
  box.scrollIntoView({behavior:'smooth', block:'start'});
}

// ── Aba Modelo ──────────────────────────────────────────────────────────────
let chartRendered = false;
function renderModelo() {
  if (chartRendered) return;
  chartRendered = true;
  document.getElementById('m-auc').textContent   = META.auc;
  document.getElementById('m-ap').textContent    = META.ap;
  document.getElementById('m-gap').textContent   = META.gap;
  document.getElementById('m-brier').textContent = META.brier;
  document.getElementById('cm-tn').textContent   = META.cm.tn;
  document.getElementById('cm-fp').textContent   = META.cm.fp;
  document.getElementById('cm-fn').textContent   = META.cm.fn;
  document.getElementById('cm-tp').textContent   = META.cm.tp;

  const prec_cancel = (META.cm.tp/(META.cm.tp+META.cm.fp)).toFixed(3);
  const rec_cancel  = (META.cm.tp/(META.cm.tp+META.cm.fn)).toFixed(3);
  const f1          = (2*prec_cancel*rec_cancel/(+prec_cancel+ +rec_cancel)).toFixed(3);
  document.getElementById('metrics-detail').innerHTML = [
    ['CV AUC Médio', META.cv_auc, 'good'], ['CV AUC Desvio', '± '+META.cv_std, ''],
    ['Gap Treino-Val.', META.gap, META.gap<0.05?'good':'warn'],
    ['Precisão (Cancelado)', prec_cancel, 'good'], ['Recall (Cancelado)', rec_cancel, 'good'],
    ['F1 (Cancelado)', f1, 'good'], ['Registros Treino', META.n_treino, ''],
    ['Algoritmo', META.algoritmo, ''], ['Data Treino', META.data, ''],
  ].map(([n,v,c]) => `<div class="metric-row"><span class="metric-name">${n}</span><span class="metric-val ${c}">${v}</span></div>`).join('');

  // SHAP global chart
  const entries = Object.entries(SHAP_GLOBAL).sort((a,b) => b[1]-a[1]).slice(0, 12);
  const labels  = entries.map(e => e[0].length > 30 ? e[0].slice(0,30)+'...' : e[0]);
  const values  = entries.map(e => e[1]);
  new Chart(document.getElementById('shap-chart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'SHAP — Importância Média Absoluta',
        data: values,
        backgroundColor: values.map((_,i) => i === 0 ? 'rgba(192,57,43,0.85)' : i < 4 ? 'rgba(192,57,43,0.6)' : 'rgba(13,110,110,0.5)'),
        borderRadius: 3,
      }]
    },
    options: {
      indexAxis:'y', responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false}, tooltip:{callbacks:{label: c => ` ${c.parsed.x.toFixed(4)}`}}},
      scales:{x:{grid:{color:'rgba(0,0,0,0.05)'}}, y:{ticks:{font:{size:12}}}},
    }
  });
}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def frontend():
    try:
        return _load_html()
    except FileNotFoundError:
        return _HTML_LEGACY

if __name__ == "__main__":
    auc = metadata["auc_roc_teste"]
    print("\n" + "=" * 60)
    print("VITALIZA — APLICACAO COMPLETA")
    print(f"Modelo: {metadata['algoritmo']} | AUC-ROC: {auc:.4f}")
    print("\nAcesse: http://localhost:8000")
    print("Docs:   http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
