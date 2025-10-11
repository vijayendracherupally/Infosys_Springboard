<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>TextMorph — Milestone 2 — README</title>

  <!-- Basic styles to look good in browser / VS Code preview -->
  <style>
    :root{
      --bg:#ffffff; --muted:#6b7280; --primary:#0b3d91; --card:#f9fafb;
      --code-bg:#0f1724; --code-fg:#e6edf3; --accent:#ef4444;
    }
    html,body{height:100%; margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; color:#0b1b2b; background:var(--bg); line-height:1.55;}
    .container{max-width:980px; margin:32px auto; padding:28px; background:linear-gradient(180deg,#fff, #fbfdff); border-radius:12px; box-shadow:0 6px 30px rgba(2,6,23,0.06);}
    h1{margin:0 0 8px; font-size:28px; color:var(--primary);}
    .sub{color:var(--muted); margin-bottom:18px;}
    .badges{margin:8px 0 18px;}
    .badge{display:inline-block;background:#eef2ff;color:#3730a3;padding:6px 10px;border-radius:999px;margin-right:8px;font-weight:600;font-size:13px;}
    h2{color:#0b2b66;margin-top:26px;margin-bottom:10px;}
    p{margin:8px 0;}
    pre{background:var(--code-bg); color:var(--code-fg); padding:12px; border-radius:8px; overflow:auto; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace; font-size:13px;}
    code{background:#f3f4f6;padding:2px 6px;border-radius:6px;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace;}
    ul{margin:8px 0 8px 18px;}
    table{width:100%; border-collapse:collapse; margin-top:8px;}
    table th, table td{border:1px solid #e6e9ef; padding:10px; text-align:left; vertical-align:top;}
    .note{background:#fffbeb;border-left:4px solid #f59e0b;padding:12px;border-radius:6px;color:#92400e;}
    .kbd{background:#111827;color:#fff;padding:4px 8px;border-radius:6px;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace;}
    .footer{margin-top:26px;color:var(--muted); font-size:14px;}
    a.btn{display:inline-block;padding:10px 14px;background:var(--primary);color:white;border-radius:8px;text-decoration:none;margin-top:8px;}
    .toc a{color:var(--primary); text-decoration:none;}
    .small{font-size:13px;color:var(--muted);}
  </style>
</head>
<body>
  <div class="container">
    <h1>TextMorph — Milestone 2</h1>
    <div class="sub">End-to-end README for Milestone 2: Advanced text summarization, evaluation, and interactive UIs (Colab / Jupyter friendly).</div>

    <div class="badges">
      <span class="badge">Colab-friendly</span>
      <span class="badge">Abstractive + Extractive</span>
      <span class="badge">ROUGE • Readability</span>
    </div>

    <section>
      <h2>Table of contents</h2>
      <div class="toc small">
        <ol>
          <li><a href="#overview">Overview</a></li>
          <li><a href="#requirements">Requirements</a></li>
          <li><a href="#quickstart">Quick start (Colab)</a></li>
          <li><a href="#project-structure">Project structure</a></li>
          <li><a href="#usage">Usage & Example Commands</a></li>
          <li><a href="#models">Models & Tokens</a></li>
          <li><a href="#evaluation">Evaluation & Metrics</a></li>
          <li><a href="#uis">Interactive UIs</a></li>
          <li><a href="#troubleshooting">Troubleshooting</a></li>
          <li><a href="#saving">Saving & Reproducibility</a></li>
          <li><a href="#license">License & Contact</a></li>
        </ol>
      </div>
    </section>

    <section id="overview">
      <h2>Overview</h2>
      <p>This milestone implements a research/engineering pipeline for multi-model text summarization:</p>
      <ul>
        <li><strong>Abstractive</strong> models (causal & seq2seq): TinyLlama, Phi, Gemma, BART (examples)</li>
        <li><strong>Extractive</strong> summarization via TextRank built on sentence embeddings (sentence-transformers)</li>
        <li>Evaluation: ROUGE-1/2/L, semantic similarity (embedding cosine), readability metrics (Flesch-Kincaid, Gunning-Fog)</li>
        <li>Interactive UIs using <code>ipywidgets</code> to test models in a notebook environment</li>
        <li>Batch evaluation + plots + saving results</li>
      </ul>
    </section>

    <section id="requirements">
      <h2>Requirements</h2>
      <p>Recommended: Google Colab with a GPU runtime (for heavier models). Python 3.8+ (3.9+ recommended).</p>
      <p>Key packages:</p>
      <pre><code>transformers
sentence-transformers
rouge-score
textstat
nltk
ipywidgets
torch
accelerate
networkx
matplotlib
</code></pre>
      <p>Installation command (Colab / fresh venv):</p>
      <pre><code>pip install transformers sentence-transformers rouge-score textstat nltk ipywidgets torch torchvision accelerate networkx matplotlib</code></pre>
      <p class="small">If you plan to use some licensed/gated models (Gemma, Phi), you must set a Hugging Face token in the environment and accept model licenses on HF.</p>
    </section>

    <section id="quickstart">
      <h2>Quick start (Google Colab)</h2>
      <ol>
        <li>Open a Colab notebook: <kbd>Runtime → Change runtime type → GPU</kbd>.</li>
        <li>Run the install cell from <strong>Requirements</strong>.</li>
        <li>If using HF-gated models: set <code>HF_TOKEN</code> in environment variables (or in Colab secrets) and accept license on Hugging Face.</li>
        <li>Run notebook cells in order: imports → helper utils → model loaders → summarizers → UIs.</li>
        <li>Use the provided UIs to paste input text and run summarization with one or multiple models.</li>
      </ol>
    </section>

    <section id="project-structure">
      <h2>Project structure</h2>
      <p>Suggested repository layout (how README refers to files):</p>
      <table>
        <thead>
          <tr><th>Path / File</th><th>Purpose</th></tr>
        </thead>
        <tbody>
          <tr><td><code>notebook.ipynb</code></td><td>Main notebook with install cells, imports, helpers, model loading, summarizers, UI widgets, and evaluation cells.</td></tr>
          <tr><td><code>models.py</code></td><td>Optional: functions to load models and generate summaries (causal & seq2seq wrappers).</td></tr>
          <tr><td><code>textrank.py</code></td><td>Extractive summarizer implementation using sentence-transformers + PageRank.</td></tr>
          <tr><td><code>eval.py</code></td><td>ROUGE, semantic similarity, readability functions and wrappers for dataset evaluation.</td></tr>
          <tr><td><code>ui.py</code></td><td>ipywidgets UI components (All models and Select-models UIs).</td></tr>
          <tr><td><code>samples/</code></td><td>Sample texts and reference summaries for quick testing and evaluation.</td></tr>
          <tr><td><code>requirements.txt</code></td><td>Pin exact package versions used during development (recommended).</td></tr>
        </tbody>
      </table>
    </section>

    <section id="usage">
      <h2>Usage & Example Commands</h2>
      <h3>1) Basic helper & safe trim</h3>
      <pre><code>from nltk import sent_tokenize

def safe_trim(text: str, max_len: int = 4096) -> str:
    if len(text) <= max_len:
        return text
    sents = sent_tokenize(text)
    out = ''
    for s in sents:
        if len(out) + len(s) + 1 > max_len:
            break
        out += (' ' + s)
    return out.strip()
</code></pre>

      <h3>2) Load an example seq2seq model (BART)</h3>
      <pre><code>from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn').to('cuda')
</code></pre>

      <h3>3) TextRank extractive example</h3>
      <pre><code>from sentence_transformers import SentenceTransformer
import numpy as np, networkx as nx

EMB_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def textrank_extract(text, top_k=3):
    sents = sent_tokenize(text)
    if len(sents) <= top_k: 
        return ' '.join(sents)
    embeddings = EMB_MODEL.encode(sents, convert_to_numpy=True)
    sim = np.inner(embeddings, embeddings)
    graph = nx.from_numpy_array(sim)
    scores = nx.pagerank_numpy(graph)
    ranked = sorted(((scores[i], s) for i,s in enumerate(sents)), reverse=True)
    selected = [s for _, s in ranked[:top_k]]
    ordered = [s for s in sents if s in selected]
    return ' '.join(ordered)
</code></pre>

      <h3>4) ROUGE scoring example</h3>
      <pre><code>from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
def compute_rouge(ref, hyp):
    s = scorer.score(ref, hyp)
    return {k: round(v.fmeasure, 4) for k, v in s.items()}
</code></pre>
    </section>

    <section id="models">
      <h2>Models & Tokens</h2>
      <p><strong>Important notes:</strong></p>
      <ul>
        <li>Some models (Gemma, Phi) are gated/licensed and require you to accept terms on Hugging Face and use an <code>HF_TOKEN</code>.</li>
        <li>Large models require more VRAM: if you get OOM errors, use smaller models or offloading (Accelerate's device_map and offload_folder).</li>
        <li>Load models only when needed to avoid memory spikes.</li>
      </ul>
      <p class="small">Example to set HF token in a Colab cell (temporary):</p>
      <pre><code>import os
os.environ['HF_TOKEN'] = 'your_hf_token_here'</code></pre>
    </section>

    <section id="evaluation">
      <h2>Evaluation & Metrics</h2>
      <p>The evaluation pipeline provides:</p>
      <ul>
        <li><strong>ROUGE-1/2/L</strong> (F-measure)</li>
        <li><strong>Semantic similarity</strong> using <code>all-MiniLM-L6-v2</code> embeddings (cosine)</li>
        <li><strong>Readability</strong> (Flesch-Kincaid grade, Gunning-Fog, SMOG) via <code>textstat</code></li>
      </ul>
      <p>Example evaluation wrapper:</p>
      <pre><code>def evaluate_summary(reference, hypothesis):
    rouge = compute_rouge(reference, hypothesis)
    # semantic similarity:
    r_emb = EMB_MODEL.encode([reference], convert_to_numpy=True)
    h_emb = EMB_MODEL.encode([hypothesis], convert_to_numpy=True)
    sim = float(np.dot(r_emb, h_emb.T) / (np.linalg.norm(r_emb)*np.linalg.norm(h_emb)))
    read = {
        'flesch_kincaid': textstat.flesch_kincaid_grade(hypothesis),
        'gunning_fog': textstat.gunning_fog(hypothesis)
    }
    return {**rouge, 'semantic_sim': round(sim,4), **read}
</code></pre>
    </section>

    <section id="uis">
      <h2>Interactive UIs</h2>
      <p>Two ipywidgets-based UIs are included in the notebook:</p>
      <ol>
        <li><strong>All-models UI</strong> — runs all loaded models plus TextRank on the same input and displays outputs.</li>
        <li><strong>Select-models UI</strong> — user selects which models to run (checkboxes).</li>
      </ol>

      <p>Call these functions after model loading in the notebook:</p>
      <pre><code>models_available = list(MODELS.keys()) + ['textrank']
all_models_ui(models_available)      # shows full UI
select_models_ui(models_available)   # shows checkbox UI
</code></pre>
      <p class="small">Note: in Colab, you may need to enable widgets with <code>jupyter nbextension enable --py widgetsnbextension</code>.</p>
    </section>

    <section id="troubleshooting">
      <h2>Troubleshooting</h2>
      <ul>
        <li><strong>OOM / memory errors:</strong> load fewer models; use smaller models for debugging; set <code>device_map='auto'</code> and <code>offload_folder</code> with Accelerate; reduce generation <code>max_new_tokens</code>.</li>
        <li><strong>Tokenization truncation:</strong> use <code>safe_trim</code> to trim at sentence boundaries.</li>
        <li><strong>Gated models:</strong> accept HF model license and ensure <code>HF_TOKEN</code> has correct scope.</li>
      </ul>
    </section>

    <section id="saving">
      <h2>Saving & Reproducibility</h2>
      <p>Save evaluation results and summaries for reproducibility:</p>
      <pre><code>import json
df.to_csv('evaluation_results.csv', index=False)
summaries = {m: df[df['model']==m][['id','summary']].to_dict(orient='records') for m in models}
with open('summaries_by_model.json','w') as f:
    json.dump(summaries, f, indent=2)
</code></pre>
      <p class="small">Store exact package versions in <code>requirements.txt</code> (example: <code>transformers==4.40.0</code>).</p>
    </section>

    <section id="license">
      <h2>License & Contact</h2>
      <p>This repo's code and notebook cells are provided for research and educational use. When using third-party models please respect their licenses.</p>
      <p>For questions or help integrating these cells into a packaged project, contact: <strong>Vijayendra</strong> (or update with repository owner / project email).</p>
    </section>

    <div class="footer">
      <p>Last updated: <em><!-- You can update this date before committing -->October 11, 2025</em></p>
      <p class="small">Tip: GitHub shows rendered HTML files in the file view — commit this file as <code>README.html</code> to preview it directly in your repo. If you want GitHub's repository landing page to show this README by default, also provide a <code>README.md</code> (markdown) or configure GitHub Pages.</p>
    </div>
  </div>
</body>
</html>
