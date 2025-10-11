<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>TextMorph — Milestone 2</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 40px;
      line-height: 1.6;
      background-color: #ffffff;
      color: #0b1b2b;
    }
    h1 {
      color: #0b3d91;
    }
    h2 {
      color: #0b2b66;
      margin-top: 1.8em;
    }
    h3 {
      margin-top: 1.2em;
    }
    code {
      background-color: #f3f4f6;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: monospace;
    }
    pre {
      background-color: #0f1724;
      color: #e6edf3;
      padding: 12px;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 13px;
    }
    ul, ol {
      margin-left: 20px;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      margin-top: 10px;
    }
    table, th, td {
      border: 1px solid #ddd;
    }
    th, td {
      padding: 10px;
      text-align: left;
    }
    .small {
      font-size: 13px;
      color: #666;
    }
  </style>
</head>
<body>

<h1>TextMorph — Milestone 2</h1>
<p><b>Advanced text summarization, evaluation, and interactive UIs (Colab / Jupyter friendly).</b></p>

<h2>Badges</h2>
<ul>
  <li>Colab-friendly</li>
  <li>Abstractive + Extractive</li>
  <li>ROUGE • Readability • Semantic Similarity</li>
</ul>

<hr>



<hr>

<h2 id="overview">Overview</h2>
<ul>
  <li><b>Abstractive:</b> TinyLlama, Phi, Gemma, BART</li>
  <li><b>Extractive:</b> TextRank using Sentence Transformers</li>
  <li><b>Evaluation:</b> ROUGE, cosine similarity, readability (Flesch-Kincaid, Gunning-Fog)</li>
  <li><b>UI:</b> Interactive via ipywidgets for notebook-based testing</li>
</ul>

<h2 id="requirements">Requirements</h2>
<p>Python 3.8+ (3.9+ recommended). Ideal for Google Colab with GPU.</p>

<pre><code>pip install transformers sentence-transformers rouge-score textstat nltk ipywidgets torch torchvision accelerate networkx matplotlib</code></pre>

<h2 id="quickstart">Quick Start (Colab)</h2>
<ol>
  <li>Change runtime to GPU in Colab</li>
  <li>Install dependencies using the above command</li>
  <li>Set <code>HF_TOKEN</code> if using gated models</li>
  <li>Run all cells: imports → models → summarizers → UI</li>
</ol>

<h2 id="project-structure">Project Structure</h2>

<table>
<tr><th>File</th><th>Description</th></tr>
<tr><td><code>notebook.ipynb</code></td><td>Main notebook with pipeline + UI</td></tr>
<tr><td><code>models.py</code></td><td>Model loading functions</td></tr>
<tr><td><code>textrank.py</code></td><td>Extractive summarizer via embeddings + PageRank</td></tr>
<tr><td><code>eval.py</code></td><td>Evaluation (ROUGE, sim, readability)</td></tr>
<tr><td><code>ui.py</code></td><td>ipywidgets UIs</td></tr>
<tr><td><code>samples/</code></td><td>Sample inputs + references</td></tr>
<tr><td><code>requirements.txt</code></td><td>Exact versions</td></tr>
</table>

<h2 id="usage">Usage & Example Commands</h2>

<h3>Safe text trimming</h3>
<pre><code>from nltk import sent_tokenize

def safe_trim(text, max_len=4096):
    if len(text) &lt;= max_len:
        return text
    sents = sent_tokenize(text)
    out = ''
    for s in sents:
        if len(out) + len(s) + 1 &gt; max_len:
            break
        out += ' ' + s
    return out.strip()</code></pre>

<h3>Load BART model</h3>
<pre><code>from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to("cuda")</code></pre>

<h3>TextRank Extractive</h3>
<pre><code>from sentence_transformers import SentenceTransformer
import numpy as np, networkx as nx

model = SentenceTransformer("all-MiniLM-L6-v2")

def textrank_extract(text, top_k=3):
    sents = sent_tokenize(text)
    embeddings = model.encode(sents, convert_to_numpy=True)
    sim = np.inner(embeddings, embeddings)
    graph = nx.from_numpy_array(sim)
    scores = nx.pagerank_numpy(graph)
    ranked = sorted(((scores[i], s) for i, s in enumerate(sents)), reverse=True)
    return ' '.join(s for _, s in ranked[:top_k])</code></pre>

<h3>ROUGE Scoring</h3>
<pre><code>from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

def compute_rouge(ref, hyp):
    s = scorer.score(ref, hyp)
    return {k: round(v.fmeasure, 4) for k, v in s.items()}</code></pre>

<h2 id="models">Models & Tokens</h2>
<ul>
  <li>HF-gated models require accepting licenses and setting <code>HF_TOKEN</code></li>
  <li>Large models may cause OOM → use <code>device_map="auto"</code> or smaller models</li>
</ul>

<h2 id="evaluation">Evaluation & Metrics</h2>
<ul>
  <li>ROUGE-1, ROUGE-2, ROUGE-L</li>
  <li>Cosine similarity (MiniLM embeddings)</li>
  <li>Readability: Flesch-Kincaid, Gunning-Fog</li>
</ul>

<h2 id="uis">Interactive UIs</h2>
<p>Notebook includes:</p>
<ol>
  <li><b>All-models UI</b> — runs all loaded models</li>
  <li><b>Select-models UI</b> — user chooses models</li>
</ol>

<pre><code>all_models_ui(models_available)
select_models_ui(models_available)</code></pre>

<h2 id="troubleshooting">Troubleshooting</h2>
<ul>
  <li><b>OOM:</b> Use fewer or smaller models</li>
  <li><b>Token limit:</b> Use <code>safe_trim()</code></li>
  <li><b>Gated models:</b> Accept license and add <code>HF_TOKEN</code></li>
</ul>

<h2 id="saving">Saving & Reproducibility</h2>
<pre><code>import json
df.to_csv("results.csv", index=False)
with open("summaries.json", "w") as f:
    json.dump(results, f)</code></pre>

<h2 id="license">License & Contact</h2>
<p>This project is licensed under the <b>MIT License</b>.</p>

</body>
</html>
