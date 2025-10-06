<!DOCTYPE html>
<html>
<head>
<title>Text Summarization & Paraphrasing Project</title>
</head>
<body>

<h1>🧠 Text Summarization, Paraphrasing & Similarity Analysis</h1>

<p>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python"/></a>
<a href="https://huggingface.co/"><img src="https://img.shields.io/badge/Hugging%20Face-Transformers-orange.svg" alt="HuggingFace"/></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/></a>
</p>

<hr>

<h2>📋 Table of Contents</h2>
<ul>
<li><a href="#features">✨ Features</a></li>
<li><a href="#workflow">⚙️ Workflow</a></li>
<li><a href="#tech-stack">💻 Tech Stack</a></li>
<li><a href="#installation">🚀 Installation</a></li>
<li><a href="#usage">🧠 Usage</a></li>
<li><a href="#code-structure">🗂️ Code Structure</a></li>
<li><a href="#models-used">🤖 Models Used</a></li>
<li><a href="#output-visualization">📊 Output & Visualization</a></li>
<li><a href="#future-enhancements">🔮 Future Enhancements</a></li>
<li><a href="#license">📜 License</a></li>
</ul>

<hr>

<h2 id="features">✨ Features</h2>
<ul>
<li>✅ Summarization using <b>T5</b>, <b>BART</b>, and <b>Pegasus</b></li>
<li>✅ Paraphrasing using <b>T5</b>, <b>BART</b>, and <b>Pegasus</b></li>
<li>✅ Semantic similarity scoring using <b>SentenceTransformer (MiniLM)</b></li>
<li>✅ Modular, function-based design for easy customization</li>
<li>✅ Visualization of similarity scores using <b>bar plots</b></li>
</ul>

<hr>

<h2 id="workflow">⚙️ Workflow</h2>
<ol>
<li>Load Texts – Read two text files (<code>Sample1.txt</code> and <code>Sample2.txt</code>).</li>
<li>Summarization – Generate summaries using three models.</li>
<li>Paraphrasing – Generate paraphrases using three models.</li>
<li>Similarity Scoring – Compute cosine similarity between original and generated texts.</li>
<li>Visualization – Plot bar charts to compare model similarity scores.</li>
</ol>

<hr>

<h2 id="tech-stack">💻 Tech Stack</h2>

<table>
<tr><th>Component</th><th>Technology</th></tr>
<tr><td>Language</td><td>Python 3</td></tr>
<tr><td>NLP</td><td>Hugging Face Transformers</td></tr>
<tr><td>Embeddings</td><td>Sentence-Transformers</td></tr>
<tr><td>Visualization</td><td>Matplotlib</td></tr>
<tr><td>Preprocessing</td><td>NLTK</td></tr>
<tr><td>Environment</td><td>Jupyter Notebook / Google Colab / VS Code</td></tr>
</table>

<hr>

<h2 id="installation">🚀 Installation</h2>

<pre><code># Clone repository
git clone https://github.com/yourusername/text-summarization-paraphrasing.git
cd text-summarization-paraphrasing

# Install dependencies
pip install transformers sentence-transformers nltk matplotlib

# Download NLTK tokenizer
import nltk
nltk.download('punkt')
</code></pre>

<hr>

<h2 id="usage">🧠 Usage</h2>

<pre><code># Place text files in the project folder:
Sample1.txt
Sample2.txt

# Run the main script
python main.py
</code></pre>

<p>The script will:</p>
<ul>
<li>Print summaries and paraphrases for each text</li>
<li>Display similarity scores for each model</li>
<li>Generate bar charts comparing similarity scores</li>
</ul>

<hr>

<h2 id="code-structure">🗂️ Code Structure</h2>

<pre><code>📂 Project Root
├── main.py                 # Main script with functions and workflow
├── Sample1.txt             # First input text file
├── Sample2.txt             # Second input text file
├── requirements.txt        # Optional dependencies list
└── README.md               # Project documentation
</code></pre>

<h3>Function Breakdown</h3>
<table>
<tr><th>Function</th><th>Purpose</th></tr>
<tr><td>load_texts()</td><td>Load and return content from two text files</td></tr>
<tr><td>summarize_with_model()</td><td>Summarize text using a Transformer model</td></tr>
<tr><td>paraphrase_with_model()</td><td>Generate paraphrased text</td></tr>
<tr><td>similarity_score()</td><td>Compute cosine similarity between original and generated text</td></tr>
<tr><td>main()</td><td>Run workflow and visualize results</td></tr>
</table>

<hr>

<h2 id="models-used">🤖 Models Used</h2>

<h3>Summarization Models</h3>
<ul>
<li>t5-base</li>
<li>facebook/bart-base</li>
<li>google/pegasus-xsum</li>
</ul>

<h3>Paraphrasing Models</h3>
<ul>
<li>ramsrigouthamg/t5_paraphraser</li>
<li>eugenesiow/bart-paraphrase</li>
<li>Vamsi/T5_Paraphrase_Paws</li>
</ul>

<h3>Similarity Model</h3>
<ul>
<li>paraphrase-MiniLM-L6-v2 → lightweight, fast sentence embeddings</li>
</ul>

<hr>

<h2 id="output-visualization">📊 Output & Visualization</h2>
<p>Console output shows summaries, paraphrases, and similarity scores. Bar charts visualize model similarity (0–1) for each text.</p>

<hr>

<h2 id="future-enhancements">🔮 Future Enhancements</h2>
<ul>
<li>Fine-tune models on custom dataset</li>
<li>Add ROUGE/BLEU metrics for evaluation</li>
<li>Support multiple text files automatically</li>
<li>Build a web-based interface for model comparison</li>
<li>Enable GPU acceleration for faster inference</li>
</ul>

<hr>

<h2 id="license">📜 License</h2>
<p>This project is licensed under the <b>MIT License</b> — free to use, modify, and distribute.</p>

</body>
</html>
