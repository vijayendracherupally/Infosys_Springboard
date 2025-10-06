<!DOCTYPE html>
<html>
<head>
</head>
<body>

<h1>Text Summarization, Paraphrasing & Similarity Analysis</h1>



<h2 id="features"> Features</h2>
<ul>
<li>Summarization using <b>T5</b>, <b>BART</b>, and <b>Pegasus</b></li>
<li>Paraphrasing using <b>T5</b>, <b>BART</b>, and <b>Pegasus</b></li>
<li>Semantic similarity scoring using <b>SentenceTransformer (MiniLM)</b></li>
<li>Visualization of similarity scores using <b>bar plots</b></li>
</ul>

<hr>

<h2 id="workflow"> Workflow</h2>
<ol>
<li>Load Texts – Read two text files (<code>Sample1.txt</code> and <code>Sample2.txt</code>).</li>
<li>Summarization – Generate summaries using three models.</li>
<li>Paraphrasing – Generate paraphrases using three models.</li>
<li>Similarity Scoring – Compute cosine similarity between original and generated texts.</li>
<li>Visualization – Plot bar charts to compare model similarity scores.</li>
</ol>




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

<h2 id="models-used"> Models Used</h2>

<h3>Summarization Models</h3>
<ul>
<li>t5-base</li>
<li>facebook/bart-base</li>
<li>google/pegasus-xsum</li>
</ul>

<h3>Paraphrasing Models</h3>
<ul>
<li>t5_paraphraser</li>
<li>bart-paraphrase</li>
<li>Pegasus</li>
</ul>

<h3>Similarity Model</h3>
<ul>
<li>paraphrase-MiniLM-L6-v2 → lightweight, fast sentence embeddings</li>
</ul>

<hr>

<h2 id="output-visualization"> Output & Visualization</h2>
<p>Console output shows summaries, paraphrases, and similarity scores. Bar charts visualize model similarity (0–1) for each text.</p>



<h2 id="license"> License</h2>
<p>This project is licensed under the <b>MIT License</b> — free to use, modify, and distribute.</p>

</body>
</html>
