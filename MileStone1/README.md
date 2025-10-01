                                         ...Sample1.txt...
The rapid advancements in artificial intelligence have sparked both excitement and concern. On one hand, AI promises to revolutionize industries, from healthcare to finance, by improving efficiency, accuracy, and productivity. On the other hand, there are growing fears about the ethical implications, job displacement, and the potential misuse of AI technology. As society grapples with these challenges, it's clear that while AI has vast potential, its development must be approached with caution and responsibility.

T5 Summary:
the rapid advancements in artificial intelligence have sparked both excitement and concern . there are growing fears about the ethical implications, job displacement, and the potential misuse of AI technology . as society grapples with these challenges, it's clear that while AI has vast potential, its development must be approached with caution .

T5 Paraphrase:
What are the challenges facing industry in view of the rapid advancement of Artificial Intelligence?

BART Summary:
The rapid advancements in artificial intelligence have sparked both excitement and concern. On one hand, AI promises to revolutionize industries, from healthcare to finance, by improving efficiency, accuracy, and productivity. On the other hand, there are growing fears about the ethical implications, job displacement, and the potential misuse of AI technology. As society grapples with these challenges, it's clear that while AI has vast potential, its development must be approached with caution and responsibility.

BART Paraphrase:
Paraphrase: The rapid advancements in artificial intelligence have sparked both excitement and concern. On one hand, AI promises to revolutionize industries, from healthcare to finance, by improving efficiency, accuracy and productivity. On the other hand, there are growing fears about the ethical implications, job displacement and the potential misuse of AI technology.

Pegasus Summary:
In our series of letters from African journalists, film-maker, and columnist Ahmed Rashid looks at some of the key issues surrounding artificial intelligence and its potential impact on the world around us, as well as some of the best examples of how the technology is being used.

Pegasus Paraphrase:
The rapid advances in artificial intelligence have sparked both excitement and concern: On one hand, AI promises to revolutionize industries from healthcare to finance by improving efficiency, accuracy, and productivity; on the other hand, there are growing fears about the ethical implications, the loss of jobs, and the potential misuse of AI technology. As society grapples with these challenges, it is clear that while AI has enormous potential, its development must be approached with caution and responsibility .

<img width="1002" height="572" alt="image" src="https://github.com/user-attachments/assets/698bd715-16f0-4023-844a-ee8d71d844a2" />

<img width="1013" height="592" alt="image" src="https://github.com/user-attachments/assets/fee7b7ca-af0e-4b00-95cd-cfa1f0099759" />


->Summarization models were used via Hugging Face's pipeline() interface.
->Paraphrasing models were used by manually preparing inputs (e.g., "paraphrase: <text>") and using AutoTokenizer + AutoModelForSeq2SeqLM.
->Similarity Scoring Embeddings were generated using the sentence-transformers model (paraphrase-MiniLM-L6-v2).

Though BART summary and paraphrase got higher similarity score T5 model dominates in understanding of context and summarizing much better than other models



