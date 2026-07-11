# Conversational Question Answering (CQA) System with BERT

A simplified Conversational Question Answering system built on top of a pre-trained BERT model, extended with a lightweight dialogue history mechanism to resolve pronouns and elliptical (word-dropping) follow-up questions.

## Overview

Standard extractive QA models like BERT answer each question in complete isolation, they have no memory of previous turns in a conversation. This becomes a problem the moment a follow-up question relies on context, such as:

- **Pronoun ambiguity**: *"What year did **she** begin her research?"*
- **Ellipsis (dropped words)**: *"In what year?"* (missing subject and verb entirely)

This project implements a rule-based dialogue history layer on top of BERT to resolve both cases before passing the question to the model, and compares system performance **with** and **without** this history mechanism across the same conversation.

## How It Works

The system is built around a `ConversationalQASystem` class with three core components:

1. **`get_answer()`** : Encodes the question and passage, runs BERT, and extracts the answer span using the model's predicted start/end positions (restricted to the passage only, never the question).
2. **`resolve_question()`** : Handles two forms of ambiguity:
   - If the question contains a pronoun (*she, her, he, they...*), it is replaced with the subject established in the previous turn.
   - If the question is very short with no pronoun at all, it is treated as elliptical and merged with the topic of the last resolved question.
3. **`update_subject()`** : After each turn, checks whether the question explicitly names a known entity, and updates the "current subject" of the conversation accordingly.

**Model used:** [`bert-large-uncased-whole-word-masking-finetuned-squad`](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad), an extractive QA model fine-tuned on SQuAD.

## Example Conversation

**Passage:**
> Marie Curie and her daughter Irène Joliot-Curie were both pioneering scientists. Marie was born in Warsaw in 1867 and began her research in Paris in 1891. Irène was born in Paris in 1897 and began her own research in 1918. Marie won her first Nobel Prize in Physics in 1903. Irène later won the Nobel Prize in Chemistry in 1935.

| Turn | Question | Without History | With History |
|------|----------|-----------------|---------------|
| 1 | Who was Irène Joliot-Curie? | pioneering scientists | pioneering scientists |
| 2 | What year did she begin her research? | 1891 ❌ (Marie's year) | 1918 ✅ (Irène's year) |
| 3 | What prize did she win? | Nobel Prize in Physics ❌ | Nobel Prize in Chemistry ✅ |
| 4 | In what year? | *(empty, no answer found)* | 1935 ✅ |

The passage deliberately mentions **two people** so that pronoun references are genuinely ambiguous, making the effect of dialogue history clearly measurable: **3 of the 4 turns** produce different, and correct answers only when history is used.

## Key Finding

Dialogue history isn't just about resolving pronouns. Real conversational follow-ups often drop words entirely (e.g., *"In what year?"*), and a working CQA system needs to handle both cases to stay coherent across turns.

## Limitations

This is a **rule-based, simplified** implementation, not a learned conversational architecture:

- Relies on manually written regular expressions to detect pronouns and short/elliptical questions.
- Entity tracking uses a fixed list of known names rather than general-purpose entity recognition.
- Designed and tuned around one specific passage/conversation, it is not guaranteed to generalize to arbitrary text (e.g., a random CoQA passage with different phrasing).

Purpose-built conversational QA architectures such as **BiDAF++** and **FlowQA** address this by learning to encode dialogue history directly through attention mechanisms across turns, rather than through hand-crafted text rewriting rules  allowing them to generalize far beyond a fixed rule set.

## Requirements

```bash
pip install transformers torch
```

## Usage

```python
from cqa_system import ConversationalQASystem

passage = """
Marie Curie and her daughter Irène Joliot-Curie were both pioneering scientists.
Marie was born in Warsaw in 1867 and began her research in Paris in 1891.
Irène was born in Paris in 1897 and began her own research in 1918.
Marie won her first Nobel Prize in Physics in 1903.
Irène later won the Nobel Prize in Chemistry in 1935.
"""

known_entities = ["Marie Curie", "Irène Joliot-Curie", "Irène", "Marie"]
questions = [
    "Who was Irène Joliot-Curie?",
    "What year did she begin her research?",
    "What prize did she win?",
    "In what year?"
]

cqa_system = ConversationalQASystem()

for q in questions:
    ans_with = cqa_system.get_answer(q, passage, use_history=True)
    ans_without = cqa_system.get_answer(q, passage, use_history=False)
    print(f"Q: {q}")
    print(f"  With history:    {ans_with}")
    print(f"  Without history: {ans_without}\n")
    cqa_system.update_subject(q, known_entities)
```

## Project Structure

```
.
├── Samar_Kh_Saleh_220223809_CIR_HW2.ipynb    # Full notebook: implementation, run, and report
├── README.md                                 # This file
├── requirements.txt                          # Python dependencies
├── .gitignore
└── LICENSE                                   # MIT License
```

## Dataset

This project uses a custom passage rather than the full CoQA/QuAC datasets, in line with the assignment's option to use "a custom passage with 3-4 related questions."

## Author

Samar Kh. Saleh
