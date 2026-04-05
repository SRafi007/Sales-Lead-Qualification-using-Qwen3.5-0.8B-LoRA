
# AI Sales Lead Qualification (Qwen3.5-0.8B + LoRA)

A lightweight AI system that converts unstructured B2B sales messages into structured lead intelligence using a fine-tuned LLM.

---

##  Overview

This project demonstrates how a small language model like Qwen3.5-0.8B can be fine-tuned using LoRA to perform structured information extraction for sales workflows.

The model takes raw input such as:

> "We need a demo for 200 users ASAP"

And converts it into:

```json
{
  "authority_level": "decision_maker",
  "budget_range": "high",
  "intent": "demo_request",
  "lead_type": "enterprise",
  "recommended_action": "Schedule executive demo within 24 hours",
  "urgency": "high"
}
````

---

##  Model

* Base Model: Qwen3.5-0.8B
* Fine-tuning: LoRA (PEFT)
* Framework: Unsloth
* Task: Structured lead qualification (classification + extraction)

 Hugging Face Link:
[https://huggingface.co/SRafi007/qwen3.5-0.8b-lora-lead-qualifier]([https://huggingface.co/YOUR-USERNAME/YOUR-MODEL](https://huggingface.co/SRafi007/qwen3.5-0.8b-lora-lead-qualifier))

---

##  Dataset

* ~1248 synthetic B2B sales conversations
* Designed to simulate real-world lead scenarios:

  * demo requests
  * pricing inquiries
  * integration questions
  * enterprise vs startup leads

⚠️ Note: Dataset is relatively small → model may not generalize perfectly.

---

##  Features

* Extracts structured lead insights from raw text
* Works with a lightweight 0.8B model
* Efficient fine-tuning using LoRA
* Simple Gradio interface for testing

---

##  Demo

![Demo Screenshot](assets/demo.png)

---

## Usage

### Install dependencies

```bash
!pip install unsloth
!pip install --no-deps transformers accelerate peft bitsandbytes datasets
```

---

### Run the app

---

### Example input

```
We are a startup exploring pricing and Slack integration.
```

---

### Output

```json
{
  "authority_level": "recommender",
  "budget_range": "low",
  "intent": "integration_question",
  "lead_type": "startup",
  "recommended_action": "Share integration documentation",
  "urgency": "medium"
}
```

---

##  Limitations

* Small dataset → risk of overfitting
* Output may not always be perfectly valid JSON
* Needs more real-world noisy data for robustness

---

##  Future Improvements

* Scale dataset to 10k+ samples
* Deploy as API (FastAPI / production system)

---

##  Contact

If you're interested in:

* Training pipeline / notebook
* Dataset generation approach
* Scaling this system

Feel free to reach out.
- sadmansakibrafi.hey@gmail.com

---
