!pip install unsloth
!pip install --no-deps transformers accelerate peft bitsandbytes datasets

from unsloth import FastLanguageModel
import torch

model_id = "SRafi007/qwen3.5-0.8b-lora"
max_seq_length = 512
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
)

FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a B2B sales intelligence assistant. Analyze the user message and extract structured lead information."}]},
    {"role": "user", "content": [{"type": "text", "text": "We run an agency and want to resell your tool. who do i talk to about partnering?"}]},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(
    input_ids = inputs,
    max_new_tokens = 150,
    use_cache = True,
)

print(tokenizer.decode(outputs[0], skip_special_tokens = True))
