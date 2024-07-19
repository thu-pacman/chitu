from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# tokenizer = AutoTokenizer.from_pretrained("/home/ss/models/Qwen2-7B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
# print(tokenizer.vocab_size)
# exit()
from transformers import pipeline

# messages = [
#     {
#         "role": "user",
#         "content": "宫保鸡丁怎么做?",
#     }
# ]
# pipe = pipeline(
#     "text-generation",
#     model="Qwen/Qwen2-7B-Instruct",
#     max_length=500,
#     device="cuda",
#     torch_dtype=torch.bfloat16,
# )
# output = pipe(messages)
# print(output)
# exit()

# Load pre-trained model and tokenizer
model_name = "Qwen/Qwen2-7B-Instruct"  # You can choose other models like 'gpt2-medium', 'gpt2-large', etc.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
).cuda()
tokenizer2 = AutoTokenizer.from_pretrained(model_name)
from cinfer.tokenizer import TokenizerHF, ChatFormatHF

tokenizer = TokenizerHF("/home/ss/models/Qwen2-7B-Instruct")
formatter = ChatFormatHF(tokenizer)

# Encode input text
# input_text = "Once upon a time"
# input_ids = tokenizer.encode(input_text, return_tensors='pt')
messages = [
    {"role": "user", "content": "宫保鸡丁怎么做?"},
]

model_inputs = formatter.encode_dialog_prompt(messages)
print(model_inputs)
model_inputs = torch.tensor(model_inputs, dtype=torch.int64).unsqueeze(0).cuda()
# text = tokenizer.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# print(text)
# model_inputs = tokenizer([text], return_tensors="pt")["input_ids"].cuda()
# print(model_inputs)
# print(tokenizer.decode(model_inputs[0], skip_special_tokens=True))

# Generate text
output = model.generate(
    model_inputs,
    max_length=500,  # Maximum length of the generated text
    num_return_sequences=1,  # Number of sequences to generate
    # no_repeat_ngram_size=2,  # No repetition of 2-grams
    # repetition_penalty=2.0,  # Penalty for repeating words
    # top_p=0.95,  # Top-p sampling
    temperature=0,  # Sampling temperature
)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0])
print(generated_text)
