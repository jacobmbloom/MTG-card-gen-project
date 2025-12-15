# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 12:28:55 2025

@author: jacob
"""

from transformers import GPT2Tokenizer, GPT2LMHeadModel

MODEL_PATH = "mtg-gpt2-model-ver7"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

def generate_card(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.4,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(out[0], skip_special_tokens=False))

generate_card("<|name|>mystic remora<|manaCost|>{1}{U}{U}<|power|><|toughness|><|type|>Creature - Fish<|text|>")
