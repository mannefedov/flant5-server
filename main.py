from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer



dev_map = {
    'shared': 1,
    'decoder.embed_tokens': 1,
    'encoder.embed_tokens': 1,
    'encoder.block.0': 1,
    'encoder.block.1': 1,
    'encoder.block.2': 1,
    'encoder.block.3': 1,
    'encoder.block.4': 1,
    'encoder.block.5': 1,
    'encoder.block.6': 1,
    'encoder.block.7': 1,
    'encoder.block.8': 1,
    'encoder.block.9': 1,
    'encoder.block.10': 1,
    'encoder.block.11': 1,
    'encoder.block.12': 1,
    'encoder.block.13': 1,
    'encoder.block.14': 1,
    'encoder.block.15': 1,
    'encoder.block.16': 1,
    'encoder.block.17': 1,
    'encoder.block.18': 1,
    'encoder.block.19': 1,
    'encoder.block.20': 1,
    'encoder.block.21': 1,
    'encoder.block.22': 1,
    'encoder.block.23': 2,
    'encoder.final_layer_norm': 2,
    'encoder.dropout': 2,
    'decoder.block.0': 2,
    'decoder.block.1': 2,
    'decoder.block.2': 2,
    'decoder.block.3': 2,
    'decoder.block.4': 2,
    'decoder.block.5': 2,
    'decoder.block.6': 2,
    'decoder.block.7': 2,
    'decoder.block.8': 2,
    'decoder.block.9': 2,
    'decoder.block.10': 2,
    'decoder.block.11': 2,
    'decoder.block.12': 2,
    'decoder.block.13': 2,
    'decoder.block.14': 2,
    'decoder.block.15': 3,
    'decoder.block.16': 3,
    'decoder.block.17': 3,
    'decoder.block.18': 3,
    'decoder.block.19': 3,
    'decoder.block.20': 3,
    'decoder.block.21': 3,
    'decoder.block.22': 3,
    'decoder.block.23': 3,
    'decoder.final_layer_norm': 3,
    'decoder.dropout': 3,
    'lm_head': 3
}

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl", 
                                           model_max_length=1024)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", 
                                                    device_map=dev_map, 
                                                    torch_dtype=torch.bfloat16, 
                                                    load_in_8bit=False)

task_prefix = "Extract the answer for the question from the given context." \
              "The answer should not contain non-relevant information." \
              "If the context does not contain the answer, just write 'No Answer'. " \
              "Question: {} Context: {} "


app = FastAPI()


class Answer(BaseModel):
    answer: str


class Query(BaseModel):
    question: str
    context: str


@app.post("/generate_qa", response_model=Answer)
def match_artists(data: Query):
    
    inputs = tokenizer([task_prefix.format(data.question, data.context) ], 
                        return_tensors="pt", padding=False)

    output_sequences = model.generate(
        max_new_tokens=15,
        temperature=0.8,
        early_stopping=True,
        num_return_sequences=1,
        input_ids=inputs["input_ids"].to(torch.device('cuda:3')),
        attention_mask=inputs["attention_mask"].to(torch.device('cuda:3')),
        do_sample=True,  # disable sampling to test if batching affects output
    )
    answer = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)[0]
    if answer == "No Answer":
        answer = ""

    
    return {"answer": answer}




@app.get("/")
def root():
    return {'message': "model is running!"}


