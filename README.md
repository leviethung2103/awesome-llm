## Inference the LLM model with VLLM library
Serving model via API. Can use the OpenAI package to send the request.
vLLM supports distributed tensor-parallel inference and serving

Run the multi-gpu inference with LLM class, set the `tensor_parallel_size` argument to the number of GPus you want to use.

## Weakness
- Doesnot support the quantized model => Out of memory 
  - Need to write code load the quantized 4 bit model manually
  - Use the qcuf format file instead of bin file

## Run server
```
python -m vllm.entrypoints.api_server \
--model="qwen/Qwen-7B-Chat" \
--revision="v1.1.8" \
--trust-remote-code
```

Run with model mistrlal
```
python -m vllm.entrypoints.api_server \
--model="mistralai/Mistral-7B-v0.1" \
--trust-remote-code
```


## Query with model
```
curl http://localhost:8000/generate \
    -d '{
    "prompt": "San Francisco is a",
    "use_beam_search": true,
    "n": 4,
    "temperature": 0
    }'
```

## OpenAI Compatible Server
vLLM can be deployed as a server that mimics the OpenAI API protocol. This allows vLLM to be used as a drop-in replacement for applications using OpenAI API.

Use entrypoints.openai.api_server instead of entrypoint.api_server
```
python -m vllm.entrypoints.openai.api_server \
--model facebook/opt-125m
```

This server can be queried in the same format as OpenAI APi
```
curl http://localhost:8000/v1/models
```

Query the model with input prompts
```
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "facebook/opt-125m",
    "prompt": "San Francisco is a",
    "max_tokens": 7,
    "temperature": 0
    }'
```


can be used with the openai package
```
import openai
# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"
completion = openai.Completion.create(model="facebook/opt-125m",
                                      prompt="San Francisco is a")
print("Completion result:", completion)
```

## Supported Models
- Mistral
- PhoGPT
- LLAMA