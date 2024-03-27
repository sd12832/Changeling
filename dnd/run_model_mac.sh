#!/bin/bash

CMAKE_ARGS="-DLLAMA_METAL=on" pip install 'llama-cpp-python[server]'
python -m llama_cpp.server --model dnd/loyal-macaroni-maid-7b.Q4_K_M.gguf