For large

`modal deploy whisper_vllm.py`
`python test_script.py`


For turbo

`modal deploy whisper_vllm.py`
`python test_script.py`


Low hanging fruit: change test scripts to kick off multiple requests in parallel (would need larger warm pool because cold start time is high)

