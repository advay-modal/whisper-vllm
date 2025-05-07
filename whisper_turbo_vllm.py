import time
import modal

app = modal.App("vllm-whisper-test-turbo")

image = modal.Image.debian_slim().pip_install(["vllm", "vllm[audio]"])


with image.imports():    
    from vllm import LLM, SamplingParams

@app.cls(min_containers=1, image = image, gpu="H100", cloud="oci")
class WhisperVLLM:
    @modal.enter()
    def enter(self):
        print("Loading")
        self.llm = LLM(
            model="openai/whisper-large-v3-turbo",
            max_model_len=448,
            max_num_seqs=1600,
            limit_mm_per_prompt={"audio": 1},
        )
        self.sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=200,
        )

    @modal.method()
    def infer(self, prompts):
        return self.llm.generate(prompts, self.sampling_params)