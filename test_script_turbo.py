import modal
import time
from vllm.assets.audio import AudioAsset
def main():
    prompts = [
        {
            "prompt": "<|startoftranscript|>",
            "multi_modal_data": {
                "audio": AudioAsset("winning_call").audio_and_sample_rate,
            },
        },
        {  # Test explicit encoder/decoder prompt
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": AudioAsset("winning_call").audio_and_sample_rate,
                },
            },
            "decoder_prompt": "<|startoftranscript|>",
        }
    ] * 60
    print("Looking up")
    obj = modal.Cls.lookup("vllm-whisper-test-turbo", "WhisperVLLM")()
    print("About to call")
    durations = []
    for i in range(30):
        print(f"Run {i+1}/30")
        start = time.time()
        outputs = obj.infer.remote(prompts)
        duration = time.time() - start
        durations.append(duration)
        print(f"Run {i+1} Duration: {duration}")
        print(f"Run {i+1} RPS: {len(prompts) / duration}")
    
    durations.sort()
    p50 = durations[len(durations) // 2]
    p90 = durations[int(len(durations) * 0.9)]
    
    print("\nSummary:")
    print(f"p50 Duration: {p50}")
    print(f"p90 Duration: {p90}")
    print(f"p50 RPS: {len(prompts) / p50}")
    print(f"p90 RPS: {len(prompts) / p90}")

if __name__ == "__main__":
    main()
