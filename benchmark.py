#!/usr/bin/env python3
import time
import csv
from faster_whisper import WhisperModel
import gc

filename = "benchmark_results_additional.csv"

def get_model_and_compute_type(model_name: str) -> (str, str):
    """
    If the model name ends with '-int8', remove that suffix and return the base model name with compute_type 'int8'.
    Otherwise, return the model name with compute_type 'float32'.
    """
    if model_name.endswith("-int8"):
        # Remove the '-int8' suffix for a valid model name.
        base_model = model_name.replace("-int8", "")
        return base_model, "int8"
    return model_name, "float32"

def benchmark(models, beam_sizes, audio_files):
    """
    For each model and beam size, transcribe every audio file and measure the time taken.
    Returns a list of dictionaries with the average time for each (model, beam_size) combination.
    """
    results = []
    for model_alias in models:
        base_model, compute_type = get_model_and_compute_type(model_alias)
        print(f"\nLoading model '{base_model}' (alias: {model_alias}) with compute type '{compute_type}' on CPU...")
        # Instantiate the model for CPU use
        model = WhisperModel(base_model, device="cpu", compute_type=compute_type)
        
        for beam_size in beam_sizes:
            times = []
            print(f"\nBenchmarking model '{model_alias}' with beam size {beam_size}...")
            for audio_file in audio_files:
                print(f"  Processing file: {audio_file}")
                start_time = time.perf_counter()
                segment, info = model.transcribe(audio_file, beam_size=beam_size)
                transcription = " ".join([s.text for s in segment])
                print(f"    Transcription: {transcription}")
                elapsed = time.perf_counter() - start_time
                times.append(elapsed)
                print(f"    {audio_file} took {elapsed:.2f} seconds")
            average_time = sum(times) / len(times)
            print(f"  => Average for model '{model_alias}' with beam size {beam_size}: {average_time:.2f} seconds")
            results.append({
                "model": model_alias,
                "beam_size": beam_size,
                "average_time": average_time
            })
        
        del model
        gc.collect()
        print(f"\nModel '{model_alias}' unloaded. Waiting 10 seconds before loading the next model...")
        time.sleep(10)
        save_results_to_csv(results, filename)
        time.sleep(1)
    return results

def save_results_to_csv(results, filename):
    """
    Saves the benchmark results (only the average times) into a CSV file.
    """
    fieldnames = ["model", "beam_size", "average_time"]
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nResults saved to '{filename}'.")

if __name__ == "__main__":
    # Models to benchmark.
    # Use the -int8 suffix to denote that you want int8 quantization;
    # the code will convert them to valid model names.
    models = [
        "tiny-int8",
        "tiny",
        "tiny.en",
        "base-int8",
        "base",
        "base.en",
        "small-int8",
        "distil-small.en",
        "small",
        "small.en",
        "medium-int8",
        "distil-medium.en",
        "medium",
        "medium.en",
        # "large",
        # "large-v1",
        # "distil-large-v2",
        # "large-v2",
        # "distil-large-v3",
        # "large-v3",
        # "turbo"
    ]

    beam_sizes = [2, 5]
    # beam_sizes = [1,2,3,4,5,6,7,8,9,10]
    
    audio_files = [
        "audio/01_stt-stt.faster_whisper.wav",
        "audio/01_stt-stt.faster_whisper (1).wav",
        "audio/01_stt-stt.faster_whisper (2).wav",
        "audio/01_stt-stt.faster_whisper (3).wav",
        "audio/01_stt-stt.faster_whisper (4).wav",
        "audio/01_stt-stt.faster_whisper (5).wav",
    ]
    
    results = benchmark(models, beam_sizes, audio_files)
    save_results_to_csv(results, filename)
