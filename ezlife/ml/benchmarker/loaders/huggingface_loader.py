import time
import torch

import numpy as np

from ezlife.ml.benchmarker.utils.mem_utils import gc_cuda
from ezlife.ml.benchmarker.loaders.loader import Loader
from transformers import AutoModelForCausalLM, AutoTokenizer

class HuggingFaceLoader(Loader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def relevant_pkgs(self):
        return ['transformers', 'torch']

    def load(self):
        super().load()
        print(f"loading model....")
        common_params = {
            'local_files_only' : True
        }
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, **self.model_loader_args, **common_params)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, **common_params)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.to(self.device)
        print(f"model loaded, device - {self.model.device}")

    def warmup_model(self):
        print(f"warming up model....")
        example = "What is the meaning of life?"
        gc_cuda()

        inputs = self.tokenizer(example, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        for _ in range(self.warmup):
            self.model.generate(**inputs, **self.generate_args)
        print(f"model warmed up")

    def run_inference(self, example):
        gc_cuda()
        inputs = self.tokenizer(example, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        num_output_tokens = []
        latencies = []
        num_input_tokens = len(inputs['input_ids'][0])

        for _ in range(self.runs):
            start = time.perf_counter()
            tokenized_output = self.model.generate(**inputs, **self.generate_args)
            decoded_text = self.tokenizer.decode(tokenized_output[0], skip_special_tokens=True)
            print(f"decoded_text: {decoded_text}")
            end = time.perf_counter()

            latencies.append((end - start)  * 1000)
            tokens = len(tokenized_output[0])
            num_output_tokens.append(tokens)

        latency_avg = sum(latencies) / self.runs

        ret_val = {
            'latency_avg' : latency_avg,
            'latency_std' : np.std(latencies),
            'latency_p50' : np.percentile(latencies, 50, interpolation='higher'),
            'latency_p90' : np.percentile(latencies, 90, interpolation='higher'),
            'latency_p99' : np.percentile(latencies, 99, interpolation='higher'),
            'input_tokens' : num_input_tokens,
            'output_tokens' : np.mean(num_output_tokens),
            'total_tokens' : np.mean(num_output_tokens) + num_input_tokens,
            'tps_avg' : np.mean(num_output_tokens) / (latency_avg / 1000),
        }


        return ret_val