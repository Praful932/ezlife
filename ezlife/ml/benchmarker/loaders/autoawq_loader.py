import time
import random
import torch

import numpy as np

from ezlife.ml.benchmarker.utils.mem_utils import gc_cuda
from ezlife.ml.benchmarker.utils.model_utils import decoder_parser
from ezlife.ml.benchmarker.loaders.loader import Loader
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

class AutoAWQLoader(Loader):
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
        self.model = AutoAWQForCausalLM.from_pretrained(self.model_dir, **self.model_loader_args, **common_params)
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
        print("model warmed up\n")

    def run_inference(self, examples):
        gc_cuda()


        num_output_tokens = []
        latencies = []


        for _ in range(self.runs):
            example = random.choice(examples)

            inputs = self.tokenizer(example, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            num_input_tokens = len(inputs['input_ids'][0])

            start = time.perf_counter()
            tokenized_output = self.model.generate(**inputs, **self.generate_args)
            decoded_text = self.tokenizer.decode(tokenized_output[0], skip_special_tokens=True)
            decoded_text = decoder_parser([decoded_text], [example], lambda x: x.strip())[0]
            print(f"decoded_text: {decoded_text}")
            end = time.perf_counter()

            latencies.append((end - start)  * 1000)
            num_output_tokens = len(tokenized_output[0]) - num_input_tokens

            del inputs

        latency_avg = sum(latencies) / self.runs

        ret_val = {
            'latency_avg' : latency_avg,
            'latency_std' : np.std(latencies),
            'latency_p50' : np.percentile(latencies, 50, interpolation='higher'),
            'latency_p90' : np.percentile(latencies, 90, interpolation='higher'),
            'latency_p99' : np.percentile(latencies, 99, interpolation='higher'),
            'avg_input_tokens' : num_input_tokens,
            'avg_output_tokens' : np.mean(num_output_tokens),
            'avg_total_tokens' : np.mean(num_output_tokens) + num_input_tokens,
            'tps_avg' : np.mean(num_output_tokens) / (latency_avg / 1000),
        }


        return ret_val