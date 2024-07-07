import time
import random
import numpy as np

from ezlife.ml.benchmarker.utils.mem_utils import gc_cuda
from ezlife.ml.benchmarker.loaders.loader import Loader
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer, Timer
from exllamav2.generator import ExLlamaV2DynamicGenerator


class ExllamaV2Loader(Loader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def relevant_pkgs(self):
        return ['transformers', 'torch', 'exllamav2', 'flash-attn' 'auto-gptq']

    def load(self, **kwargs):
        super().load(**kwargs)
        self.model_dir = str(self.model_dir)
        self.config = ExLlamaV2Config(self.model_dir)
        self.model = ExLlamaV2(self.config, **self.model_loader_args)
        self.cache = ExLlamaV2Cache(self.model, max_seq_len = 8192, lazy = True)
        self.model.load_autosplit(self.cache, progress = False)
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.generator = ExLlamaV2DynamicGenerator(
            model = self.model,
            cache = self.cache,
            tokenizer = self.tokenizer,
        )

    def warmup_model(self):
        print("Warming up model")

        self.generator.warmup()

        example = "What is the meaning of life?"

        output = self.generator.generate(
            prompt = example,
            **self.generate_args,
        )

        print(output)

        print("model warmed up\n")

    def run_inference(self, examples):
        gc_cuda()

        num_output_tokens = []
        latencies = []

        for _ in range(self.runs):
            example = random.choice(examples)
            num_input_tokens = len(self.tokenizer.encode(example)[0])
            start = time.perf_counter()
            output_including_prompt = self.generator.generate(
                prompt = example,
                **self.generate_args,
            )
            output_excluding_prompt = output_including_prompt[len(example):]
            end = time.perf_counter()
            num_output_tokens = len(self.tokenizer.encode(output_excluding_prompt)[0])
            latency = (end - start) * 1000
            latencies.append(latency)
            tps = num_output_tokens / (latency / 1000)
            print(f"tps - {tps}, latency -  {latency}, input - {example}, output - {repr(output_excluding_prompt)}")

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