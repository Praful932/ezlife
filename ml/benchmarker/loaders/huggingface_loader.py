import time

from ezlife.ml.benchmarker.utils.mem_utils import gc_cuda
from ezlife.ml.benchmarker.loaders.loader import Loader
from ezlife.ml.transformers import AutoModelForCausalLM

class HFLoader(Loader):
    def __init__(self, model_id):
        super().__init__(model_id)
        self.model = None
        self.tokenizer = None

    def load(self):
        super().load(model_dir)
        params = {
            'low_cpu_mem_usage' : True,
            'torch_dtype' : torch.blfloat16,
        }
        model = AutoModelForCausalLM.from_pretrained(model_dir, **params)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def warmup(self):
        example = "What is the meaning of life?"
        gc_cuda()

        inputs = self.tokenizer(example, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        for _ in range(self.warmup):
            self.model.generate(**inputs)



    def run_inference(self, data):
        example = "What is the meaning of life?"
        gc_cuda()
        inputs = self.tokenizer(example, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        num_tokens = []
        latencies = []

        for _ in range(self.runs):
            start = time.perf_counter()
            tokenized_output = self.model.generate(**inputs)
            end = time.perf_counter()

            latencies.append((end - start)) * 1000
            tokens = len(tokenized_output[0])
            num_tokens.append(tokens)

        return {
            'latency_avg' : sum(latencies) / self.runs,
            'latency_std' : np.std(latencies),
            'latency_p50' : np.percentile(latencies, 50, interpolation='higher'),
            'latency_p90' : np.percentile(latencies, 90, interpolation='higher'),
            'latency_p99' : np.percentile(latencies, 99, interpolation='higher'),
            'input_tokens' : len(inputs['input_ids'][0], interpolation='higher'),
            'output_tokens' : np.mean(num_tokens),
        }