import pandas as pd

from ezlife.ml.benchmarker.utils.mem_utils import gc_cuda
from ezlife.ml.benchmarker.utils.misc import dict_to_uuid
from ezlife.ml.benchmarker.utils.gpu_utils import get_gpu_count, get_gpu_name
from ezlife.ml.benchmarker.loaders.huggingface_loader import HuggingFaceLoader

class Benchmarker:

    __LOADERS = {
        'hf' : HuggingFaceLoader,
        # 'exllamav2' : ExLLamaV2Loader,
        # 'vllm' : VLLMLoader,
    }

    __LIBRARIES = [
        'torch',
        'transformers',
        'exllamav2',
        'awq',
    ]

    def __init__(self, model_id, loader, model_loader_args, generate_args, runs = 20, warmup = 20):
        self.model_id = model_id
        self.loader = loader
        self.model_loader_args = model_loader_args
        self.generate_args = generate_args
        self.runs = 20
        self.warmup = 20
        self.library_versions = self.get_library_versions()

    def bm_latency_single_sample(self):
        gc_cuda()

        df = {
            'loader' : [],
            'model_id' : [],
            'model_loader_args' : [],
            'generate_args' : [],

            'latency_avg' : [],
            'latency_std' : [],
            'latency_p50' : [],
            'latency_p90' : [],
            'latency_p99' : [],
            'input_tokens' : [],
            'output_tokens' : [],
            'total_tokens' : [],
            'tps_avg' : [],

            # total_gpus, gpu_name
            'gpu_config' : [],
            # will be none if gpu_config is None
            'gpu_config_hash' : [],

            'torch_version' : [],
            'transformers_version' : [],
            'exllamav2_version' : [],
            'awq_version' : [],
            # hash will be only computed for frameworks that are relevant for a particular config
            'version_hash' : [],
        }

        loader = self.__LOADERS[backend](self.model_id, self.model_loader_args, self.runs, self.warmup)
        loader.load()

        loader.warmup_model()
        gc_cuda()

        stats = loader.run_inference('What is the meaning of life?')

        df['loader'].append(backend)
        df['model_id'].append(self.model_id)
        df['model_loader_args'].append(self.model_loader_args)
        df['generate_args'].append(self.generate_args)
        df['gpu_config'] = {
            'gpu_count' : get_gpu_count(),
            'gpu_name' : get_gpu_name(),
        }

        for key, value in stats.items():
            df[key].append(value)

        gc_cuda()

        df = pd.DataFrame(df)

        return df

    def get_library_versions(self):
        # try and import the libraries using importlib
        version_map = {}
        for lib in self.__LIBRARIES:
            try:
                module  = __import__(lib)
                version = module.__version__
                version_map[lib] = version
            except ImportError:
                pass
        return version_map


