import pandas as pd

from ezlife.ml.benchmarker.utils.mem_utils import gc_cuda
from ezlife.ml.benchmarker.utils.misc import dict_to_uuid
from ezlife.ml.benchmarker.utils.gpu_utils import get_gpu_count, get_gpu_name
from ezlife.ml.benchmarker.loaders.huggingface_loader import HuggingFaceLoader
from ezlife.ml.benchmarker.loaders.exllamav2_loader import ExllamaV2Loader

class Benchmarker:

    __LOADERS = {
        'hf' : HuggingFaceLoader,
        'exllamav2' : ExllamaV2Loader,
        # 'vllm' : VLLMLoader,
    }

    __LIBRARIES = [
        'torch',
        'transformers',
        'exllamav2',
        'awq',
        'flash-attn'
    ]

    def __init__(self, model_id, loader, model_loader_args, generate_args, model_downloader_args = None, runs = 20, warmup = 20):
        self.model_id = model_id
        self.loader = loader
        self.model_loader_args = model_loader_args
        self.model_downloader_args = {} if model_downloader_args is None else model_downloader_args
        self.generate_args = generate_args
        self.runs = 20
        self.warmup = 20
        self.library_versions = self.get_library_versions()

    def bm_latency_single_sample(self, examples = None):
        if examples is None:
            examples = [
                "What is the meaning of life?",
                "Explain how to make a french toast in the funniest way possible",
                "What is the best way to make a million dollars in a day?",
                "Write a poem about the moon and the stars",
            ]


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
            'avg_input_tokens' : [],
            'avg_output_tokens' : [],
            'avg_total_tokens' : [],
            'tps_avg' : [],

            # total_gpus, gpu_name
            'gpu_config' : [],
            # will be none if gpu_config is None

            'pkg_version' : [],

            'model_loader_args_hash' : [],
            'gpu_config_hash' : [],
            # hash will be only computed for frameworks that are relevant for a particular config
            'version_hash' : [],
        }

        loader_ob = self.__LOADERS[self.loader](
            model_id = self.model_id,
            model_loader_args = self.model_loader_args,
            generate_args = self.generate_args,
            runs = self.runs,
            warmup = self.warmup,
        )
        loader_ob.load(**self.model_downloader_args)

        loader_ob.warmup_model()
        gc_cuda()

        stats = loader_ob.run_inference(
            examples = examples
        )

        df['loader'].append(self.loader)
        df['model_id'].append(self.model_id)
        df['model_loader_args'].append(self.model_loader_args)
        df['generate_args'].append(self.generate_args)
        df['gpu_config'].append({
            'gpu_count' : get_gpu_count(),
            'gpu_name' : get_gpu_name(),
        })
        df['gpu_config_hash'] = dict_to_uuid(df['gpu_config'])

        relevant_pkg_versions = {k : v for k, v in self.library_versions.items() if k in loader_ob.relevant_pkgs}



        df['pkg_version'] = [relevant_pkg_versions]
        df['version_hash'] = dict_to_uuid(relevant_pkg_versions)
        df['model_loader_args_hash'] = dict_to_uuid(self.model_loader_args)

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


