from ezlife.ml.benchmarker.utils.mem_utils import gc_cuda
from ezlife.ml.benchmarker.loaders.huggingface_loader import HuggingFaceLoader

class Benchmarker:

    __BACKEND_LOADERS = {
        'hf' : HuggingFaceLoader,
        # 'exllamav2' : ExLLamaV2Loader,
        # 'vllm' : VLLMLoader,
    }

    __LIBRARIES = [
        'torch',
        'transformers',
        'exllamav2',
        'vllm',
    ]

    def __init__(self, model_id, backends, runs = 20, warmup = 20):
        self.model_id = model_id
        self.backends = backends
        self.runs = 20
        self.warmup = 20
        self.library_versions = self.get_library_versions()

    def bm_latency_single_sample(self):
        gc_cuda()

        df = {
            'backend' : [],
            'latency_avg' : [],
            'latency_std' : [],
            'latency_p50' : [],
            'latency_p90' : [],
            'latency_p99' : [],
            'input_tokens' : [],
            'output_tokens' : [],
        }

        for backend in self.backends:
            loader = self.__BACKEND_LOADERS[backend](self.model_id)
            loader.load()
            stats = loader.run_inference('Hello World')
            df['backend'].append(backend)
            for key, value in stats.items():
                df[key].append(value)

            gc_cuda()

        df = pd.DataFrame(df)

        return df, library_versions

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


