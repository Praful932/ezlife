from pathlib import Path
from ezlife.ml.benchmarker.utils import model_downloader
from ezlife.ml.benchmarker.utils.mem_utils import get_device, gc_cuda
from abc import abstractmethod

current_dir = Path(__file__).resolve().parent
MODEL_CACHE_DIR = current_dir / 'models'
MODEL_CACHE_DIR.mkdir(exist_ok=True)


class Loader:
    def __init__(self, model_id, model_loader_args, generate_args, runs, warmup):
        self.model_id = model_id
        self.model_loader_args = model_loader_args
        self.generate_args = generate_args
        self.warmup = warmup
        self.runs = runs
        self.device = get_device()
        self.model = None
        self.tokenizer = None
        self.model_dir = None


    def load(self, **kwargs):
        print(f"downloading model....")
        self.model_dir = model_downloader.download_model_from_hf(
            model_id = self.model_id,
            save_dir = MODEL_CACHE_DIR,
            **kwargs
        )
        print(f"downloaded model")
        gc_cuda()

    @abstractmethod
    def warmup_model(self):
        pass

    def run_inference(self, data):
        pass

    @property
    @abstractmethod
    def relevant_pkgs(self):
        pass