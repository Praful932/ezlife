from pathlib import Path
from ezlife.ml.benchmarker.utils import model_downloader
from ezlife.ml.benchmarker.utils.mem_utils import get_device
from abc import abstractmethod

current_dir = Path(__file__).resolve().parent
MODEL_CACHE_DIR = current_dir / 'models'
MODEL_CACHE_DIR.mkdir(exist_ok=True)


class Loader:
    def __init__(self, model_id, runs, warmup):
        self.model_id = model_id
        self.warmup = warmup
        self.runs = runs
        self.device = get_device()
        self.model = None
        self.tokenizer = None
        self.model_dir = None


    def load(self):
        print(f"downloading model....")
        self.model_dir = model_downloader.download_model_from_hf(
            model_id = self.model_id,
            save_dir = MODEL_CACHE_DIR
        )
        print(f"downloaded model")

    @abstractmethod
    def warmup_model(self):
        pass

    def run_inference(self, data):
        pass