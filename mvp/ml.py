import pytorch_lightning as pl
from loguru import logger
from transformers import LlamaTokenizer

from mvp import raw_data
from settings import Path, settings


class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self) -> None:
        raw_train_path = raw_data.generate_dataset(name='raw_train', seed=369, size=10, num_jobs=[10, 20])
        raw_val_path = raw_data.generate_dataset(name='raw_val', seed=369, size=10, num_jobs=[10, 20])
        # tokenizer = LlamaTokenizer.from_pretrained(settings.project_folder, legacy=False)
        # tokenizer.encode()


def main():
    dm = DataModule()
    dm.prepare_data()
    folder: Path = raw_data.Trajectory.get_data_folder() / 'raw_train'
    path = list(folder.glob('*.json'))[0]
    traj = raw_data.Trajectory.load(path)
    text = str(traj.samples[0].state)

    tokenizer = LlamaTokenizer.from_pretrained(settings.project_folder, legacy=False)
    # text = 'Salem Alem!!!!!!!!!'
    token_ids = tokenizer.encode(text)
    logger.debug("token_ids: {}", token_ids)
    # res = tokenizer.decode(token_ids, skip_special_tokens=True)
    # logger.debug("res: {}", res)
    res = tokenizer.tokenize(text)
    logger.debug("res: {}", res)
    pass


if __name__ == '__main__':
    main()
