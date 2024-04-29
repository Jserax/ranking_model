from math import ceil
from typing import Generator

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, data: pd.DataFrame, batch_size: int = 32) -> None:
        self.data = data
        self.q_ids = self.data.query_id.unique()
        self.seq_len = self.data.groupby("query_id")[["rank"]].count()
        self.batch_size = batch_size
        self.max_len = self.seq_len.max().item()

    def __len__(self) -> int:
        return ceil(len(self.q_ids) / self.batch_size)

    def batch(self) -> Generator[tuple[torch.Tensor], None, None]:
        np.random.shuffle(self.q_ids)

        for i in range(0, len(self.q_ids), self.batch_size):
            tmp1, tmp2 = list(), list()
            ids = self.q_ids[i : i + self.batch_size]
            seq_max = self.seq_len.loc[ids].max().item()
            for q_id in ids:
                seq_cur = self.seq_len.loc[q_id].item()
                tmp1.append(
                    np.pad(
                        self.data[self.data.query_id == q_id].drop(
                            columns=["query_id"]
                        ),
                        ((0, seq_max - seq_cur), (0, 0)),
                    )
                )
                tmp2.append([1] * seq_cur + [0] * (seq_max - seq_cur))
            tmp1 = torch.from_numpy(np.array(tmp1, dtype=np.float32))
            tmp2 = torch.from_numpy(np.array(tmp2, dtype=np.float32))
            yield tmp1[:, :, 1:], tmp1[:, :, 0], tmp2


def split_data(
    path: str, test_size: float = 0.1, random_state: int = 42
) -> tuple[pd.DataFrame]:
    data = pd.read_csv(path)
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    return train, test
