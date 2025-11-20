from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import logging
import torch
import esm
import esm.inverse_folding
import os, re, gzip, io
import torch.nn.functional as F
from glob import glob
import numpy as np
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import statistics as stats

LOG = logging.getLogger(__name__)



AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {a:i for i,a in enumerate(AA20)}

def _open_any(path):
    return io.TextIOWrapper(gzip.open(path, "rb")) if str(path).endswith(".gz") else open(path, "r")

def read_msa_a2m(path):
    """Read A2M/A3M-like file -> list[str] (raw)."""
    seqs, cur = [], []
    with _open_any(path) as f:
        for line in f:
            if not line: break
            if line.startswith(">"):
                if cur: seqs.append(''.join(cur).strip()); cur = []
            else:
                cur.append(line.strip())
        if cur: seqs.append(''.join(cur).strip())
    return seqs  # [N]

def _strip_a2m_insertions(seq: str) -> str:
    """
    A2M rule: keep only alignment columns (uppercase letters and '-').
    Drop lowercase (insertions) and '.' (gaps aligned to insert columns).
    """
    # remove lowercase letters and '.' characters
    seq = re.sub(r'[a-z\.]', '', seq)
    # keep uppercase A-Z and '-' only
    seq = re.sub(r'[^A-Z\-]', '', seq)
    return seq

def clean_alignment_a2m(msa_raw):
    """
    Returns ndarray [N, Lmatch] of {A..Z or '-'} after:
    1) Removing inserts (lowercase) and '.' from each sequence.
    2) Ensuring all remaining sequences have the same number of columns (match/deletion columns).
    3) Dropping columns where the QUERY has '-' so final L equals target length.
    """
    # 1) strip per-seq inserts/dots
    query = msa_raw[0]
    # 2) sanity: all have same length (match+delete columns)
    col_len = len(query)
    assert all(len(s) == col_len for s in msa_raw), "A2M parsing error: unequal column lengths after stripping inserts/dots."
    arr = np.array([list(s) for s in msa_raw], dtype='<U1')  # [N, Lmatch]

    return arr  # characters in {A..Z or '-'}

def filter_sequences(arr_all, max_gap_frac=0.5, min_cov_to_query=0.7):
    """Keep sequences with <= max_gap_frac gaps and >= min_cov_to_query identity vs query (ignoring gaps)."""
    N, L = arr_all.shape
    q = arr_all[0]
    keep = [0]
    for i in range(1, N):
        row = arr_all[i]
        if (row == '-').mean() > max_gap_frac:
            continue
        mask = (row != '-') & (q != '-')
        if mask.sum() == 0:
            continue
        if (row[mask] == q[mask]).mean() < min_cov_to_query:
            continue
        keep.append(i)
    out = arr_all[keep]
    # Invariant: columns unchanged by row filtering
    assert out.shape[1] == L, "Row filtering must not change the number of columns."
    return out

def sequence_reweighting_fast(arr, theta=0.8):
    """
    Fast approx: weight by identity to query (not full O(N^2)).
    Lower weight for very similar sequences.
    """
    N, L = arr.shape
    query = arr[0]
    w = np.ones(N, dtype=np.float32)
    for i in range(1, N):
        mask = (arr[i] != '-') & (query != '-')
        ident = np.mean(arr[i,mask] == query[mask]) if mask.sum() else 0.0
        if ident >= theta:
            w[i] = 1.0 / (1.0 + 5.0*(ident-theta)/(1.0-theta))
    return w

def msa_to_prob(arr, tau=1.0, pseudocount=0.5, use_reweight=True):
    """
    Convert cleaned A2M alignment [N,L] -> probs [L,20].
    Ignore gaps/ambiguous letters; Dirichlet pseudocount; optional reweighting and temperature.
    """
    N, L = arr.shape
    w = sequence_reweighting_fast(arr) if use_reweight else np.ones(N, dtype=np.float32)
    counts = np.full((L, 20), fill_value=pseudocount, dtype=np.float64)
    for i in range(N):
        row = arr[i]
        wi = float(w[i])
        for j in range(L):
            a = row[j]
            if a == '-' or a not in AA2IDX:
                continue
            counts[j, AA2IDX[a]] += wi
    probs = counts / counts.sum(-1, keepdims=True)
    if tau != 1.0:
        logits = np.log(probs + 1e-8) / tau
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs = probs / probs.sum(-1, keepdims=True)
    return torch.from_numpy(probs.astype(np.float32))

def build_msa_prob_for_assay_a2m(msa_path, expect_length=None, 
                                 max_gap_frac=0.5, min_cov_to_query=0.7, 
                                 tau=0.9, pseudocount=0.5, use_reweight=True):
    """
    End-to-end: A2M -> [L,20] probs aligned to the target sequence (drop query deletions).
    """
    raw = read_msa_a2m(msa_path)
    arr = clean_alignment_a2m(raw)
    arr = filter_sequences(arr, max_gap_frac=max_gap_frac, min_cov_to_query=min_cov_to_query)
    probs = msa_to_prob(arr, tau=tau, pseudocount=pseudocount, use_reweight=use_reweight)
    if expect_length is not None:
        assert probs.shape[0] == expect_length, f"L mismatch: MSA L={probs.shape[0]} vs expected {expect_length}"
    return probs


def build_msa_bank_a2m(name, msa_dir, length, **kwargs):
    prefix = "_".join(name[0].split("_")[:2])
    match = next(Path(msa_dir).glob(f"{prefix}*"), None)  # Path or None
    candidate = str(match) if match else None  

    probs = build_msa_prob_for_assay_a2m(candidate, expect_length=length, **kwargs)
    return probs


def ensure_k(items, k=100, seed=42):
    if not items:
        return []
    if len(items) >= k:
        return items[:k]
    rng = np.random.default_rng(seed)
    return items + rng.choice(items, size=k - len(items), replace=True).tolist()

def seq_acts(model, tokens):
    with torch.no_grad():
        model = model.cuda()
        tokens = tokens.cuda()
        result = model(tokens, repr_layers=[32], need_head_weights=True)
    x = {'input': tokens, 
            'logits': result['logits'].detach()[0], 
            'representation': torch.stack([v.detach() for _, v in sorted(result['representations'].items())], dim=2), 
            'attention': result['attentions'].detach().permute(0, 4, 3, 1, 2).flatten(3, 4)}
    return x   

class ProteinGymSubstitutionDataset(Dataset):
    def __init__(self, assay_name, wt_sequence, coords, train_data, valid_data):
        super().__init__()
        self.assay_name = assay_name
        self.higher_mutseqs_100 = []
        self.lower_mutseqs_100 = []
        self.msa_bank = build_msa_bank_a2m(name=assay_name, msa_dir="DMS_msa_files", length=len(wt_sequence[0]), tau=0.9, pseudocount=0.5, max_gap_frac=0.5, min_cov_to_query=0.7, use_reweight=True)

        _, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        _, self.structure_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.coord_converter = esm.inverse_folding.util.CoordBatchConverter(self.structure_alphabet)
        self.coords = []
        for coord in coords:
            batch = [(coord, None, None)]
            coord, confidence, _, _, padding_mask = self.coord_converter(batch)
            self.coords.append((coord, padding_mask, confidence))

        self.batch_tokens = []
        for sequence in wt_sequence:
            _, _, batch_tokens = self.batch_converter([('protein', sequence)])
            self.batch_tokens.append(batch_tokens)

        self.raw_scores = []
        for assay_idx, dms_data in enumerate(train_data):
            for index, data in dms_data.iterrows():
                self.raw_scores.append(data['DMS_score'])

        for assay_idx, dms_data in enumerate(valid_data):
            for index, data in dms_data.iterrows():
                self.raw_scores.append(data['DMS_score'])
        self._compute_stats()

        self.train_labels = []
        for assay_idx, dms_data in enumerate(train_data):
            train_label = []
            dms = dms_data.copy()
            dms['DMS_score'] = pd.to_numeric(dms['DMS_score'], errors='coerce')

            # presort by score (descending) once
            sorted_df   = dms.sort_values('DMS_score', ascending=False).reset_index(drop=True)
            mutseq_desc = sorted_df['mutated_sequence'].tolist()

            higher_mutseqs_100 = mutseq_desc[:100] 
            for sequence in higher_mutseqs_100:
                _, _, batch_tokens = self.batch_converter([('protein', sequence)])
                self.higher_mutseqs_100.append(batch_tokens)

            lower_mutseqs_100 = mutseq_desc[-100:] 
            for sequence in lower_mutseqs_100:
                _, _, batch_tokens = self.batch_converter([('protein', sequence)])
                self.lower_mutseqs_100.append(batch_tokens)

            for index, data in dms_data.iterrows():
                mutants = data['mutant'].split(':')
                mutant_list = []
                for mutant in mutants:
                    location = int(mutant[1:-1])
                    mutant_list.append((location, torch.tensor(self.alphabet.tok_to_idx[mutant[:1]], dtype=torch.long), torch.tensor(self.alphabet.tok_to_idx[mutant[-1:]], dtype=torch.long)))
                
                _, _, batch_tokens = self.batch_converter([('protein', data['mutated_sequence'])])

                train_label.append((torch.tensor(data['DMS_score'], dtype=torch.float32), mutant_list, batch_tokens))
            self.train_labels.append(train_label)
        
        self.valid_labels = []
        for assay_idx, dms_data in enumerate(valid_data):
            valid_label = []

            for index, data in dms_data.iterrows():
                mutants = data['mutant'].split(':')
                mutant_list = []
                for mutant in mutants:
                    location = int(mutant[1:-1])
                    mutant_list.append((location, torch.tensor(self.alphabet.tok_to_idx[mutant[:1]], dtype=torch.long), torch.tensor(self.alphabet.tok_to_idx[mutant[-1:]], dtype=torch.long)))
                _, _, batch_tokens = self.batch_converter([('protein', data['mutated_sequence'])])

                valid_label.append((torch.tensor(data['DMS_score'], dtype=torch.float32), mutant_list, batch_tokens))
            self.valid_labels.append(valid_label)

    def __getitem__(self, index):
        return self.assay_name, self.batch_tokens, self.coords, self.train_labels, self.valid_labels, self.msa_bank, self.higher_mutseqs_100, self.lower_mutseqs_100
    

    def __len__(self):
        return 1
    
    def _compute_stats(self):
        arr = np.asarray(self.raw_scores, dtype=float)

        self.med = arr.mean()
        self.std = arr.std()


    def _norm(self, raw):
        return (raw - self.med) / (self.std + 1e-10)


class ProteinGymSubstitutionData(LightningDataModule):
    """`LightningDataModule` for the ProteinGym dataset.
    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        assay_index: int = 0, # 0 - 100
        split_type: str = "random", # random, modulo, contiguous
        split_index: int = 0, # 0 - 4
        support_assay_num: int = 40,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        _, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 1

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """

        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            self.hparams.data_dir = Path(self.hparams.data_dir)
            
            assay_reference_file = pd.read_csv(self.hparams.data_dir/'reference'/'DMS_substitutions.csv')
            assay_id = assay_reference_file["DMS_id"][self.hparams.assay_index]
            print("assay id", assay_id)
            assay_file_name = assay_reference_file["DMS_filename"][assay_reference_file["DMS_id"]==assay_id].values[0]
            pdb_file_name = assay_reference_file["pdb_file"][assay_reference_file["DMS_id"]==assay_id].values[0]
            assay_data = pd.read_csv(self.hparams.data_dir/'substitutions'/assay_file_name)

            wt_sequence = assay_reference_file["target_seq"][assay_reference_file["DMS_id"]==assay_id].values[0]
            
            structure = esm.inverse_folding.util.load_structure(str(self.hparams.data_dir/'structure'/pdb_file_name), 'A')
            coord, _ = esm.inverse_folding.util.extract_coords_from_structure(structure)

            self.data_train = ProteinGymSubstitutionDataset([assay_id], [wt_sequence], [coord],
                 [assay_data[assay_data[f"fold_{self.hparams.split_type}_5"] != self.hparams.split_index].reset_index()], 
                 [assay_data[assay_data[f"fold_{self.hparams.split_type}_5"] == self.hparams.split_index].reset_index()])
            self.data_val = ProteinGymSubstitutionDataset([assay_id], [wt_sequence], [coord],
                 [assay_data[assay_data[f"fold_{self.hparams.split_type}_5"] != self.hparams.split_index].reset_index()], 
                 [assay_data[assay_data[f"fold_{self.hparams.split_type}_5"] == self.hparams.split_index].reset_index()])

            LOG.info(f'Target assay {assay_id}; Length: {len(wt_sequence)}')

    def collator(self, raw_batch):
        return raw_batch[0]

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator,
            shuffle=True
        )
    
    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator,
            shuffle=False
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
