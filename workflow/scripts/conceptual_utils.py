from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Mutation:
    """Class for keeping track of mutations."""

    codon: int
    ref_allele: str
    mut_allele: str

    @staticmethod
    def from_name(name):
        """A name like: C35R"""

        ref_allele = name[0]
        mut_allele = name[-1]
        codon = int(name[1:-1])

        return Mutation(codon, ref_allele, mut_allele)

    @property
    def name(self):
        return f"{self.ref_allele}{self.codon}{self.mut_allele}"

    def make_change(self, seq):

        lseq = list(seq)
        orig = lseq[self.codon - 1]
        if orig == self.ref_allele:
            new = self.mut_allele
            desc = "gain"
        elif orig == self.mut_allele:
            new = self.ref_allele
            desc = "loss"
        else:
            return None, None, None

        lseq[self.codon - 1] = new
        return "".join(lseq), self.name, desc


def mutate_sequence(mutation, examples):

    out = {"sequence": [], "change": [], "desc": []}
    for _id, seq in zip(examples["id"], examples["sequence"]):

        nseq, change, desc = mutation.make_change(seq)

        if nseq is None:
            out["sequence"].append(seq)
            out["change"].append("original")
            out["desc"].append("original")
        else:
            out["sequence"].append(nseq)
            out["change"].append(change)
            out["desc"].append(desc)
    return out


def process_sklearn(model, dataset, targets):
    preds = model.predict_proba(np.array(dataset["sequence"]))

    out_data = {}
    for num, field in enumerate(targets):
        try:
            out_data[field] = preds[num][:, 1]
        except IndexError:
            out_data[field] = preds[num][:, 0]
        
    return pd.DataFrame(out_data)


def process_huggingface(trainer, dataset, targets):

    preds = trainer.predict(dataset)
    info = pd.DataFrame(1 / (1 + np.exp(-preds.predictions)), columns=targets)

    return info
