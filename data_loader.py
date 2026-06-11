import os
import re
import glob
from typing import Union, Callable, Literal
import numpy as np
from mp_api.client import MPRester
from dataclasses import dataclass
from diffpy.structure.parsers import getParser
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.utils.parsers.loaddata import loadData
from dotenv import load_dotenv


@dataclass
class ToyPDFLoader:
    q: tuple
    r: tuple
    rstep: float = .01
    qdamp: float = 0.0
    material_ids: list[str] | None = None
    formulas: list[str] | None = None
    U: float | dict = .01  # if different elements have specified "wobbling" factor
    api_key: str | None = None
    key_name: str = "MP_API_KEY"

    def __post_init__(
        self,
    ):
        if self.api_key is None:
            load_dotenv()
            self.api_key = os.environ[self.key_name]
        if self.material_ids is None and self.formulas is None:
            raise ValueError(
                "Either material_ids or formulas have to be specified"
            )

    def _to_diffpy(
        self,
        pmg: list
    ):
        structure = getParser("cif").parse(pmg.to(fmt="cif"))
        for phase in structure:
            phase.element = re.sub(r"[0-9+\-]", "", phase.element)
        return structure

    def _pdf(
        self,
        structure
    ):
        if isinstance(self.U, dict):
            for el, num in self.U.items():
                structure[structure.element == el].Uisoequiv = num
        else:
            structure.Uisoequiv = self.U
        dpc = PDFCalculator(
            qmin=self.q[0], qmax=self.q[1], rmin=self.r[0], rmax=self.r[1] + self.rstep, qdamp=self.qdamp, rstep=self.rstep
        )
        return dpc(structure)

    def _fetch(
        self,
        mpr,
        *,
        formula: str = None,
        mid: str = None
    ):
        if formula is not None:
            docs = mpr.materials.summary.search(
                formula=formula, fields=["material_id", "energy_above_hull"]
            )
            mid = min(docs, key=lambda d: d.energy_above_hull).material_id
        pmg = mpr.get_structure_by_material_id(mid)
        return self._to_diffpy(pmg)

    def load(
        self
    ):
        with MPRester(self.api_key) as mpr:
            structs = [self._fetch(mpr, mid=m)
                       for m in (self.material_ids or [])]
            structs += [self._fetch(mpr, formula=f)
                        for f in (self.formulas or [])]
        r, gs = None, []
        for s in structs:
            r, g = self._pdf(s)
            # pmg.structure -> cif -> diff.py.Structure -> two arrays: g, r
            # (do not know concentration, thus, we cannot find F from X -- upload experimentally-hypothesized F)
            gs.append(g)
        return r, np.column_stack(gs)


@dataclass
class ExperimentalPDFLoader:
    directory: str
    pattern: str = "*.gr"
    sort: Union[Callable, None, Literal["name", "time"]] = None,
    r: tuple = (None, None)
    rstep: float = .01

    def _timestamp(
        self,
        file_name: str
    ):
        m = re.search(r"(\d{8}-\d{6})", os.path.basename(file_name))
        if m is not None:
            return m.group(1)
        raise FileNotFoundError(
            f"The file {os.path.basename(file_name)} does not have timestamp"
        )

    def _files_retriever(self):
        files = glob.glob(os.path.join(self.directory, self.pattern))
        if self.sort is None:
            return files
        if self.sort == "name":
            return sorted(files)
        if self.sort == "time":
            return sorted(files, key=self._timestamp)
        if callable(self.sort):
            return sorted(files, key=self.sort)

    def _read_data(
        self,
        file: str
    ):
        data = loadData(file)
        return data[:, 0], data[:, 1]

    def read_meta(
        self,
        file: str,
        meta: dict | None = None,
    ):
        if meta is None:
            meta = {}
        with open(file) as file:
            for line in file:
                if line.startswith('#### start data'):
                    break
                if '=' in line:
                    key, value = line.split('=', 1)
                    meta[key.strip()] = value.strip()  # remove white spaces
        return meta

    def _rescale(
        self,
        data: list
    ):
        if None not in (self.r[0], self.r[1], self.rstep):
            r = np.arange(self.r[0], self.r[1]+self.rstep, self.rstep)
            X = np.array([np.interp(r, r_prev, G_r) for r_prev, G_r in data])
        else:
            n = min([len(G) for _, G in data]) - 1
            r = data[0][0][:n]
            X = np.array([np.array(G[:n]) for _, G in data])
        return X, r

    def load(
        self,
        meta_apply: bool = False
    ):
        files = self._files_retriever()
        data = [self._read_data(f) for f in files]
        meta = []
        if meta_apply:
            meta = [self.read_meta(f) for f in files]
            self.r = (meta["rmin"], meta["rmax"])
            self.rstep = meta["rstep"]
        X, r = self._rescale(data=data)
        return X, r, meta
