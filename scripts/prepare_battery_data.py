#!/usr/bin/env python3
"""Convert user-provided NASA PCoE and CALCE CS2 archives to cycle-level CSVs.

The source archives remain local and are not committed to GitHub. This is an
independently implemented, reproducible *data preparation* utility; it does not
replicate any paper's full feature-engineering, EOL definition or splits.

NASA is a ZIP of six nested ZIPs containing battery-specific MATLAB files.
CALCE CS2 archives contain chronologically segmented XLSX workbooks, where
Discharge_Capacity(Ah) is cumulative *within* a workbook. Per-cycle capacity
is max-min across that workbook/cycle (not the cumulative maximum).
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
import posixpath
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

NASA_DEFAULT = ["B0005", "B0006", "B0007", "B0018", "B0029", "B0030", "B0031", "B0032", "B0053", "B0054", "B0055", "B0056"]
CALCE_DEFAULT = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
SENSOR_FEATURES = ["voltage_mean", "voltage_std", "current_mean", "current_std", "temperature_mean", "discharge_duration_s"]

def _num(x):
    try:
        v = float(np.asarray(x).reshape(-1)[0])
        return v if math.isfinite(v) else math.nan
    except (ValueError, TypeError, IndexError):
        return math.nan

def _mean(x):
    try:
        a = np.asarray(x, dtype=float).reshape(-1)
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else math.nan
    except (ValueError, TypeError):
        return math.nan

def _std(x):
    try:
        a = np.asarray(x, dtype=float).reshape(-1)
        a = a[np.isfinite(a)]
        return float(a.std()) if a.size else math.nan
    except (ValueError, TypeError):
        return math.nan

def _elapsed(x):
    try:
        a = np.asarray(x, dtype=float).reshape(-1)
        a = a[np.isfinite(a)]
        return float(a[-1] - a[0]) if a.size > 1 else math.nan
    except (ValueError, TypeError):
        return math.nan

def _finish(df: pd.DataFrame, cell: str, nominal_ah: float | None = None):
    if df.empty:
        return df
    df = df.reset_index(drop=True).copy()
    # Nominal-capacity and first-observed-capacity SOH are different targets.
    first = float(df["capacity_ah"].iloc[0])
    df["soh_first_observed"] = df["capacity_ah"] / first
    if nominal_ah is not None:
        df["soh_nominal"] = df["capacity_ah"] / nominal_ah
    df["target"] = df["soh_first_observed"]
    df["time"] = np.arange(1, len(df)+1)
    df["cycle"] = df["time"]
    df["cell_id"] = cell
    # An observed-end countdown is NOT an EOL-based RUL label.
    df["remaining_observed_cycles"] = len(df) - df["time"]
    return df

def load_nasa_mat(mat_bytes: bytes, cell: str) -> pd.DataFrame:
    mat = loadmat(io.BytesIO(mat_bytes), squeeze_me=True, struct_as_record=False)
    b = mat[cell] if cell in mat else next(v for k, v in mat.items() if not k.startswith('__'))
    cycles = np.atleast_1d(b.cycle)
    rows=[]
    for cycle in cycles:
        if str(getattr(cycle, "type", "")).lower() != "discharge":
            continue
        d = cycle.data
        cap = _num(getattr(d, "Capacity", None))
        if not (math.isfinite(cap) and cap > 0):
            continue
        rows.append({"capacity_ah":cap,
                     "voltage_mean":_mean(getattr(d, "Voltage_measured", None)),
                     "voltage_std":_std(getattr(d, "Voltage_measured", None)),
                     "current_mean":_mean(getattr(d, "Current_measured", None)),
                     "current_std":_std(getattr(d, "Current_measured", None)),
                     "temperature_mean":_mean(getattr(d, "Temperature_measured", None)),
                     "discharge_duration_s":_elapsed(getattr(d, "Time", None)),
                     "ambient_temperature":_num(getattr(cycle, "ambient_temperature", None))})
    return _finish(pd.DataFrame(rows), cell, nominal_ah=2.0)

def convert_nasa_zip(archive: Path, outdir: Path, cells: list[str], extract_mat_to: Path | None = None):
    results={}
    seen=set()
    with zipfile.ZipFile(archive) as outer:
        for entry in outer.infolist():
            if entry.is_dir() or not entry.filename.lower().endswith('.zip'):
                continue
            if all(c in seen for c in cells):
                break
            with zipfile.ZipFile(io.BytesIO(outer.read(entry))) as nested:
                for member in nested.infolist():
                    cell = Path(member.filename).stem.upper()
                    if cell in cells and cell not in seen and member.filename.lower().endswith('.mat'):
                        mat_bytes=nested.read(member)
                        if extract_mat_to is not None:
                            extract_mat_to.mkdir(parents=True,exist_ok=True)
                            (extract_mat_to / f"{cell}.mat").write_bytes(mat_bytes)
                        df = load_nasa_mat(mat_bytes, cell)
                        if df.empty: raise ValueError(f"NASA {cell} has no usable discharge cycles")
                        target = outdir / "nasa" / f"{cell}.csv"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        df.to_csv(target, index=False)
                        results[cell] = {"records":len(df),"file":str(target),"capacity_first_ah":round(float(df.capacity_ah.iloc[0]),4), "capacity_last_ah":round(float(df.capacity_ah.iloc[-1]),4)}
                        seen.add(cell)
    missing = sorted(set(cells)-seen)
    if missing:raise ValueError(f"NASA cells not found: {missing}")
    return results

class CycleAccumulator:
    __slots__=("count", "min_cap", "max_cap", "volt_sum", "volt_sq", "curr_sum", "curr_sq", "n_sensor", "disch_samples", "first_t", "last_t", "start_dt","disch_first_t","disch_last_t")
    def __init__(self):
        self.count=0; self.min_cap=math.inf; self.max_cap=-math.inf
        self.volt_sum=0.; self.volt_sq=0.; self.curr_sum=0.; self.curr_sq=0.
        self.n_sensor=0; self.disch_samples=0; self.first_t=math.inf; self.last_t=-math.inf
        self.start_dt=None;self.disch_first_t=math.inf;self.disch_last_t=-math.inf
    def add(self, date, t, current, voltage, discharge_cap):
        self.count+=1
        if date is not None and (self.start_dt is None or date < self.start_dt):self.start_dt=date
        if math.isfinite(discharge_cap):
            self.min_cap=min(self.min_cap,discharge_cap);self.max_cap=max(self.max_cap,discharge_cap)
        if math.isfinite(t):self.first_t=min(self.first_t,t);self.last_t=max(self.last_t,t)
        if math.isfinite(current) and math.isfinite(voltage):
            self.curr_sum+=current;self.curr_sq+=current*current
            self.volt_sum+=voltage;self.volt_sq+=voltage*voltage
            self.n_sensor+=1
        if math.isfinite(current) and current < -0.01:
            self.disch_samples+=1
            if math.isfinite(t):self.disch_first_t=min(self.disch_first_t,t);self.disch_last_t=max(self.disch_last_t,t)
    def row(self, workbook, local_cycle):
        if self.max_cap == -math.inf:return None
        cap=self.max_cap-self.min_cap
        if not (math.isfinite(cap) and cap>0.01 and self.disch_samples>=2 and self.n_sensor>=3):return None
        n=self.n_sensor
        vm=self.volt_sum/n; cm=self.curr_sum/n
        return {"source_file":Path(workbook).name,"local_cycle":local_cycle,
            "start_datetime":((datetime(1899,12,30)+timedelta(days=self.start_dt)).isoformat() if isinstance(self.start_dt,(float,int)) else (self.start_dt.isoformat() if hasattr(self.start_dt,"isoformat") else str(self.start_dt or ""))),
            "capacity_ah":cap,"voltage_mean":vm,"voltage_std":math.sqrt(max(0,self.volt_sq/n-vm*vm)),
            "current_mean":cm,"current_std":math.sqrt(max(0,self.curr_sq/n-cm*cm)),
            # CALCE CS2 Channel files typically have no temperature channel.
            "temperature_mean":math.nan,
            "discharge_duration_s":(self.disch_last_t-self.disch_first_t) if self.disch_last_t>self.disch_first_t else math.nan,
            "n_discharge_samples":self.disch_samples,
            "first_or_last_cycle_in_file":False}

def _parse_calce_workbook(b: bytes, path: str):
    """Stream XLSX XML with lxml; substantially faster than openpyxl for Arbin exports."""
    from lxml import etree
    NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
    tag = lambda name: f"{{{NS}}}{name}"
    with zipfile.ZipFile(io.BytesIO(b)) as wb:
        strings=[]
        if 'xl/sharedStrings.xml' in wb.namelist():
            root=etree.fromstring(wb.read('xl/sharedStrings.xml'))
            for si in root.findall(tag('si')):
                strings.append(''.join((t.text or '') for t in si.iter(tag('t'))))
        doc=etree.fromstring(wb.read('xl/workbook.xml'))
        relations=etree.fromstring(wb.read('xl/_rels/workbook.xml.rels'))
        relmap={r.get('Id'):r.get('Target') for r in relations.findall(f"{{{PKG_REL}}}Relationship")}
        sheets=[]
        for sh in doc.iter(tag('sheet')):
            if not sh.get('name','').startswith('Channel_'): continue
            target=relmap.get(sh.get(f"{{{REL}}}id"))
            if not target: continue
            location=target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target)
            sheets.append(location)
        if not sheets:raise ValueError(f"No Channel_ worksheet in {path}")
        results=[]
        for sheet_path in sheets:
            accumulator=defaultdict(CycleAccumulator)
            columns=None
            with wb.open(sheet_path) as xml:
                context=etree.iterparse(xml,events=('end',),tag=tag('row'))
                for _,row in context:
                    values={}
                    for c in row.iterchildren(tag('c')):
                        col=re.match(r'[A-Z]+',c.get('r',''))
                        if not col:continue
                        colidx=0
                        for letter in col.group():colidx=26*colidx+ord(letter)-64
                        colidx-=1
                        if columns is not None and colidx not in columns.values():continue
                        cell=c.find(tag('v'))
                        if cell is None or cell.text is None:
                            inline = c.find(tag('is'))
                            if inline is None:continue
                            v = ''.join((t.text or '') for t in inline.iter(tag('t')))
                        else:v=cell.text
                        if c.get('t')=='s':
                            try:v=strings[int(v)]
                            except (ValueError,IndexError):v=''
                        else:
                            try:v=float(v)
                            except ValueError:pass
                        values[colidx]=v
                    if columns is None:
                        columns={str(v).strip():k for k,v in values.items()}
                        for required in ('Cycle_Index','Current(A)','Voltage(V)','Discharge_Capacity(Ah)'):
                            if required not in columns:raise ValueError(f"CALCE missing {required} in {path}")
                    else:
                        def val(name):return values.get(columns.get(name,-1))
                        idx=_num(val('Cycle_Index'))
                        if math.isfinite(idx):
                            accumulator[int(idx)].add(val('Date_Time'),_num(val('Test_Time(s)')),
                                _num(val('Current(A)')),_num(val('Voltage(V)')),
                                _num(val('Discharge_Capacity(Ah)')))
                    row.clear()
                    while row.getprevious() is not None:del row.getparent()[0]
            if accumulator:
                first=min(accumulator);last=max(accumulator)
                for idx,a in accumulator.items():
                    summary=a.row(path,idx)
                    if summary is not None:
                        summary['first_or_last_cycle_in_file']=idx in (first,last)
                        results.append(summary)
        return results

def convert_calce_zips(archives:list[Path],outdir:Path,max_workbooks_per_cell:int|None=None):
    results={}
    for archive in archives:
        cell=archive.stem.upper()
        if cell not in CALCE_DEFAULT:raise ValueError(f"Unexpected CALCE archive {archive.name}; expected CS2_35..CS2_38")
        rows=[]
        with zipfile.ZipFile(archive) as z:
            members=sorted((x for x in z.namelist() if x.lower().endswith('.xlsx')))
            chosen = members[:max_workbooks_per_cell] if max_workbooks_per_cell else members
            for filename in chosen:
                rows.extend(_parse_calce_workbook(z.read(filename),filename))
        df=pd.DataFrame(rows)
        if df.empty:raise ValueError(f"No usable discharge cycles for {cell}")
        df=df.sort_values(["start_datetime","source_file","local_cycle"],kind="stable").reset_index(drop=True)
        # For overlapping exports, skip byte-identical time/capacity duplicates.
        dup=df.duplicated(subset=["start_datetime", "capacity_ah"], keep="first")
        duplicates=int(dup.sum());df=df.loc[~dup].reset_index(drop=True)
        # CALCE cells have nominal 1.1 Ah; reference from the measured first cycle
        # rather than unverified cell-specific nominal rating.
        df=_finish(df,cell)
        target=outdir / "calce" / f"{cell}.csv"
        target.parent.mkdir(parents=True,exist_ok=True)
        df.to_csv(target,index=False)
        results[cell]={"records":len(df),"workbooks_scanned":len(chosen),"duplicates_removed":duplicates,
                       "file":str(target),"capacity_first_ah":round(float(df.capacity_ah.iloc[0]),4),"capacity_last_ah":round(float(df.capacity_ah.iloc[-1]),4),
                       "missing_temperature":True,"boundary_cycle_candidates":int(df.first_or_last_cycle_in_file.sum())}
    return results

def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nasa',type=Path,help='Path to the original NASA 5.+Battery+Data+Set.zip')
    p.add_argument('--calce',type=Path,nargs='*',default=[],help='Paths to CS2_35..CS2_38.zip archives')
    p.add_argument('--out',type=Path,default=Path('data/processed'))
    p.add_argument('--nasa-cells',nargs='+',default=NASA_DEFAULT)
    p.add_argument('--extract-nasa-mat-to',type=Path,default=None,help='Optional local path for original selected .mat files (ignored by Git)')
    p.add_argument('--max-calce-workbooks-per-cell',type=int,default=None,help='For a fast loader test only; results incomplete')
    return p.parse_args()

def main():
    args=parse_args()
    if args.nasa is None and not args.calce:raise SystemExit('Specify --nasa or --calce')
    args.out.mkdir(parents=True,exist_ok=True)
    manifest={"method":"cycle-level dataset preparation, not scientific reproduction", "nasa":{},"calce":{},"source_zip_sha256":{}}
    if args.nasa:
        manifest['source_zip_sha256'][args.nasa.name]=_sha256(args.nasa)
        manifest['nasa']=convert_nasa_zip(args.nasa,args.out,[v.upper() for v in args.nasa_cells],args.extract_nasa_mat_to)
    if args.calce:
        for z in args.calce:
            manifest['source_zip_sha256'][z.name]=_sha256(z)
        manifest['calce']=convert_calce_zips(args.calce,args.out,args.max_calce_workbooks_per_cell)
    # Store locations relative to output root, not machine-specific absolute paths.
    for domain in ('nasa','calce'):
        for cell,report in manifest[domain].items():
            report['file']=str(Path(report['file']).relative_to(args.out))
    (args.out/'quality_report.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({"nasa":manifest['nasa'],"calce":manifest['calce'],"quality_report":str(args.out/'quality_report.json')},indent=2))

if __name__=='__main__':main()
