"""Rows of the T3 study: a directory of run_t3.sh CSVs (<pass>-*.csv) or a
committed archive (results/t3-*.csv.gz, with a leading 'pass' column)."""
import csv, glob, gzip, os


def rows(src, prefix=None):
    """Yields (pass, row dict); prefix: keep only this pass (A, B25, B100)."""
    if os.path.isdir(src):
        for fn in sorted(glob.glob(f'{src}/{prefix or ""}*.csv')):
            p = os.path.basename(fn).split('-')[0]
            if prefix is None or p == prefix:
                for r in csv.DictReader(open(fn)):
                    yield p, r
    else:
        with gzip.open(src, 'rt') as f:
            for r in csv.DictReader(f):
                p = r.pop('pass')
                if prefix is None or p == prefix:
                    yield p, r
