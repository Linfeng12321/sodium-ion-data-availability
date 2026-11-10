"""Compute the manuscript's Appendix A field availability and rule scores.

Prepared for materials sharing from the documented definitions. This is a new
analysis utility, not the original model training implementation.
"""
import argparse
import csv
import json
from fractions import Fraction
from pathlib import Path


def read_records(path, sheet=None):
    """Read exact column names; do not impute values or change chemistry labels."""
    path = Path(path)
    if path.suffix.lower() == '.csv':
        if sheet is not None:
            raise ValueError('--sheet applies only to an Excel main input')
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            data = list(reader)
    elif path.suffix.lower() == '.xlsx':
        from openpyxl import load_workbook
        book = load_workbook(str(path), read_only=True, data_only=False)
        try:
            if sheet is None and len(book.sheetnames) != 1:
                raise ValueError('Choose --sheet for a workbook containing multiple sheets')
            if sheet is not None and sheet not in book.sheetnames:
                raise ValueError('Requested sheet is not in the workbook')
            ws = book[sheet] if sheet is not None else book[book.sheetnames[0]]
            raw = []
            for row in ws.iter_rows():
                if any(c.data_type == 'f' for c in row):
                    raise ValueError('Input contains formulas; provide a values-only export')
                raw.append([c.value for c in row])
            header = raw[0] if raw else None
            data = raw[1:]
        finally:
            book.close()
    else:
        raise ValueError('Input must be CSV or XLSX')
    if not header or any(not isinstance(h, str) or not h for h in header):
        raise ValueError('A nonempty text header is required for every column')
    if len(header) != len(set(header)):
        raise ValueError('Duplicate column names are not supported')
    if not data:
        raise ValueError('Input has no records')
    if any(len(row) != len(header) for row in data):
        raise ValueError('Each record must have the same number of fields as the header')
    return header, [dict(zip(header, row)) for row in data]


def is_available(value, markers):
    return value is not None and str(value).strip().lower() not in markers


def analyze(header, records, specification, groups, markers):
    n = len(records)
    if n != specification['expected_records']:
        raise ValueError('Expected {} records for {}; received {}'.format(
            specification['expected_records'], specification['label'], n))
    missing = sorted({c for cols in specification['fields'].values() for c in cols} - set(header))
    if missing:
        raise ValueError('Missing required columns: ' + repr(missing))
    counts = {}
    values = {}
    for item, columns in specification['fields'].items():
        counts[item] = sum(any(is_available(row[c], markers) for c in columns) for row in records)
        values[item] = Fraction(counts[item], n)
    grouped = {name: sum((values[k] for k in items), Fraction(0)) / len(items)
               for name, items in groups.items()}
    return counts, values, grouped


def rule_rows(key, label, values, grouped, groups):
    terms = {
        'R1': [values[k] for k in groups['system_context']],
        'R4': [values[k] for k in groups['protocol_resolution']]
    }
    if key == 'main':
        terms['R2'] = [values['capacity'], grouped['system_context'], grouped['protocol_resolution'], Fraction(1)]
        terms['R3'] = [values['retention'], values['milestone_anchor'], grouped['system_context'], Fraction(1)]
    rows = []
    for rule in sorted(terms):
        m = len(terms[rule])
        total = sum(terms[rule], Fraction(0))
        rows.append(dict(dataset=label, rule=rule, terms=m, score_sum=float(total),
                         complement=float(m-total), score=float(total/m), score_percent=float(100*total/m)))
    return rows


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_manuscript(coverage, rules, reference):
    """Compare computed summaries at the manuscript's displayed precision.

    Reference values are comparisons only; they never enter the calculation.
    """
    tables = {'dataset_coverage.csv': coverage, 'rule_scores.csv': rules}
    with Path(reference).open(encoding='utf-8', newline='') as handle:
        expected = list(csv.DictReader(handle))
    if not expected:
        raise ValueError('Manuscript reference has no values')
    seen = set()
    for item in expected:
        identity = (item['output'], item['dataset'], item['rule'], item['column'])
        if identity in seen:
            raise ValueError('Duplicate manuscript reference: ' + repr(identity))
        seen.add(identity)
        matches = [r for r in tables[item['output']]
                   if r['dataset'] == item['dataset'] and r.get('rule', '') == item['rule']]
        if len(matches) != 1:
            raise ValueError('Manuscript reference does not identify one computed row')
        decimals = int(item['decimals'])
        if not 0 <= decimals <= 6:
            raise ValueError('Invalid reference precision')
        actual = format(matches[0][item['column']], '.{}f'.format(decimals))
        if actual != item['value']:
            raise ValueError('Manuscript mismatch {}: calculated {}, expected {}'.format(
                identity, actual, item['value']))
    return len(expected)


def render_tables(coverage, rules):
    # Labels are fixed strings in the supplied configuration, not raw cell text.
    text = [r'\begin{tabular}{lrrrr}', r'\hline',
            r'Dataset & Records & System (0--100) & Protocol (0--100) & Response (0--100) \\', r'\hline']
    for row in coverage:
        text.append('{} & {} & {:.1f} & {:.1f} & {:.1f} \\'.format(
            row['dataset'], row['records'], row['system_context_percent'],
            row['protocol_resolution_percent'], row['response_evidence_percent']) + '\\')
    text += [r'\hline', r'\end{tabular}', r'\par\medskip\noindent', r'\begin{tabular}{llrrrr}', r'\hline',
             r'Dataset & Rule & Terms & Sum & Complement & Score (0--100) \\', r'\hline']
    for row in rules:
        text.append('{} & {} & {} & {:.3f} & {:.3f} & {:.1f} \\'.format(
            row['dataset'], row['rule'], row['terms'], row['score_sum'],
            row['complement'], row['score_percent']) + '\\')
    text += [r'\hline', r'\end{tabular}', '']
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', required=True, type=Path, help='Processed main XLSX or CSV')
    parser.add_argument('--auxiliary', required=True, type=Path, help='Processed auxiliary CSV')
    parser.add_argument('--sheet', help='Main workbook sheet, needed only for multiple sheets')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'configs/field_availability.json')
    parser.add_argument('--output', required=True, type=Path, help='New output directory; existing directories are not overwritten')
    parser.add_argument('--check-manuscript', action='store_true',
                        help='Check Table 4 and Appendix A.2 values before writing outputs')
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError('Output directory already exists; choose a new directory')
        config = json.loads(args.config.read_text(encoding='utf-8'))
        items, coverage, rules = [], [], []
        for key, path, sheet in [('main', args.main, args.sheet), ('auxiliary', args.auxiliary, None)]:
            spec = config['datasets'][key]
            header, records = read_records(path, sheet)
            counts, values, grouped = analyze(header, records, spec, config['groups'], set(config['missing_markers']))
            for item, value in values.items():
                items.append(dict(dataset=spec['label'], item=item, source_columns=json.dumps(spec['fields'][item]),
                                  available_records=counts[item], total_records=len(records), availability=float(value)))
            row = dict(dataset=spec['label'], records=len(records))
            row.update({name+'_percent': float(100*value) for name, value in grouped.items()})
            coverage.append(row)
            rules.extend(rule_rows(key, spec['label'], values, grouped, config['groups']))
        if args.check_manuscript:
            reference = Path(__file__).resolve().parents[1]/'data/manuscript_values.csv'
            count = check_manuscript(coverage, rules, reference)
            print('Matched {} manuscript values at their displayed precision.'.format(count))
        args.output.mkdir(parents=True, exist_ok=False)
        write_csv(args.output/'item_availability.csv', items)
        write_csv(args.output/'dataset_coverage.csv', coverage)
        write_csv(args.output/'rule_scores.csv', rules)
        (args.output/'availability_tables.tex').write_text(render_tables(coverage, rules), encoding='utf-8')
        print('Wrote 18 item summaries, 2 dataset summaries, 6 rule summaries and LaTeX tables.')
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, 'Input error: {}\n'.format(exc))


if __name__ == '__main__':
    main()
