"""Plot computed coverage summaries. New sharing utility, not an original paper figure."""
import argparse
import csv
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='dataset_coverage.csv from analyze_availability.py')
    parser.add_argument('--output', required=True, type=Path, help='New directory for PNG and PDF')
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError('Output directory already exists; choose a new directory')
        with args.input.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        labels = ['Main dataset', 'Auxiliary cathode dataset']
        if len(rows) != 2 or {r['dataset'] for r in rows} != set(labels):
            raise ValueError('Expected one summary for each dataset')
        rows.sort(key=lambda r: labels.index(r['dataset']))
        keys = ['system_context_percent', 'protocol_resolution_percent', 'response_evidence_percent']
        values = [[float(row[key]) for key in keys] for row in rows]
        if any(not math.isfinite(v) or not 0 <= v <= 100 for row in values for v in row):
            raise ValueError('Scores must be finite and between 0 and 100')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'pdf.fonttype': 42})
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for i, (label, scores, color) in enumerate(zip(labels, values, ['#23687C', '#D49B45'])):
            pos = [x+(i-0.5)*0.34 for x in range(3)]
            bars = ax.bar(pos, scores, width=0.31, color=color, label=label)
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x()+bar.get_width()/2, score+1.5, '{:.1f}'.format(score), ha='center', fontsize=10)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['System context', 'Protocol resolution', 'Response evidence'])
        ax.set_ylabel('Availability score (0–100)')
        ax.set_ylim(0, 115)
        ax.set_yticks(range(0, 101, 20))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=10)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#E5E8EB', linewidth=0.6)
        fig.tight_layout()
        args.output.mkdir(parents=True, exist_ok=False)
        fig.savefig(str(args.output/'coverage.png'), dpi=300, facecolor='white')
        fig.savefig(str(args.output/'coverage.pdf'), metadata={'Title': 'Information availability', 'Author': '', 'CreationDate': None, 'ModDate': None})
        plt.close(fig)
        print('Wrote coverage.png and coverage.pdf from computed dataset summaries.')
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, 'Input error: {}\n'.format(exc))


if __name__ == '__main__':
    main()
