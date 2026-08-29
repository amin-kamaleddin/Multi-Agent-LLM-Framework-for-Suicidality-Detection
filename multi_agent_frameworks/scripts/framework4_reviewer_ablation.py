from __future__ import annotations

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import hashlib
import json
import platform
import textwrap

import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

SEED = 42
N_BOOTSTRAP = 5000

ALL_LABELS = ['Unsupportive', 'Indicator', 'Ideation', 'Behavior', 'Attempt']
POSITIVE_LABELS = ['Indicator', 'Ideation', 'Behavior', 'Attempt']

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_RESULTS_ROOT = PROJECT_ROOT / 'evaluation_results'
RUN_DIR = EVAL_RESULTS_ROOT / '20260802_142022_framework4_luna_full_final_run'

DATASET_FILE = RUN_DIR / 'sampled_dataset.csv'
TRACES_FILE = RUN_DIR / 'traces.jsonl'
PREDICTIONS_FILE = RUN_DIR / 'predictions.csv'
RUN_CONFIG_FILE = RUN_DIR / 'run_config.json'
RUN_MANIFEST_FILE = RUN_DIR / 'manifest.json'
METRICS_FILE = RUN_DIR / 'metrics.json'
DATASET_AUDIT_FILE = RUN_DIR / 'dataset_audit.json'
TRACE_SUMMARY_FILE = RUN_DIR / 'trace_summary.json'

OUTPUT_ROOT = RUN_DIR / 'reviewer_ablation'
TABLE_DIR = OUTPUT_ROOT / 'tables'
STATS_DIR = OUTPUT_ROOT / 'stats'
FIGURES_DIR = OUTPUT_ROOT / 'figures'
PROVENANCE_DIR = OUTPUT_ROOT / 'provenance'
for d in [OUTPUT_ROOT, TABLE_DIR, STATS_DIR, FIGURES_DIR, PROVENANCE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SELECTED_REPORT_METRICS = [
    ('Five-class accuracy', 'five_class_accuracy'),
    ('Five-class Macro F1', 'five_class_macro_f1'),
    ('Positive-only Macro F1', 'positive_only_macro_f1'),
    ('Binary sensitivity/recall', 'binary_sensitivity_recall'),
    ('Binary specificity', 'binary_specificity'),
    ('Binary precision', 'binary_precision'),
    ('Binary F1', 'binary_f1'),
    ('Indicator F1', 'indicator_f1'),
]

plt.rcParams['figure.dpi'] = 160
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalize_id(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    try:
        x = float(s)
        if x.is_integer():
            return str(int(x))
    except ValueError:
        pass
    return s


def normalize_label(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    lookup = {x.lower(): x for x in ALL_LABELS}
    return lookup.get(s.lower(), s)


def save_json(path: Path, payload):
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def multiclass_metrics(y_true, y_pred):
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=ALL_LABELS, average='macro', zero_division=0
    )
    wp, wr, wf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=ALL_LABELS, average='weighted', zero_division=0
    )
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_precision': float(mp),
        'macro_recall': float(mr),
        'macro_f1': float(mf),
        'weighted_precision': float(wp),
        'weighted_recall': float(wr),
        'weighted_f1': float(wf),
    }


def positive_only_metrics(y_true, y_pred):
    mask = np.isin(y_true, POSITIVE_LABELS)
    yt, yp = y_true[mask], y_pred[mask]
    mp, mr, mf, _ = precision_recall_fscore_support(
        yt, yp, labels=POSITIVE_LABELS, average='macro', zero_division=0
    )
    return {
        'accuracy': float(accuracy_score(yt, yp)),
        'macro_precision': float(mp),
        'macro_recall': float(mr),
        'macro_f1': float(mf),
    }


def binary_metrics(y_true, y_pred):
    yt = np.where(np.asarray(y_true) == 'Unsupportive', 0, 1)
    yp = np.where(np.asarray(y_pred) == 'Unsupportive', 0, 1)
    p, r, f1, _ = precision_recall_fscore_support(
        yt, yp, average='binary', pos_label=1, zero_division=0
    )
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = map(int, [cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]])
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        'accuracy': float(accuracy_score(yt, yp)),
        'precision': float(p),
        'sensitivity_recall': float(r),
        'specificity': float(specificity),
        'f1': float(f1),
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
    }


def per_class_metrics(y_true, y_pred):
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=ALL_LABELS, average=None, zero_division=0
    )
    return pd.DataFrame({
        'label': ALL_LABELS,
        'precision': p.astype(float),
        'recall': r.astype(float),
        'f1': f1.astype(float),
        'support': s.astype(int),
    })


def metric_bundle(y_true, y_pred):
    mc = multiclass_metrics(y_true, y_pred)
    pos = positive_only_metrics(y_true, y_pred)
    bi = binary_metrics(y_true, y_pred)
    pc = per_class_metrics(y_true, y_pred)
    out = {
        'five_class_accuracy': mc['accuracy'],
        'five_class_macro_precision': mc['macro_precision'],
        'five_class_macro_recall': mc['macro_recall'],
        'five_class_macro_f1': mc['macro_f1'],
        'positive_only_macro_f1': pos['macro_f1'],
        'binary_precision': bi['precision'],
        'binary_sensitivity_recall': bi['sensitivity_recall'],
        'binary_specificity': bi['specificity'],
        'binary_f1': bi['f1'],
    }
    for _, row in pc.iterrows():
        k = row['label'].lower()
        out[f'{k}_precision'] = float(row['precision'])
        out[f'{k}_recall'] = float(row['recall'])
        out[f'{k}_f1'] = float(row['f1'])
    return out


def paired_user_cluster_bootstrap(y_true, pre, post, users, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    users = np.asarray(users)
    unique_users = np.unique(users)
    clusters = [np.flatnonzero(users == u) for u in unique_users]
    y_true = np.asarray(y_true)
    pre = np.asarray(pre)
    post = np.asarray(post)

    observed_pre = metric_bundle(y_true, pre)
    observed_post = metric_bundle(y_true, post)
    names = list(observed_pre)
    store = {m: np.empty(n_bootstrap, dtype=float) for m in names}

    for i in range(n_bootstrap):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        idx = np.concatenate([clusters[j] for j in sampled])
        a = metric_bundle(y_true[idx], pre[idx])
        b = metric_bundle(y_true[idx], post[idx])
        for m in names:
            store[m][i] = b[m] - a[m]
    return observed_pre, observed_post, store


def summarize_dist(values):
    values = np.asarray(values, dtype=float)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {
        'bootstrap_mean_difference': float(values.mean()),
        'bootstrap_se': float(values.std(ddof=1)),
        'ci_95_lower': float(lo),
        'ci_95_upper': float(hi),
        'ci_excludes_zero': bool(lo > 0 or hi < 0),
        'iterations': int(len(values)),
    }


def make_publication_summary_table(boot_df: pd.DataFrame, out_csv: Path, out_md: Path):
    rows = []
    for label, metric in SELECTED_REPORT_METRICS:
        row = boot_df.loc[boot_df['metric'] == metric].iloc[0]
        rows.append({
            'metric': label,
            'before_reviewer': float(row['pre_review']),
            'after_reviewer': float(row['post_review']),
            'difference_post_minus_pre': float(row['observed_difference_post_minus_pre']),
            'ci_95_lower': float(row['ci_95_lower']),
            'ci_95_upper': float(row['ci_95_upper']),
            'ci_excludes_zero': bool(row['ci_excludes_zero']),
        })
    table = pd.DataFrame(rows)
    table.to_csv(out_csv, index=False)

    md_lines = [
        '# Publication Summary Table',
        '',
        '| Metric | Before reviewer | After reviewer | Difference | 95% CI |',
        '|---|---:|---:|---:|---:|',
    ]
    for _, row in table.iterrows():
        md_lines.append(
            f"| {row['metric']} | {row['before_reviewer']:.4f} | {row['after_reviewer']:.4f} | {row['difference_post_minus_pre']:+.4f} | [{row['ci_95_lower']:.4f}, {row['ci_95_upper']:.4f}] |"
        )
    out_md.write_text('\n'.join(md_lines), encoding='utf-8')
    return table


def plot_before_after_selected_metrics(summary_table: pd.DataFrame, out_path: Path):
    plot_df = summary_table.copy()
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(plot_df['before_reviewer'], y, s=45, label='Before reviewer')
    ax.scatter(plot_df['after_reviewer'], y, s=45, marker='s', label='After reviewer')

    for i, row in plot_df.iterrows():
        ax.plot([row['before_reviewer'], row['after_reviewer']], [i, i], linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels([textwrap.fill(x, 28) for x in plot_df['metric']])
    ax.set_xlabel('Metric value')
    ax.set_title('Framework 4 reviewer ablation: selected metrics before vs after')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(frameon=False, loc='lower right')
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_metric_differences_with_ci(summary_table: pd.DataFrame, out_path: Path):
    plot_df = summary_table.copy()
    y = np.arange(len(plot_df))
    x = plot_df['difference_post_minus_pre'].to_numpy(dtype=float)
    left = x - plot_df['ci_95_lower'].to_numpy(dtype=float)
    right = plot_df['ci_95_upper'].to_numpy(dtype=float) - x

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, y, xerr=[left, right], fmt='o', capsize=3)
    ax.axvline(0.0, linestyle='--', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([textwrap.fill(x, 28) for x in plot_df['metric']])
    ax.set_xlabel('Post-review minus pre-review difference')
    ax.set_title('Framework 4 reviewer effect with 95% user-cluster bootstrap CIs')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_reviewer_activity(activity: dict, out_path: Path):
    labels = [
        'Reviewed cases',
        'Label changed',
        'Wrong→correct',
        'Correct→wrong',
        'Wrong→wrong',
    ]
    values = [
        activity['review_required_count'],
        activity['label_change_count'],
        activity['wrong_to_correct_count'],
        activity['correct_to_wrong_count'],
        activity['wrong_to_different_wrong_count'],
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values)
    ax.set_ylabel('Number of records')
    ax.set_title('Framework 4 reviewer activity summary')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=15)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{value:,}',
            ha='center',
            va='bottom',
            fontsize=10,
        )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


print('\n' + '=' * 96)
print('FRAMEWORK 4 SECONDARY-REVIEWER ABLATION — FULL DATASET')
print('=' * 96)

for path in [DATASET_FILE, TRACES_FILE, PREDICTIONS_FILE]:
    if not path.exists():
        raise FileNotFoundError(path.resolve())
print('\nRequired inputs found.')

D = pd.read_csv(DATASET_FILE)
required = {'ID', 'users', 'text', 'sentiment'}
missing = required - set(D.columns)
assert not missing, f'Missing dataset columns: {sorted(missing)}'
assert D['ID'].is_unique and D['ID'].notna().all()
assert D['users'].notna().all()
D['sample_id'] = D['ID'].map(normalize_id)
D['true_label'] = D['sentiment'].map(normalize_label)
assert not (set(D['true_label']) - set(ALL_LABELS))
print(f'Dataset rows: {len(D):,}')
print(f'Unique users: {D["users"].nunique():,}')

events_by_sample = defaultdict(list)
event_counts = Counter()
with TRACES_FILE.open('r', encoding='utf-8') as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue
        e = json.loads(line)
        sid = normalize_id(e.get('sample_id'))
        if sid is None:
            raise ValueError(f'Trace line {line_no} missing sample_id')
        events_by_sample[sid].append(e)
        event_counts[e.get('event')] += 1

print('\nTrace event counts:')
for k, v in event_counts.most_common():
    print(f'- {k}: {v:,}')

rows = []
for sid, events in events_by_sample.items():
    judges = [e for e in events if e.get('event') == 'judge_complete']
    reviews = [e for e in events if e.get('event') == 'secondary_review_complete']
    finals = [e for e in events if e.get('event') == 'workflow_finalized']
    failures = [e for e in events if e.get('event') in {'node_failed', 'judge_skipped'}]
    if len(judges) > 1 or len(reviews) > 1 or len(finals) != 1:
        raise ValueError(f'Unexpected event multiplicity for sample {sid}')
    judge = judges[0] if judges else None
    review = reviews[0] if reviews else None
    final = finals[0]

    pre = normalize_label(judge.get('label_name')) if judge else None
    required_review = bool(judge.get('review_required')) if judge else False
    reviewed = normalize_label(review.get('reviewed_label')) if review else None
    final_label_raw = final.get('final_label')
    final_label = normalize_label(final_label_raw)
    review_used = bool(final.get('secondary_review_used', False))

    if required_review:
        assert review is not None, f'{sid}: review required but missing'
        assert review_used, f'{sid}: review required but not used'
    if review is not None:
        assert judge is not None
        assert normalize_label(review.get('original_judge_label')) == pre
        assert reviewed in ALL_LABELS
        assert final_label == reviewed
    if (not required_review) and judge is not None:
        assert review is None
        assert not review_used
        assert final_label == pre

    rows.append({
        'sample_id': sid,
        'pre_review_label': pre,
        'review_required': required_review,
        'judge_confidence': judge.get('confidence') if judge else None,
        'judge_competing_label': normalize_label(judge.get('competing_label')) if judge else None,
        'judge_margin': judge.get('margin') if judge else None,
        'review_reasons': json.dumps(judge.get('review_reasons', []), ensure_ascii=False) if judge else None,
        'reviewed_label': reviewed,
        'review_confidence': review.get('confidence') if review else None,
        'secondary_review_used': review_used,
        'final_label': final_label,
        'final_label_raw': final_label_raw,
        'escalation_flag': bool(final.get('escalation_flag', False)),
        'has_node_failure_or_judge_skip': bool(failures),
        'failure_events': json.dumps(failures, ensure_ascii=False),
    })

T = pd.DataFrame(rows)
assert T['sample_id'].is_unique
assert len(T) == len(D), 'Trace coverage does not match dataset rows'

full = D[['sample_id', 'ID', 'users', 'true_label', 'text']].merge(
    T, on='sample_id', how='left', validate='one_to_one'
)
valid = full['pre_review_label'].isin(ALL_LABELS) & full['final_label'].isin(ALL_LABELS)
analysis = full.loc[valid].copy().reset_index(drop=True)
excluded = full.loc[~valid].copy()

print('\nPrimary paired ablation cohort:')
print(f'- Full workflow records: {len(full):,}')
print(f'- Paired analyzable records: {len(analysis):,}')
print(f'- Excluded records: {len(excluded):,}')
if len(excluded):
    print('- Excluded IDs: ' + ', '.join(excluded['sample_id'].astype(str)))

analysis['label_changed'] = analysis['pre_review_label'] != analysis['final_label']
analysis['pre_correct'] = analysis['pre_review_label'] == analysis['true_label']
analysis['post_correct'] = analysis['final_label'] == analysis['true_label']
analysis['wrong_to_correct'] = (~analysis['pre_correct']) & analysis['post_correct']
analysis['correct_to_wrong'] = analysis['pre_correct'] & (~analysis['post_correct'])
analysis['wrong_to_different_wrong'] = (~analysis['pre_correct']) & (~analysis['post_correct']) & analysis['label_changed']

n = len(analysis)
review_n = int(analysis['review_required'].sum())
changed_n = int(analysis['label_changed'].sum())
w2c = int(analysis['wrong_to_correct'].sum())
c2w = int(analysis['correct_to_wrong'].sum())
w2w = int(analysis['wrong_to_different_wrong'].sum())
net = w2c - c2w

activity = {
    'full_workflow_records': len(full),
    'paired_analyzable_records': n,
    'excluded_records': len(excluded),
    'review_required_count': review_n,
    'review_trigger_rate_among_analyzable': review_n / n,
    'review_trigger_rate_among_all_workflow_records': review_n / len(full),
    'secondary_review_used_count': int(analysis['secondary_review_used'].sum()),
    'label_change_count': changed_n,
    'label_change_rate_among_analyzable': changed_n / n,
    'label_change_rate_among_reviewed': changed_n / review_n if review_n else 0.0,
    'wrong_to_correct_count': w2c,
    'correct_to_wrong_count': c2w,
    'wrong_to_different_wrong_count': w2w,
    'net_correct_change': net,
}

trans_all = pd.crosstab(analysis['pre_review_label'], analysis['final_label']).reindex(
    index=ALL_LABELS, columns=ALL_LABELS, fill_value=0
)
trans_reviewed = pd.crosstab(
    analysis.loc[analysis['review_required'], 'pre_review_label'],
    analysis.loc[analysis['review_required'], 'final_label'],
).reindex(index=ALL_LABELS, columns=ALL_LABELS, fill_value=0)

y_true = analysis['true_label'].to_numpy()
pre = analysis['pre_review_label'].to_numpy()
post = analysis['final_label'].to_numpy()
users = analysis['users'].to_numpy()

pre_mc, post_mc = multiclass_metrics(y_true, pre), multiclass_metrics(y_true, post)
pre_pos, post_pos = positive_only_metrics(y_true, pre), positive_only_metrics(y_true, post)
pre_bin, post_bin = binary_metrics(y_true, pre), binary_metrics(y_true, post)
pre_pc, post_pc = per_class_metrics(y_true, pre), per_class_metrics(y_true, post)

print('\n' + '=' * 96)
print('PAIRED USER-CLUSTER BOOTSTRAP')
print('=' * 96)
print(f'Replicates: {N_BOOTSTRAP:,}')
print(f'Seed: {SEED}')
print(f'Unique users in paired cohort: {len(np.unique(users)):,}')
obs_pre, obs_post, boot = paired_user_cluster_bootstrap(y_true, pre, post, users, N_BOOTSTRAP, SEED)
print('User-cluster bootstrap complete.')

boot_rows = []
for metric, dist in boot.items():
    boot_rows.append({
        'metric': metric,
        'pre_review': obs_pre[metric],
        'post_review': obs_post[metric],
        'observed_difference_post_minus_pre': obs_post[metric] - obs_pre[metric],
        **summarize_dist(dist),
    })
boot_df = pd.DataFrame(boot_rows)

overall = pd.DataFrame([
    {
        'phase': 'pre_review',
        **{f'five_class_{k}': v for k, v in pre_mc.items()},
        **{f'positive_only_{k}': v for k, v in pre_pos.items()},
        **{f'binary_{k}': v for k, v in pre_bin.items()},
    },
    {
        'phase': 'post_review',
        **{f'five_class_{k}': v for k, v in post_mc.items()},
        **{f'positive_only_{k}': v for k, v in post_pos.items()},
        **{f'binary_{k}': v for k, v in post_bin.items()},
    },
])

pc = pre_pc.rename(columns={'precision': 'pre_precision', 'recall': 'pre_recall', 'f1': 'pre_f1'})
pc = pc.merge(
    post_pc.rename(columns={'precision': 'post_precision', 'recall': 'post_recall', 'f1': 'post_f1', 'support': 'post_support'}),
    on='label',
    validate='one_to_one',
)
assert (pc['support'] == pc['post_support']).all()
pc = pc.drop(columns=['post_support'])
pc['delta_precision'] = pc['post_precision'] - pc['pre_precision']
pc['delta_recall'] = pc['post_recall'] - pc['pre_recall']
pc['delta_f1'] = pc['post_f1'] - pc['pre_f1']
for label in ALL_LABELS:
    for metric in ['precision', 'recall', 'f1']:
        key = f'{label.lower()}_{metric}'
        br = boot_df.loc[boot_df['metric'] == key].iloc[0]
        mask = pc['label'] == label
        pc.loc[mask, f'delta_{metric}_ci95_lower'] = br['ci_95_lower']
        pc.loc[mask, f'delta_{metric}_ci95_upper'] = br['ci_95_upper']
        pc.loc[mask, f'delta_{metric}_ci_excludes_zero'] = bool(br['ci_excludes_zero'])

reason_counter = Counter()
for value in analysis.loc[analysis['review_required'], 'review_reasons']:
    if pd.isna(value):
        continue
    for reason in json.loads(value):
        reason_counter[reason] += 1
reasons_df = pd.DataFrame([
    {'review_reason': k, 'count': v, 'rate_among_reviewed_cases': v / review_n if review_n else 0.0}
    for k, v in reason_counter.most_common()
])

analysis.to_csv(TABLE_DIR / 'framework4_reviewer_ablation_reconstructed_predictions.csv', index=False)
excluded.to_csv(TABLE_DIR / 'excluded_records.csv', index=False)
overall.to_csv(TABLE_DIR / 'before_after_overall_metrics.csv', index=False)
pc.to_csv(TABLE_DIR / 'before_after_per_class_metrics.csv', index=False)
trans_all.to_csv(TABLE_DIR / 'label_transition_matrix_all_cases.csv')
trans_reviewed.to_csv(TABLE_DIR / 'label_transition_matrix_reviewed_cases.csv')
reasons_df.to_csv(TABLE_DIR / 'review_reason_counts.csv', index=False)
pd.DataFrame([activity]).to_csv(TABLE_DIR / 'reviewer_activity_summary.csv', index=False)
boot_df.to_csv(STATS_DIR / 'paired_user_cluster_bootstrap_metric_differences.csv', index=False)
np.savez_compressed(STATS_DIR / 'raw_paired_user_cluster_bootstrap.npz', **boot)

summary_table = make_publication_summary_table(
    boot_df,
    TABLE_DIR / 'publication_summary_table.csv',
    TABLE_DIR / 'publication_summary_table.md',
)
plot_before_after_selected_metrics(summary_table, FIGURES_DIR / 'figure1_before_after_selected_metrics.png')
plot_metric_differences_with_ci(summary_table, FIGURES_DIR / 'figure2_metric_differences_with_ci.png')
plot_reviewer_activity(activity, FIGURES_DIR / 'figure3_reviewer_activity_summary.png')

input_paths = [
    DATASET_FILE,
    TRACES_FILE,
    PREDICTIONS_FILE,
    RUN_CONFIG_FILE,
    RUN_MANIFEST_FILE,
    METRICS_FILE,
    DATASET_AUDIT_FILE,
    TRACE_SUMMARY_FILE,
]
pd.DataFrame([
    {'path': str(p), 'exists': p.exists(), 'sha256': sha256_file(p) if p.exists() and p.is_file() else None}
    for p in input_paths
]).to_csv(PROVENANCE_DIR / 'input_provenance.csv', index=False)

provenance = {
    'analysis_name': 'Framework 4 secondary-reviewer ablation on original full-dataset run',
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'run_directory': str(RUN_DIR),
    'output_directory': str(OUTPUT_ROOT),
    'full_workflow_records': len(full),
    'paired_analyzable_records': n,
    'excluded_records': len(excluded),
    'exclusion_rule': 'Exclude records lacking either a valid five-class judge_complete.label_name or valid five-class workflow_finalized.final_label from the paired reviewer-effect analysis.',
    'known_exception': "sample_id 12714 contains node_failed and judge_skipped events and final_label='Error'; it has no valid pre-review prediction.",
    'pre_review_definition': 'judge_complete.label_name',
    'review_trigger_definition': 'judge_complete.review_required == true',
    'reviewer_recommendation_definition': 'secondary_review_complete.reviewed_label',
    'post_review_definition': 'workflow_finalized.final_label',
    'review_used_definition': 'workflow_finalized.secondary_review_used',
    'bootstrap_method': 'Paired nonparametric user-cluster bootstrap: sample unique users with replacement, include all posts for each sampled user, and use identical resamples for pre-review and post-review predictions.',
    'bootstrap_iterations': N_BOOTSTRAP,
    'random_seed': SEED,
    'confidence_interval_method': '95% percentile intervals of the paired post-review minus pre-review bootstrap difference distribution.',
    'event_counts': {str(k): int(v) for k, v in event_counts.items()},
    'labels': ALL_LABELS,
    'positive_labels': POSITIVE_LABELS,
    'python_version': platform.python_version(),
    'platform': platform.platform(),
    'numpy_version': np.__version__,
    'pandas_version': pd.__version__,
    'scipy_version': scipy.__version__,
    'sklearn_version': sklearn.__version__,
    'matplotlib_version': plt.matplotlib.__version__,
}
save_json(PROVENANCE_DIR / 'analysis_provenance.json', provenance)

master = {
    'experiment': 'Framework 4 secondary-reviewer ablation on original full-dataset run',
    'reviewer_activity': activity,
    'before_review': {'five_class': pre_mc, 'positive_only': pre_pos, 'binary': pre_bin},
    'after_review': {'five_class': post_mc, 'positive_only': post_pos, 'binary': post_bin},
    'paired_user_cluster_bootstrap': boot_df.to_dict(orient='records'),
    'per_class': pc.to_dict(orient='records'),
    'excluded_records': excluded[['sample_id', 'final_label_raw', 'has_node_failure_or_judge_skip', 'failure_events']].to_dict(orient='records'),
    'publication_figures': [
        str(FIGURES_DIR / 'figure1_before_after_selected_metrics.png'),
        str(FIGURES_DIR / 'figure2_metric_differences_with_ci.png'),
        str(FIGURES_DIR / 'figure3_reviewer_activity_summary.png'),
    ],
}
save_json(OUTPUT_ROOT / 'framework4_reviewer_ablation_results.json', master)


def br(metric: str):
    return boot_df.loc[boot_df['metric'] == metric].iloc[0]


lines = [
    '# Framework 4 Secondary-Reviewer Ablation',
    '',
    'This analysis quantifies the contribution of the secondary reviewer within the original full-dataset Framework 4 run.',
    '',
    f'The workflow contains {len(full):,} records; {n:,} have valid paired pre-review and post-review five-class predictions. {len(excluded):,} record(s) were excluded because a paired comparison could not be reconstructed.',
    '',
    '## Reviewer activity',
    '',
    f'- Reviewer triggered: **{review_n:,} / {n:,}** analyzable records ({100 * review_n / n:.2f}%)',
    f'- Final label changed: **{changed_n:,}** cases ({100 * changed_n / review_n:.2f}% of reviewed cases)',
    f'- Wrong → correct: **{w2c:,}**',
    f'- Correct → wrong: **{c2w:,}**',
    f'- Wrong → different wrong: **{w2w:,}**',
    f'- Net change in correct classifications: **{net:+d}**',
    '',
    '## Before/after performance',
    '',
    '| Metric | Before reviewer | After reviewer | Difference | 95% user-cluster bootstrap CI |',
    '|---|---:|---:|---:|---:|',
]
for label, metric in SELECTED_REPORT_METRICS:
    row = br(metric)
    lines.append(
        f"| {label} | {row['pre_review']:.4f} | {row['post_review']:.4f} | {row['observed_difference_post_minus_pre']:+.4f} | [{row['ci_95_lower']:.4f}, {row['ci_95_upper']:.4f}] |"
    )
lines += [
    '',
    '## Generated publication assets',
    '',
    '- `figures/figure1_before_after_selected_metrics.png`',
    '- `figures/figure2_metric_differences_with_ci.png`',
    '- `figures/figure3_reviewer_activity_summary.png`',
    '- `tables/publication_summary_table.csv`',
    '- `tables/publication_summary_table.md`',
    '',
    '## Method',
    '',
    f'Uncertainty was quantified using a paired user-cluster bootstrap with {N_BOOTSTRAP:,} replicates (seed {SEED}). Unique users were sampled with replacement, all posts belonging to each sampled user were included, and identical bootstrap samples were used for the pre-review and post-review predictions.',
    '',
    'Complete reconstructed predictions, exclusions, transition matrices, per-class results, bootstrap distributions, input hashes, and software metadata are saved with the analysis.',
]
(OUTPUT_ROOT / 'framework4_reviewer_ablation.md').write_text('\n'.join(lines), encoding='utf-8')

print('\n' + '=' * 96)
print('REVIEWER ACTIVITY')
print('=' * 96)
print(f'Reviewer triggered: {review_n:,} / {n:,} ({100 * review_n / n:.2f}%)')
print(f'Label changed: {changed_n:,} ({100 * changed_n / review_n:.2f}% of reviewed cases)')
print(f'Wrong -> correct: {w2c:,}')
print(f'Correct -> wrong: {c2w:,}')
print(f'Wrong -> different wrong: {w2w:,}')
print(f'Net correct change: {net:+d}')

print('\n' + '=' * 96)
print('KEY BEFORE/AFTER RESULTS')
print('=' * 96)
for _, metric in SELECTED_REPORT_METRICS:
    row = br(metric)
    print(
        f"\n{metric}\n"
        f"  Before: {row['pre_review']:.6f}\n"
        f"  After:  {row['post_review']:.6f}\n"
        f"  Delta:  {row['observed_difference_post_minus_pre']:+.6f}\n"
        f"  95% CI: [{row['ci_95_lower']:.6f}, {row['ci_95_upper']:.6f}]"
    )

print('\n' + '=' * 96)
print('FRAMEWORK 4 REVIEWER ABLATION COMPLETE')
print('=' * 96)
print(f'\nOutput directory:\n{OUTPUT_ROOT.resolve()}')
print('\nGenerated publication assets:')
print(f'- {FIGURES_DIR / "figure1_before_after_selected_metrics.png"}')
print(f'- {FIGURES_DIR / "figure2_metric_differences_with_ci.png"}')
print(f'- {FIGURES_DIR / "figure3_reviewer_activity_summary.png"}')
print(f'- {TABLE_DIR / "publication_summary_table.csv"}')
print('\nNo model training or LLM inference was performed by this script.')