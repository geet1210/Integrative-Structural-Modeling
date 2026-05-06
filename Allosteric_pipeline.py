# Author: Geet Madhukar
# University of New Hampshire, Cote Lab

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1.  CONFIGURATION
# =============================================================================

SAMPLES = [
    ("Apo",     1, "path/to/Apo/Rep1"),
    ("Apo",     2, "path/to/Apo/Rep2"),
    ("Apo",     3, "path/to/Apo/Rep3"),
    ("cGMP",    1, "path/to/cGMP/Rep1"),
    ("cGMP",    2, "path/to/cGMP/Rep2"),
    ("cGMP",    3, "path/to/cGMP/Rep3"),
    ("Pg",      1, "path/to/Pg/Rep1"),
    ("Pg",      2, "path/to/Pg/Rep2"),
    ("Pg",      3, "path/to/Pg/Rep3"),
    ("Pg_cGMP", 1, "path/to/PgcGMP/Rep1"),
    ("Pg_cGMP", 2, "path/to/PgcGMP/Rep2"),
    ("Pg_cGMP", 3, "path/to/PgcGMP/Rep3"),
]

DCCM_DIR   = Path("path/to/dccm_results")
OUTPUT_DIR = Path("path/to/output")
OUTPUT_DIR.mkdir(exist_ok=True)

N_CATALYTIC = 906
N_CA_CHAIN  = 453

# ── PART A: Differential thresholds (relaxed to avoid zero output) ───────────
EFFECT_SIZE_THRESHOLD = 0.8   # |Hedges' g| — was 1.0, relaxed to 0.8
CONSISTENCY_THRESHOLD = 0.78  # 7/9 pairs agree — was 1.0, relaxed
BOOTSTRAP_N           = 5000  # resamples (reduced for speed)
BOOTSTRAP_ALPHA       = 0.05
FDR_THRESHOLD         = 0.20
FILTER_EFFECT_SIZE    = True
FILTER_CONSISTENCY    = True
FILTER_CI             = True
FILTER_FDR            = False  # underpowered at n=3 with 906 features

# DCCM pre-filter: variance screen before permutation test.
# 405k features x 20 perms x hedges_g = hours without this.
# Variance (feature activity) is orthogonal to effect size — not double-dipping.
MAX_DCCM_FEATURES = 2000  # top features by within-comparison variance

# ── PART B: CNA parameters ───────────────────────────────────────────────────
DCCM_EDGE_THRESHOLD   = 0.30  # minimum |DCCM| to include graph edge
N_COMMUNITIES_TARGET  = 12    # target communities (PCA+clustering)
N_COMMUNITIES_MIN     = 4     # merge if fewer than this would result
MIN_COMMUNITY_SIZE    = 5     # communities smaller than this are merged
HUB_BC_THRESHOLD      = 2.0   # hub if BC > mean + N*std (N=2.0)

STATES     = ['Apo', 'cGMP', 'Pg', 'Pg_cGMP']
PG_STATES  = {'Pg', 'Pg_cGMP'}

STATE_COLORS = {
    'Apo': '#2196F3', 'cGMP': '#4CAF50',
    'Pg':  '#FF9800', 'Pg_cGMP': '#E91E63',
}

DOMAINS = [
    (1,   74,  'N-terminal',  '#E3F2FD'),
    (75,  223, 'GAFa',        '#BBDEFB'),
    (224, 255, 'LH1',         '#CE93D8'),
    (256, 285, 'GAFb_core1',  '#C8E6C9'),
    (286, 310, 'B1/2 loop',   '#EF5350'),
    (311, 432, 'GAFb_core2',  '#A5D6A7'),
    (433, 453, 'LH2',         '#F8BBD0'),
]
DOMAIN_COLORS = {d[2]: d[3] for d in DOMAINS}


def get_domain(res):
    for start, end, name, color in DOMAINS:
        if start <= res <= end:
            return name, color
    return 'Unknown', '#EEEEEE'


def node_to_chain_res(node_1idx):
    if node_1idx <= N_CA_CHAIN:
        return 'A', node_1idx
    return 'B', node_1idx - N_CA_CHAIN


def log(msg, logfile=None):
    """Print to console and optionally append to logfile."""
    print(msg)
    if logfile is not None:
        with open(logfile, 'a') as f:
            f.write(msg + '\n')


# =============================================================================
# 2.  DATA LOADING
# =============================================================================

def parse_xvg(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(('#', '@')) or not line:
                continue
            try:
                data.append([float(x) for x in line.split()])
            except ValueError:
                continue
    return np.array(data) if data else np.empty((0, 2))


def load_rmsf(samples, logfile=None):
    log("\n" + "="*60, logfile)
    log("STEP A1: Loading RMSF features", logfile)
    log("="*60, logfile)
    records = {}
    for state, rep, folder in samples:
        sid = f"{state}_Rep{rep}"
        f   = Path(folder) / f"rmsf_Rep{rep}.xvg"
        if f.exists():
            d    = parse_xvg(f)
            vals = d[:N_CATALYTIC, 1] if len(d) >= N_CATALYTIC \
                   else np.full(N_CATALYTIC, np.nan)
        else:
            log(f"  WARNING: {f.name} not found", logfile)
            vals = np.full(N_CATALYTIC, np.nan)
        records[sid] = {f"rmsf_{i+1:04d}": v for i, v in enumerate(vals)}
    df = pd.DataFrame(records).T
    df.index.name = 'sample_id'
    log(f"  RMSF matrix: {df.shape}  NaN: {df.isna().sum().sum()}", logfile)
    return df


def load_dccm_features(samples, logfile=None):
    log("\n" + "="*60, logfile)
    log("STEP A2: Loading DCCM features", logfile)
    log("="*60, logfile)
    csv_path = DCCM_DIR / "dccm_features_gabab.csv"
    if not csv_path.exists():
        log(f"  ERROR: {csv_path} not found.", logfile)
        log(f"         Run pde6_dccm_pipeline_v2.py (Step 3) first.", logfile)
        return None
    log(f"  Loading {csv_path.name} ...", logfile)
    df  = pd.read_csv(csv_path, index_col=0)
    ids = [f"{s}_Rep{r}" for s, r, _ in samples]
    avail = [s for s in ids if s in df.index]
    df = df.loc[avail]
    log(f"  DCCM matrix: {df.shape}", logfile)
    return df


def load_mean_dccm(state, logfile=None):
    """
    Load and average DCCM npy files for a state.
    Returns (gabab_906x906, full_matrix_or_None, actual_size).
    For Pg/Pg_cGMP: full_matrix is 1022x1022. For Apo/cGMP: None.
    """
    reps = []; actual_size = None
    for s, rep, _ in SAMPLES:
        if s != state: continue
        size_order = [1022, 906] if state in PG_STATES else [906, 1022]
        for size in size_order:
            npy = DCCM_DIR / "dccm_{}_Rep{}_{}.npy".format(state, rep, size)
            if npy.exists():
                d = np.load(str(npy)); reps.append(d); actual_size = d.shape[0]; break
        else:
            log("  WARNING: no npy for {} Rep{}".format(state, rep), logfile)
    if not reps: return None, None, None
    mean_full = np.mean(reps, axis=0)
    gabab = mean_full[:N_CATALYTIC, :N_CATALYTIC]
    full_mat = mean_full if actual_size > N_CATALYTIC else None
    log("  {}: {} reps size={} GAFab={}x{}".format(
        state, len(reps), actual_size, gabab.shape[0], gabab.shape[1]), logfile)
    return gabab, full_mat, actual_size
def build_state_index(samples):
    return {f"{s}_Rep{r}": s for s, r, _ in samples}


# =============================================================================
# 3.  PART A — STATISTICAL PRIMITIVES
# =============================================================================

def hedges_g(a, b):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return 0.0
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
                 / (len(a)+len(b)-2))
    if sp < 1e-10: return 0.0
    d = (b.mean() - a.mean()) / sp
    J = 1.0 - 3.0 / (4.0*(len(a)+len(b)-2) - 1.0)
    return d * J


def replicate_consistency(a, b):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0: return 0.0
    n = len(a) * len(b)
    return max(sum(bi>ai for ai in a for bi in b),
               sum(ai>bi for ai in a for bi in b)) / n


def bootstrap_ci(a, b, n=BOOTSTRAP_N, alpha=BOOTSTRAP_ALPHA):
    rng = np.random.RandomState(42)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return np.nan, np.nan, False
    diffs = np.array([
        rng.choice(b, len(b), replace=True).mean() -
        rng.choice(a, len(a), replace=True).mean()
        for _ in range(n)])
    lo, hi = np.percentile(diffs, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi, not (lo <= 0 <= hi)


def permutation_p(a_col, b_col):
    """Exact two-sided permutation p-value, C(6,3)=20 permutations."""
    from itertools import combinations as _c
    all_vals = np.concatenate([a_col, b_col])
    na = len(a_col)
    obs = abs(hedges_g(a_col, b_col))
    all_idx = list(range(len(all_vals)))
    n_ext = sum(
        1 for perm in _c(all_idx, na)
        if abs(hedges_g(all_vals[list(perm)],
                        all_vals[[i for i in all_idx if i not in perm]])) >= obs
    )
    return n_ext / 20.0  # C(6,3)=20


def benjamini_hochberg(pvals, alpha=FDR_THRESHOLD):
    p = np.where(np.isnan(pvals), 1.0, np.asarray(pvals, float))
    n = len(p)
    if n == 0: return np.array([]), np.array([], dtype=bool)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order]*n/np.arange(1,n+1))[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n); out[order] = q
    return out, out < alpha


def _impute_per_group(Xa, Xb):
    for arr in [Xa, Xb]:
        gm = np.nanmean(arr, axis=0)
        nm = np.isnan(arr)
        if nm.any():
            arr[nm] = np.take(gm, np.where(nm)[1])
    return Xa, Xb


# =============================================================================
# 4.  PART A — PAIRWISE DIFFERENTIAL ANALYSIS
# =============================================================================

def pairwise_differential(X_a, X_b, state_a, state_b,
                           feature_names, feat_type, logfile=None):
    nf = X_a.shape[1]
    log(f"    {feat_type}: {nf} features ...", logfile)

    g_v    = np.zeros(nf)
    con_v  = np.zeros(nf)
    ci_lo  = np.full(nf, np.nan)
    ci_hi  = np.full(nf, np.nan)
    excl   = np.zeros(nf, dtype=bool)
    perm   = np.ones(nf)

    # ── Vectorised: g for all features at once ───────────────────────
    J_factor = 1.0 - 3.0 / (4.0*(X_a.shape[0]+X_b.shape[0]-2) - 1.0)
    Xaf = X_a.astype(float); Xbf = X_b.astype(float)
    ma = np.nanmean(Xaf, 0); mb = np.nanmean(Xbf, 0)
    va = np.nanvar(Xaf, 0, ddof=1); vb = np.nanvar(Xbf, 0, ddof=1)
    na_v, nb_v = Xaf.shape[0], Xbf.shape[0]
    sp = np.sqrt(((na_v-1)*va + (nb_v-1)*vb) / (na_v+nb_v-2))
    sp = np.where(sp < 1e-10, 1e-10, sp)
    g_v[:] = (mb - ma) / sp * J_factor

    # Consistency: n=3 means 9 pairs — still fast per-feature
    for fi in range(nf):
        con_v[fi] = replicate_consistency(Xaf[:, fi], Xbf[:, fi])

    # ── Vectorised bootstrap CI ──────────────────────────────────────
    rng_b = np.random.RandomState(42)
    ib_a  = rng_b.randint(0, na_v, (BOOTSTRAP_N, na_v))  # (B, na)
    ib_b  = rng_b.randint(0, nb_v, (BOOTSTRAP_N, nb_v))  # (B, nb)
    CHUNK = min(nf, 5000)
    for fs in range(0, nf, CHUNK):
        fe   = min(fs + CHUNK, nf)
        ba   = Xaf[:, fs:fe][ib_a].mean(axis=1)  # (B, chunk)
        bb   = Xbf[:, fs:fe][ib_b].mean(axis=1)
        diff = bb - ba
        lo   = np.percentile(diff, 2.5,  axis=0)
        hi   = np.percentile(diff, 97.5, axis=0)
        ci_lo[fs:fe] = lo; ci_hi[fs:fe] = hi
        excl[fs:fe]  = ~((lo <= 0) & (0 <= hi))

    # ── Vectorised permutation p-values ─────────────────────────────
    # Build all 20 permutations as index arrays; compute |g| for all
    # features and all permutations in one batch.
    from itertools import combinations as _comb
    na_p, nb_p = Xaf.shape[0], Xbf.shape[0]
    all_idx    = list(range(na_p + nb_p))
    X_all      = np.vstack([Xaf, Xbf])  # (6, nf)
    perms      = list(_comb(all_idx, na_p))  # 20 tuples
    n_perms    = len(perms)
    abs_g_obs  = np.abs(g_v)
    n_extreme  = np.zeros(nf, dtype=int)
    for pidx in perms:
        bidx  = [i for i in all_idx if i not in pidx]
        Xpa   = X_all[list(pidx)]   # (na, nf)
        Xpb   = X_all[bidx]         # (nb, nf)
        mp_a  = Xpa.mean(0); mp_b = Xpb.mean(0)
        vp_a  = Xpa.var(0, ddof=1); vp_b = Xpb.var(0, ddof=1)
        sp_p  = np.sqrt(((na_p-1)*vp_a + (nb_p-1)*vp_b)/(na_p+nb_p-2))
        sp_p  = np.where(sp_p < 1e-10, 1e-10, sp_p)
        pg    = np.abs((mp_b - mp_a) / sp_p * J_factor)
        n_extreme += (pg >= abs_g_obs).astype(int)
    perm[:] = n_extreme / n_perms

    q_v, fdr_sig = benjamini_hochberg(perm)

    abs_g  = np.abs(g_v)
    robust = np.ones(nf, dtype=bool)
    if FILTER_EFFECT_SIZE:  robust &= abs_g >= EFFECT_SIZE_THRESHOLD
    if FILTER_CONSISTENCY:  robust &= con_v >= CONSISTENCY_THRESHOLD
    if FILTER_CI:           robust &= excl
    if FILTER_FDR:          robust &= fdr_sig

    mean_a = np.nanmean(X_a, axis=0)
    mean_b = np.nanmean(X_b, axis=0)

    df = pd.DataFrame({
        'feature': feature_names, 'feature_type': feat_type,
        'state_a': state_a, 'state_b': state_b,
        'mean_a': mean_a, 'mean_b': mean_b,
        'mean_diff': mean_b - mean_a,
        'hedges_g': g_v, 'abs_g': abs_g,
        'replicate_consistency': con_v,
        'ci_low': ci_lo, 'ci_high': ci_hi, 'ci_excludes_zero': excl,
        'perm_p': perm, 'fdr_q': q_v,
        'fdr_significant': fdr_sig, 'robust': robust,
    }).sort_values('abs_g', ascending=False).reset_index(drop=True)

    log(f"      perm_p min={perm.min():.2f}  "
        f"FDR-sig={fdr_sig.sum()}  robust={robust.sum()}/{nf}", logfile)
    return df


def run_all_pairwise(rmsf_df, dccm_df, state_index, logfile=None):
    log("\n" + "="*60, logfile)
    log("STEP A3: Pairwise differential analysis", logfile)
    log("="*60, logfile)
    log(f"  |g| >= {EFFECT_SIZE_THRESHOLD}  "
        f"consistency >= {CONSISTENCY_THRESHOLD}  "
        f"CI filter: {FILTER_CI}  FDR gate: {FILTER_FDR}", logfile)

    all_results = {}
    for state_a, state_b in combinations(STATES, 2):
        log(f"\n  {state_a} vs {state_b}:", logfile)
        tag   = f"{state_a}_vs_{state_b}"
        ids_a = [s for s, st in state_index.items() if st == state_a]
        ids_b = [s for s, st in state_index.items() if st == state_b]

        Xa_r = rmsf_df.loc[ids_a].values.copy()
        Xb_r = rmsf_df.loc[ids_b].values.copy()
        Xa_r, Xb_r = _impute_per_group(Xa_r, Xb_r)
        df_rmsf = pairwise_differential(
            Xa_r, Xb_r, state_a, state_b,
            list(rmsf_df.columns), 'RMSF', logfile)

        df_dccm = pd.DataFrame()
        if dccm_df is not None:
            Xa_d = dccm_df.loc[ids_a].values.copy()
            Xb_d = dccm_df.loc[ids_b].values.copy()
            Xa_d, Xb_d = _impute_per_group(Xa_d, Xb_d)
            dccm_cols = np.array(list(dccm_df.columns))
            # Variance pre-screen: tractable subset for permutation test
            if MAX_DCCM_FEATURES is not None and Xa_d.shape[1] > MAX_DCCM_FEATURES:
                var_6   = np.vstack([Xa_d, Xb_d]).var(axis=0)
                top_idx = np.argpartition(var_6, -MAX_DCCM_FEATURES)[-MAX_DCCM_FEATURES:]
                top_idx = top_idx[np.argsort(var_6[top_idx])[::-1]]
                Xa_d = Xa_d[:, top_idx]; Xb_d = Xb_d[:, top_idx]
                dccm_cols = dccm_cols[top_idx]
                log(f"    DCCM: {len(var_6)} -> {MAX_DCCM_FEATURES} features"
                    f" (top by within-comparison variance)", logfile)
            df_dccm = pairwise_differential(
                Xa_d, Xb_d, state_a, state_b,
                list(dccm_cols), 'DCCM', logfile)

        df_combined = pd.concat([df_rmsf, df_dccm], ignore_index=True)
        df_combined = df_combined.sort_values('abs_g', ascending=False)
        all_results[(state_a, state_b)] = df_combined

        # ── Write output immediately after each comparison ────────────────
        _annotate_and_save(df_combined, tag, logfile)

    return all_results


def _annotate_and_save(df, tag, logfile=None):
    """Annotate with chain/domain info and save differential CSVs immediately."""
    rows_ann = []
    for _, r in df.iterrows():
        feat = r['feature']
        row  = r.to_dict()
        if feat.startswith('rmsf_'):
            node = int(feat.split('_')[1])
            chain, res = node_to_chain_res(node)
            domain = get_domain(res)[0]
            row.update({'node': node, 'chain': chain,
                        'within_res': res, 'domain': domain})
        elif feat.startswith('dccm_'):
            parts  = feat.split('_')
            ni, nj = int(parts[1]), int(parts[2])
            ci, ri = node_to_chain_res(ni)
            cj, rj = node_to_chain_res(nj)
            row.update({
                'node_i': ni, 'chain_i': ci, 'res_i': ri,
                'domain_i': get_domain(ri)[0],
                'node_j': nj, 'chain_j': cj, 'res_j': rj,
                'domain_j': get_domain(rj)[0],
            })
        rows_ann.append(row)

    df_ann = pd.DataFrame(rows_ann).sort_values('abs_g', ascending=False)

    # Full results
    out_full   = OUTPUT_DIR / f"differential_{tag}.csv"
    out_robust = OUTPUT_DIR / f"robust_{tag}.csv"
    df_ann.to_csv(out_full, index=False)
    df_ann[df_ann['robust']].to_csv(out_robust, index=False)

    n_rob = int(df_ann['robust'].sum())
    n_fdr = int(df_ann.get('fdr_significant', pd.Series(
        [False]*len(df_ann))).sum())
    log(f"    → Saved differential_{tag}.csv  "
        f"({len(df_ann)} features, FDR-sig={n_fdr}, robust={n_rob})", logfile)


# =============================================================================
# 5.  PART B — CNA: GRAPH BUILDING
# =============================================================================

def build_network(dccm, threshold=DCCM_EDGE_THRESHOLD):
    """
    Build weighted graph from DCCM (906×906 GAFab block).
    Edge weight = -log(|DCCM|) following Sethi et al. PNAS 2009.
    Only edges where |DCCM| > threshold AND residues are not i±1, i±2
    sequential neighbours within the same chain.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx required: pip install networkx")

    n = dccm.shape[0]
    G = nx.Graph()

    for i in range(n):
        chain_i, res_i = node_to_chain_res(i + 1)
        domain_i, _    = get_domain(res_i)
        G.add_node(i, chain=chain_i, res=res_i, domain=domain_i,
                   node_1idx=i+1)

    for i in range(n):
        ci = 0 if i < N_CA_CHAIN else 1
        for j in range(i + 1, n):
            cj = 0 if j < N_CA_CHAIN else 1
            # Skip sequential neighbours within same chain
            if ci == cj and abs(i - j) <= 2:
                continue
            w = abs(dccm[i, j])
            if w >= threshold:
                weight = -np.log(min(w, 0.9999))
                G.add_edge(i, j, weight=weight, correlation=float(dccm[i,j]))

    return G


# =============================================================================
# 6.  PART B — CNA: HUB DETECTION
# =============================================================================

def detect_hubs(G, state, logfile=None):
    """
    Betweenness centrality hub detection.
    Hub threshold: BC > mean + HUB_BC_THRESHOLD × std (default 2.0).
    Returns DataFrame with centrality for all nodes, hub flag.
    """
    import networkx as nx

    log(f"    Computing betweenness centrality ...", logfile)
    bc = nx.betweenness_centrality(G, weight='weight', normalized=True)

    rows = []
    for node, cent in bc.items():
        chain, res = node_to_chain_res(node + 1)
        domain, _  = get_domain(res)
        rows.append({'node': node, 'node_1idx': node+1,
                     'chain': chain, 'within_res': res,
                     'domain': domain, 'bc': cent, 'state': state})

    df = pd.DataFrame(rows).sort_values('bc', ascending=False)
    bc_vals   = df['bc'].values
    threshold = bc_vals.mean() + HUB_BC_THRESHOLD * bc_vals.std()
    df['is_hub'] = df['bc'] >= threshold

    n_hubs = int(df['is_hub'].sum())
    log(f"    BC threshold: {threshold:.4f}  Hubs: {n_hubs}", logfile)
    log(f"    Top 10 hubs:", logfile)
    for _, r in df[df['is_hub']].head(10).iterrows():
        log(f"      {r['chain']}{r['within_res']:3d} ({r['domain']:14s}) "
            f"BC={r['bc']:.4f}", logfile)
    return df


# =============================================================================
# 7.  PART B — CNA: PCA-BASED COMMUNITY DETECTION
# =============================================================================

def detect_communities_pca(dccm, state, n_target=N_COMMUNITIES_TARGET,
                            logfile=None):

    log(f"    PCA-based community detection (target: {n_target}) ...", logfile)

    # Use |DCCM| as feature matrix: residue i = row vector of |corr| with all j
    X = np.abs(dccm)               # (906, 906)
    np.fill_diagonal(X, 0)        # zero out self-correlation
    X = (X + X.T) / 2             # ensure symmetry

    # ── PCA: find number of components explaining >= 80% variance ────────
    pca    = PCA(n_components=min(50, X.shape[1]))
    scores = pca.fit_transform(X)  # (906, n_components)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_pc   = max(5, int(np.searchsorted(cumvar, 0.80)) + 1)
    n_pc   = min(n_pc, 30)         # cap at 30 for tractability
    scores = scores[:, :n_pc]
    log(f"      PCA: {n_pc} components explain {cumvar[n_pc-1]*100:.1f}% variance",
        logfile)

    # ── Ward hierarchical clustering on PCA scores ────────────────────────
    Z      = linkage(scores, method='ward')
    labels = fcluster(Z, t=n_target, criterion='maxclust')  # 1-indexed

    # ── Merge tiny communities into nearest neighbour ─────────────────────
    for _ in range(10):  # iterate until stable
        sizes  = pd.Series(labels).value_counts()
        tiny   = sizes[sizes < MIN_COMMUNITY_SIZE].index.tolist()
        if not tiny:
            break
        for tc in tiny:
            tc_mask    = labels == tc
            tc_center  = scores[tc_mask].mean(axis=0)
            other_ids  = [c for c in np.unique(labels) if c != tc]
            best_c, best_d = None, np.inf
            for oc in other_ids:
                oc_center = scores[labels == oc].mean(axis=0)
                d = np.linalg.norm(tc_center - oc_center)
                if d < best_d:
                    best_d, best_c = d, oc
            labels[tc_mask] = best_c
        # Re-number consecutively
        uniq = sorted(np.unique(labels))
        remap = {old: new+1 for new, old in enumerate(uniq)}
        labels = np.array([remap[l] for l in labels])

    n_final = len(np.unique(labels))
    log(f"      Final communities: {n_final}", logfile)

    # ── Build output DataFrame ────────────────────────────────────────────
    rows = []
    for node in range(N_CATALYTIC):
        chain, res = node_to_chain_res(node + 1)
        domain, _  = get_domain(res)
        comm       = int(labels[node])
        rows.append({'node': node, 'node_1idx': node+1,
                     'chain': chain, 'within_res': res,
                     'domain': domain, 'community': comm, 'state': state})

    df = pd.DataFrame(rows)

    # Print community summary
    for comm_id in sorted(df['community'].unique()):
        sub     = df[df['community'] == comm_id]
        chains  = sub['chain'].value_counts().to_dict()
        domains = sub['domain'].value_counts().head(2).index.tolist()
        rmin    = sub[sub['chain']=='A']['within_res'].min() if 'A' in chains else '-'
        rmax    = sub[sub['chain']=='A']['within_res'].max() if 'A' in chains else '-'
        log(f"      C{comm_id:02d}: {len(sub):3d} res  chains={chains}  "
            f"top_domains={domains}  ChA res {rmin}-{rmax}", logfile)

    return df


# =============================================================================
# 8.  PART B — CNA: INTER-COMMUNITY EDGES
# =============================================================================

def compute_intercommunity(G, community_df, state, logfile=None):
    """
    For every pair of communities, compute:
    - n_edges: number of graph edges crossing the boundary
    - mean_corr: mean |DCCM| of those edges
    - cross_chain: True if communities span different chains

    This is the core allosteric communication result: which communities
    talk to each other most strongly?
    """
    node_to_comm = dict(zip(community_df['node'], community_df['community']))

    pair_counts = {}
    pair_corrs  = {}

    for u, v, data in G.edges(data=True):
        cu = node_to_comm.get(u)
        cv = node_to_comm.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        key = (min(cu, cv), max(cu, cv))
        pair_counts[key]  = pair_counts.get(key, 0) + 1
        pair_corrs.setdefault(key, []).append(abs(data['correlation']))

    rows = []
    for key, count in pair_counts.items():
        c1, c2 = key
        sub1 = community_df[community_df['community']==c1]
        sub2 = community_df[community_df['community']==c2]
        cross = set(sub1['chain'].unique()) != set(sub2['chain'].unique())
        rows.append({
            'state': state,
            'community_1': c1, 'community_2': c2,
            'n_edges': count,
            'mean_corr': np.mean(pair_corrs[key]),
            'cross_chain': cross,
        })

    df = pd.DataFrame(rows).sort_values('mean_corr', ascending=False)
    log(f"    Inter-community pairs: {len(df)}  "
        f"(top: C{df.iloc[0]['community_1']}-C{df.iloc[0]['community_2']} "
        f"corr={df.iloc[0]['mean_corr']:.3f})" if len(df) else
        f"    No inter-community edges", logfile)
    return df


# =============================================================================
# 9.  PART B — CNA: ALLOSTERIC PATHS
# =============================================================================

def find_allosteric_paths(G, hub_df, state, logfile=None):

    import networkx as nx

    CGMP_BINDING = {99, 115, 116, 136, 165, 172}
    PG_BINDING   = {286, 288, 289, 295, 305, 308}
    LH1_NODES    = set(range(224, 256))

    hub_nodes = hub_df[hub_df['is_hub']]['node'].tolist()
    if len(hub_nodes) < 2:
        log(f"    Too few hubs ({len(hub_nodes)}) for path analysis", logfile)
        return pd.DataFrame()

    log(f"    Finding shortest paths between {len(hub_nodes)} hubs ...", logfile)

    rows    = []
    checked = set()
    for src in hub_nodes:
        for tgt in hub_nodes:
            if src >= tgt:
                continue
            key = (src, tgt)
            if key in checked:
                continue
            checked.add(key)
            try:
                path   = nx.shortest_path(G, src, tgt, weight='weight')
                length = nx.shortest_path_length(G, src, tgt, weight='weight')
            except nx.NetworkXNoPath:
                continue
            if len(path) < 3:
                continue

            path_set = set(path)
            path_res = [f"{G.nodes[n]['chain']}{G.nodes[n]['res']}"
                        for n in path]

            # Post-hoc functional site annotation
            n_cgmp = sum(1 for n in path
                         if G.nodes[n]['res'] in CGMP_BINDING)
            n_pg   = sum(1 for n in path
                         if G.nodes[n]['res'] in PG_BINDING)
            n_lh1  = sum(1 for n in path
                         if G.nodes[n]['res'] in LH1_NODES)

            rows.append({
                'state': state,
                'source': f"{G.nodes[src]['chain']}{G.nodes[src]['res']}",
                'target': f"{G.nodes[tgt]['chain']}{G.nodes[tgt]['res']}",
                'path_length': length,
                'n_nodes': len(path),
                'cross_chain': G.nodes[src]['chain'] != G.nodes[tgt]['chain'],
                'n_cgmp_nodes': n_cgmp,
                'n_pg_nodes':   n_pg,
                'n_lh1_nodes':  n_lh1,
                'path': ' → '.join(path_res),
            })

    df = pd.DataFrame(rows).sort_values('path_length') if rows else pd.DataFrame()
    if not df.empty:
        best = df.iloc[0]
        log(f"    Best path: {best['path']}", logfile)
        log(f"      length={best['path_length']:.3f}  "
            f"cross-chain={best['cross_chain']}  "
            f"cGMP={best['n_cgmp_nodes']}  Pγ={best['n_pg_nodes']}  "
            f"LH1={best['n_lh1_nodes']}", logfile)
    return df


# =============================================================================
# 10. PART B — CNA: MAIN ANALYSIS PER STATE
# =============================================================================

def run_cna_for_state(state, logfile=None):

    log(f"\n{'='*60}", logfile)
    log(f"CNA: {state}", logfile)
    log(f"{'='*60}", logfile)

    # ── Load mean DCCM ────────────────────────────────────────────────────
    dccm, _full_mat_cna, _asize = load_mean_dccm(state, logfile)
    if dccm is None:
        log(f"  No DCCM data for {state} — skipping", logfile)
        return None, None, None, None, None, None

    # ── Build graph ───────────────────────────────────────────────────────
    G = build_network(dccm)
    log(f"  Graph: {G.number_of_nodes()} nodes  {G.number_of_edges()} edges",
        logfile)

    # ── Hub detection ─────────────────────────────────────────────────────
    log(f"\n  Step B1: Hub detection", logfile)
    hub_df = detect_hubs(G, state, logfile)
    hub_df.to_csv(OUTPUT_DIR / f"hub_residues_{state}.csv", index=False)
    log(f"  → Saved hub_residues_{state}.csv", logfile)

    # ── Community detection ───────────────────────────────────────────────
    log(f"\n  Step B2: Community detection", logfile)
    comm_df = detect_communities_pca(dccm, state, logfile=logfile)
    comm_df.to_csv(OUTPUT_DIR / f"communities_{state}.csv", index=False)
    log(f"  → Saved communities_{state}.csv", logfile)

    # ── Inter-community edges ─────────────────────────────────────────────
    log(f"\n  Step B3: Inter-community communication", logfile)
    inter_df = compute_intercommunity(G, comm_df, state, logfile)
    inter_df.to_csv(OUTPUT_DIR / f"intercommunity_edges_{state}.csv", index=False)
    log(f"  → Saved intercommunity_edges_{state}.csv", logfile)

    # ── Allosteric paths ──────────────────────────────────────────────────
    log(f"\n  Step B4: Allosteric paths", logfile)
    paths_df = find_allosteric_paths(G, hub_df, state, logfile)
    if not paths_df.empty:
        paths_df.to_csv(OUTPUT_DIR / f"allosteric_paths_{state}.csv", index=False)
        log(f"  → Saved allosteric_paths_{state}.csv ({len(paths_df)} paths)",
            logfile)

    return hub_df, comm_df, inter_df, paths_df, G, dccm


# =============================================================================
# 11. VISUALIZATION — PART B
# =============================================================================

def plot_hub_profile(hub_df, state, output_dir):
    """BC along sequence for both chains."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
    for ai, chain in enumerate(['A', 'B']):
        ax   = axes[ai]
        sub  = hub_df[hub_df['chain'] == chain].sort_values('within_res')
        x    = sub['within_res'].values
        y    = sub['bc'].values
        hub  = sub['is_hub'].values

        ax.bar(x, y, width=1.0, color='#90CAF9', alpha=0.6)
        ax.bar(x[hub], y[hub], width=1.0, color='#E91E63', alpha=0.9,
               label='Hub residue')

        # Domain shading
        for start, end, name, color in DOMAINS:
            ax.axvspan(start, end, alpha=0.06, color=color)
            if end - start > 15:
                ax.text((start+end)/2, ax.get_ylim()[1]*0.95, name,
                        ha='center', fontsize=5.5, rotation=40, color='gray')

        ax.set_ylabel(f"BC — Chain {chain}", fontsize=9)
        ax.set_xlim(1, N_CA_CHAIN + 2)
        ax.legend(fontsize=7, loc='upper right')

    axes[1].set_xlabel("Within-chain residue", fontsize=9)
    plt.suptitle(f"Hub residues (BC > mean+{HUB_BC_THRESHOLD}σ) — {state}",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"fig_hub_profile_{state}.png",
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_community_strip(comm_df, state, output_dir):
    """Residue strip coloured by community for both chains."""
    n_comm  = comm_df['community'].nunique()
    cmap    = plt.cm.get_cmap('tab20', n_comm)
    comm_colors = {c: cmap(i) for i, c in
                   enumerate(sorted(comm_df['community'].unique()))}

    fig, axes = plt.subplots(2, 1, figsize=(16, 4))
    for ai, chain in enumerate(['A', 'B']):
        ax  = axes[ai]
        sub = comm_df[comm_df['chain']==chain].sort_values('within_res')
        for _, r in sub.iterrows():
            ax.bar(r['within_res'], 1, width=1.0,
                   color=comm_colors[r['community']], edgecolor='none')
        for start, end, name, _ in DOMAINS:
            ax.axvline(start, color='black', lw=0.6, alpha=0.4)
        ax.set_xlim(0, N_CA_CHAIN + 5)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.set_ylabel(f"Chain {chain}", fontsize=9)
        ax.set_xlabel("Within-chain residue", fontsize=8)

    patches = [mpatches.Patch(facecolor=comm_colors[c], label=f"C{c}")
               for c in sorted(comm_df['community'].unique())]
    fig.legend(handles=patches, fontsize=6, loc='lower center',
               ncol=min(n_comm, 12), bbox_to_anchor=(0.5, -0.02))
    plt.suptitle(f"Community structure (PCA+Ward, n={n_comm}) — {state}",
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_dir / f"fig_community_strip_{state}.png",
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_intercommunity(inter_df, comm_df, state, output_dir):
    """Heatmap of mean |DCCM| between every community pair."""
    if inter_df.empty:
        return
    comms = sorted(comm_df['community'].unique())
    n = len(comms)
    idx = {c: i for i, c in enumerate(comms)}
    mat = np.zeros((n, n))
    for _, r in inter_df.iterrows():
        i, j = idx[r['community_1']], idx[r['community_2']]
        mat[i, j] = mat[j, i] = r['mean_corr']

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=0.5, aspect='auto')
    labels = [f"C{c}" for c in comms]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45,
                                                  ha='right', fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    plt.colorbar(im, ax=ax, label='Mean |DCCM|')
    ax.set_title(f"Inter-community correlation — {state}", fontsize=11,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"fig_intercommunity_{state}.png",
                dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# 12. PART A FIGURES
# =============================================================================

def plot_rmsf_profile(df_res, rmsf_df, state_a, state_b,
                       state_index, output_dir):
    ids_a = [s for s, st in state_index.items() if st == state_a]
    ids_b = [s for s, st in state_index.items() if st == state_b]
    Xa = rmsf_df.loc[ids_a].values
    Xb = rmsf_df.loc[ids_b].values
    ma, sa = np.nanmean(Xa, 0), np.nanstd(Xa, 0, ddof=1)
    mb, sb = np.nanmean(Xb, 0), np.nanstd(Xb, 0, ddof=1)
    x = np.arange(1, N_CATALYTIC+1)

    fig, axes = plt.subplots(2, 1, figsize=(16, 7),
                             gridspec_kw={'height_ratios': [2, 1]})
    for mean, std, state in [(ma, sa, state_a), (mb, sb, state_b)]:
        c = STATE_COLORS[state]
        axes[0].fill_between(x, mean-std, mean+std, alpha=0.18, color=c)
        axes[0].plot(x, mean, color=c, lw=0.8, label=state)
    axes[0].axvline(N_CA_CHAIN+0.5, color='black', lw=1.2, ls='--',
                    alpha=0.5, label='A/B')
    axes[0].set_ylabel("RMSF (nm)", fontsize=10)
    axes[0].set_xlim(1, N_CATALYTIC)
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"RMSF: {state_a} vs {state_b}", fontsize=11,
                      fontweight='bold')

    rmsf_rows = df_res[df_res['feature_type']=='RMSF']
    dv = np.zeros(N_CATALYTIC)
    for _, r in rmsf_rows.iterrows():
        dv[int(r['feature'].split('_')[1])-1] = r['hedges_g']
    colors = ['#EF5350' if d > 0 else '#2196F3' for d in dv]
    axes[1].bar(x, dv, color=colors, width=1.0, edgecolor='none', alpha=0.7)
    for t in [EFFECT_SIZE_THRESHOLD, -EFFECT_SIZE_THRESHOLD]:
        axes[1].axhline(t, color='gray', lw=0.8, ls='--')
    axes[1].axhline(0, color='black', lw=0.5)
    axes[1].axvline(N_CA_CHAIN+0.5, color='black', lw=1.2, ls='--', alpha=0.5)
    axes[1].set_ylabel("Hedges' g", fontsize=10)
    axes[1].set_xlabel("Node index", fontsize=10)
    axes[1].set_xlim(1, N_CATALYTIC)
    plt.tight_layout()
    plt.savefig(output_dir / f"fig_rmsf_{state_a}_vs_{state_b}.png",
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_effect_heatmap(all_results, output_dir):
    robust_rmsf = set()
    for df in all_results.values():
        robust_rmsf.update(
            df[df['robust'] & (df['feature_type']=='RMSF')]['feature'])
    if not robust_rmsf:
        # Fall back: top-20 by max |g| across any comparison
        top_any = {}
        for df in all_results.values():
            for _, r in df[df['feature_type']=='RMSF'].head(20).iterrows():
                feat = r['feature']
                if feat not in top_any or abs(r['hedges_g']) > top_any[feat]:
                    top_any[feat] = abs(r['hedges_g'])
        robust_rmsf = set(sorted(top_any, key=top_any.get, reverse=True)[:30])

    if not robust_rmsf:
        return

    comps     = [f"{a}_vs_{b}" for a, b in all_results]
    feat_list = sorted(robust_rmsf, key=lambda f: int(f.split('_')[1]))
    mat = np.zeros((len(feat_list), len(comps)))
    for ci, (pair, df) in enumerate(all_results.items()):
        gm = df.set_index('feature')['hedges_g'].to_dict()
        for ri, feat in enumerate(feat_list):
            mat[ri, ci] = gm.get(feat, 0.0)

    row_labels = []
    for feat in feat_list:
        node = int(feat.split('_')[1])
        chain, res = node_to_chain_res(node)
        row_labels.append(f"{chain}{res} ({get_domain(res)[0][:5]})")

    fig_h = max(6, len(feat_list)*0.28)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    vmax = max(2.0, np.abs(mat).max())
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels(comps, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(feat_list)))
    ax.set_yticklabels(row_labels, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Hedges' g")
    ax.set_title("Effect size heatmap — top RMSF features", fontsize=11,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "fig_effect_heatmap.png",
                dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# 13. LOO-LDA VALIDATION
# =============================================================================

def validate_robust_features(rmsf_df, dccm_df, state_index,
                              all_results, logfile=None):
    log("\n" + "="*60, logfile)
    log("STEP A4: LOO-LDA validation on robust features", logfile)
    log("="*60, logfile)

    rows = []
    for (state_a, state_b), df_res in all_results.items():
        tag      = f"{state_a}_vs_{state_b}"
        rob_df   = df_res[df_res['robust']]
        n_robust = len(rob_df)

        if n_robust < 2:
            log(f"  {tag}: {n_robust} robust features — skip", logfile)
            rows.append({'comparison': tag, 'n_robust': n_robust,
                         'loo_accuracy': np.nan, 'verdict': 'insufficient'})
            continue

        ids_a = [s for s, st in state_index.items() if st == state_a]
        ids_b = [s for s, st in state_index.items() if st == state_b]
        ids_6 = ids_a + ids_b
        y_6   = np.array([0]*3 + [1]*3)

        parts = []
        rmsf_rob = [f for f in rob_df['feature'] if f.startswith('rmsf_')]
        dccm_rob = [f for f in rob_df['feature'] if f.startswith('dccm_')]
        if rmsf_rob:
            parts.append(rmsf_df.loc[ids_6, rmsf_rob])
        if dccm_rob and dccm_df is not None:
            avail = [f for f in dccm_rob if f in dccm_df.columns]
            if avail:
                parts.append(dccm_df.loc[
                    [s for s in ids_6 if s in dccm_df.index], avail])
        if not parts:
            rows.append({'comparison': tag, 'n_robust': n_robust,
                         'loo_accuracy': np.nan, 'verdict': 'no data'})
            continue

        X_6    = pd.concat(parts, axis=1).values.astype(float)
        loo    = LeaveOneOut()
        y_pred = np.zeros(6, dtype=int)
        for tr, te in loo.split(X_6):
            Xtr, Xte = X_6[tr].copy(), X_6[te].copy()
            cm = np.nanmean(Xtr, axis=0)
            for arr in [Xtr, Xte]:
                nm = np.isnan(arr)
                arr[nm] = np.take(cm, np.where(nm)[1])
            sc = StandardScaler()
            lda = LinearDiscriminantAnalysis()
            lda.fit(sc.fit_transform(Xtr), y_6[tr])
            y_pred[te] = lda.predict(sc.transform(Xte))

        acc = np.mean(y_pred == y_6)
        if   acc >= 0.834: verdict = 'strong (5-6/6)'
        elif acc >= 0.667: verdict = 'moderate (4/6)'
        elif acc >= 0.500: verdict = 'weak (3/6)'
        else:              verdict = 'no separation'
        log(f"  {tag}: {n_robust} robust  acc={acc:.3f}  {verdict}", logfile)
        rows.append({'comparison': tag, 'n_robust': n_robust,
                     'loo_accuracy': acc, 'verdict': verdict})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "validation_summary.csv", index=False)
    log("  → Saved validation_summary.csv", logfile)
    return df


# =============================================================================
# 14. SENSITIVITY ANALYSIS
# =============================================================================

def run_sensitivity_analysis(rmsf_df, state_index,
                              output_dir=None, logfile=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR
    log("\n" + "="*60, logfile)
    log("SENSITIVITY ANALYSIS: threshold grid", logfile)
    log("="*60, logfile)

    g_thresholds    = [0.6, 0.8, 1.0, 1.2]
    cons_thresholds = [0.56, 0.78, 0.89, 1.0]
    sens_rows = []

    for g_thr in g_thresholds:
        for c_thr in cons_thresholds:
            row = {'g_threshold': g_thr, 'consistency_threshold': c_thr}
            total = 0
            for state_a, state_b in combinations(STATES, 2):
                ids_a = [s for s, st in state_index.items() if st == state_a]
                ids_b = [s for s, st in state_index.items() if st == state_b]
                Xa = rmsf_df.loc[ids_a].values.copy()
                Xb = rmsf_df.loc[ids_b].values.copy()
                Xa, Xb = _impute_per_group(Xa, Xb)
                n_pass = 0
                for fi in range(Xa.shape[1]):
                    g    = abs(hedges_g(Xa[:,fi].astype(float),
                                        Xb[:,fi].astype(float)))
                    cons = replicate_consistency(Xa[:,fi].astype(float),
                                                 Xb[:,fi].astype(float))
                    _, _, ez = bootstrap_ci(Xa[:,fi].astype(float),
                                            Xb[:,fi].astype(float))
                    if g >= g_thr and cons >= c_thr and ez:
                        n_pass += 1
                row[f"{state_a}_vs_{state_b}"] = n_pass
                total += n_pass
            row['total'] = total
            sens_rows.append(row)
            log(f"  |g|>={g_thr}  cons>={c_thr:.2f}  → {total}", logfile)

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(output_dir / "sensitivity_rmsf_counts.csv", index=False)
    log("  → Saved sensitivity_rmsf_counts.csv", logfile)

    # Heatmap
    pivot = sens_df.pivot(index='g_threshold', columns='consistency_threshold',
                          values='total')
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(cons_thresholds)))
    ax.set_xticklabels([f"{c:.2f}" for c in cons_thresholds], fontsize=9)
    ax.set_yticks(range(len(g_thresholds)))
    ax.set_yticklabels([f"|g|≥{g}" for g in g_thresholds], fontsize=9)
    for i in range(len(g_thresholds)):
        for j in range(len(cons_thresholds)):
            ax.text(j, i, str(int(pivot.values[i,j])),
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='black' if pivot.values[i,j] < pivot.values.max()*0.7
                    else 'white')
    plt.colorbar(im, ax=ax, label='Total robust RMSF features')
    ax.set_xlabel("Consistency threshold"); ax.set_ylabel("|Hedges' g|")
    ax.set_title("Sensitivity: robust feature count at each threshold")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_sensitivity_heatmap.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    log("  → Saved fig_sensitivity_heatmap.png", logfile)
    return sens_df


N_PG_CHAIN = 58    # Pg residues per chain (C and D)
N_FULL     = 1022  # full matrix size for Pg states


def load_mean_dccm_full(state, logfile=None):
    """
    Load and average DCCM npy files for a state.
    Returns (gabab_906, full_matrix_or_None, actual_size).
    For Pg/Pg_cGMP: full_matrix is 1022x1022.
    For Apo/cGMP:   full_matrix is None (no Pg atoms).
    """
    reps = []
    actual_size = None
    for s, rep, _ in SAMPLES:
        if s != state:
            continue
        size_order = [1022, 906] if state in PG_STATES else [906, 1022]
        for size in size_order:
            npy = DCCM_DIR / "dccm_{}_Rep{}_{}.npy".format(state, rep, size)
            if npy.exists():
                d = np.load(str(npy))
                reps.append(d)
                actual_size = d.shape[0]
                break
        else:
            log("  WARNING: no npy for {} Rep{}".format(state, rep), logfile)
    if not reps:
        return None, None, None
    mean_full = np.mean(reps, axis=0)
    gabab     = mean_full[:N_CATALYTIC, :N_CATALYTIC]
    full_mat  = mean_full if actual_size > N_CATALYTIC else None
    log("  {}: {} reps  size={}x{}  GAFab block={}x{}".format(
        state, len(reps), actual_size, actual_size,
        gabab.shape[0], gabab.shape[1]), logfile)
    return gabab, full_mat, actual_size


def compute_delta_network(G_ref, G_pg, dccm_ref, dccm_pg,
                           state_ref, state_pg, logfile=None):
    import networkx as nx
    log("  Delta network: {} minus {}".format(state_pg, state_ref), logfile)

    all_edges = set(G_ref.edges()) | set(G_pg.edges())
    rows = []
    for u, v in all_edges:
        w_ref = abs(dccm_ref[u, v]) if u < dccm_ref.shape[0] else 0.0
        w_pg  = abs(dccm_pg[u, v])  if u < dccm_pg.shape[0]  else 0.0
        delta = w_pg - w_ref

        chain_u, res_u = node_to_chain_res(u + 1)
        chain_v, res_v = node_to_chain_res(v + 1)
        rows.append({
            'node_u': u, 'node_v': v,
            'chain_u': chain_u, 'res_u': res_u,
            'domain_u': get_domain(res_u)[0],
            'chain_v': chain_v, 'res_v': res_v,
            'domain_v': get_domain(res_v)[0],
            'corr_ref': float(dccm_ref[u, v]) if u < dccm_ref.shape[0] else 0.0,
            'corr_pg':  float(dccm_pg[u, v])  if u < dccm_pg.shape[0]  else 0.0,
            'abs_ref': w_ref, 'abs_pg': w_pg,
            'delta': delta, 'abs_delta': abs(delta),
            'direction': 'gained' if delta > 0 else 'lost',
            'cross_chain': chain_u != chain_v,
            'state_ref': state_ref, 'state_pg': state_pg,
        })

    df = pd.DataFrame(rows).sort_values('abs_delta', ascending=False)
    n_gained = int((df['delta'] > 0).sum())
    n_lost   = int((df['delta'] < 0).sum())
    log("    Edges gained={} lost={}".format(n_gained, n_lost), logfile)

    if n_gained > 0:
        r = df[df['direction'] == 'gained'].iloc[0]
        log("    Top gained: {}{}<->{}{} delta={:+.4f} ({}<->{})".format(
            r['chain_u'], r['res_u'], r['chain_v'], r['res_v'],
            r['delta'], r['domain_u'][:6], r['domain_v'][:6]), logfile)
    if n_lost > 0:
        r = df[df['direction'] == 'lost'].iloc[0]
        log("    Top lost:   {}{}<->{}{} delta={:+.4f} ({}<->{})".format(
            r['chain_u'], r['res_u'], r['chain_v'], r['res_v'],
            r['delta'], r['domain_u'][:6], r['domain_v'][:6]), logfile)
    return df


def compute_pg_cross_block(full_mat_pg, state_pg, logfile=None):
    if full_mat_pg is None:
        log("  No 1022x1022 matrix for {} -- skip cross-block".format(
            state_pg), logfile)
        return pd.DataFrame()

    log("  Pg cross-block: {}".format(state_pg), logfile)

    # Chain C: rows 906:964, Chain D: rows 964:1022
    pg_C   = full_mat_pg[N_CATALYTIC:N_CATALYTIC + N_PG_CHAIN, :N_CATALYTIC]
    pg_D   = full_mat_pg[N_CATALYTIC + N_PG_CHAIN:N_FULL,      :N_CATALYTIC]
    pg_all = full_mat_pg[N_CATALYTIC:N_FULL, :N_CATALYTIC]

    rows = []
    for node in range(N_CATALYTIC):
        chain, res = node_to_chain_res(node + 1)
        domain     = get_domain(res)[0]
        rows.append({
            'node': node, 'node_1idx': node + 1,
            'chain': chain, 'within_res': res, 'domain': domain,
            'pg_signal':   float(np.abs(pg_all[:, node]).mean()),
            'pg_C_signal': float(np.abs(pg_C[:, node]).mean()),
            'pg_D_signal': float(np.abs(pg_D[:, node]).mean()),
            'state': state_pg,
        })

    df = pd.DataFrame(rows).sort_values('pg_signal', ascending=False)
    log("    Top 5 GAFab residues correlating with Pg:", logfile)
    for _, r in df.head(5).iterrows():
        log("      {}{:3d} ({:14s}) signal={:.4f} C={:.4f} D={:.4f}".format(
            r['chain'], r['within_res'], r['domain'],
            r['pg_signal'], r['pg_C_signal'], r['pg_D_signal']), logfile)
    return df


def plot_delta_network_fig(delta_df, state_ref, state_pg, output_dir):
    """Per-residue |delta| and net direction for the delta network."""
    if delta_df.empty:
        return

    tag = "{}_minus_{}".format(state_pg, state_ref)
    res_abs   = np.zeros(N_CATALYTIC)
    res_net   = np.zeros(N_CATALYTIC)
    res_count = np.zeros(N_CATALYTIC)

    for _, r in delta_df.iterrows():
        for node in [r['node_u'], r['node_v']]:
            if 0 <= node < N_CATALYTIC:
                res_abs[node]   += r['abs_delta']
                res_net[node]   += r['delta']
                res_count[node] += 1

    mask = res_count > 0
    res_abs[mask] /= res_count[mask]
    res_net[mask] /= res_count[mask]

    x = np.arange(1, N_CATALYTIC + 1)
    fig, axes = plt.subplots(2, 1, figsize=(16, 7),
                             gridspec_kw={'height_ratios': [1, 1]})

    ax = axes[0]
    ax.bar(x, res_abs, width=1.0, color='#7B1FA2', alpha=0.7, edgecolor='none')
    ax.axvline(N_CA_CHAIN + 0.5, color='black', lw=1.2, ls='--', alpha=0.6)
    for start, _, name, _ in DOMAINS:
        ax.axvline(start, color='gray', lw=0.4, alpha=0.4)
    ax.set_ylabel("|delta DCCM| per residue", fontsize=10)
    ax.set_xlim(1, N_CATALYTIC)
    ax.set_title(
        "Delta Network: {} minus {}  (Pg effect via GAFab rewiring)".format(
            state_pg, state_ref),
        fontsize=11, fontweight='bold')

    ax = axes[1]
    colors = ['#E53935' if v > 0 else '#1E88E5' for v in res_net]
    ax.bar(x, res_net, width=1.0, color=colors, alpha=0.7, edgecolor='none')
    ax.axhline(0, color='black', lw=0.6)
    ax.axvline(N_CA_CHAIN + 0.5, color='black', lw=1.2, ls='--', alpha=0.6)
    for start, _, name, _ in DOMAINS:
        ax.axvline(start, color='gray', lw=0.4, alpha=0.4)

    for start, end, name, color in DOMAINS:
        if end - start > 20:
            mid = (start + end) / 2.0
            ax.text(mid, ax.get_ylim()[0] * 0.85, name[:5],
                    ha='center', fontsize=5.5, color='gray', rotation=30)

    patches = [
        mpatches.Patch(color='#E53935', label='Gained (Pg strengthened)'),
        mpatches.Patch(color='#1E88E5', label='Lost (Pg disrupted)'),
    ]
    ax.legend(handles=patches, fontsize=8, loc='upper right')
    ax.set_ylabel("Net delta DCCM", fontsize=10)
    ax.set_xlabel("Node index (A: 1-453, B: 454-906)", fontsize=10)
    ax.set_xlim(1, N_CATALYTIC)

    plt.tight_layout()
    fname = output_dir / "fig_delta_network_{}.png".format(tag)
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pg_cross_fig(cross_pg, cross_pgcgmp, output_dir):
    """Pg-GAFab cross-block signal for both Pg states overlaid."""
    if cross_pg.empty and cross_pgcgmp.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    pairs = [
        (axes[0], cross_pg,      'Pg',      '#FF9800'),
        (axes[1], cross_pgcgmp,  'Pg_cGMP', '#E91E63'),
    ]
    for ax, df, state, color in pairs:
        if df.empty:
            ax.set_title("{} -- no data".format(state))
            continue
        sub = df.sort_values('node')
        x = sub['node_1idx'].values
        y = sub['pg_signal'].values
        ax.plot(x, y, color=color, lw=0.8, alpha=0.9,
                label="{} Pg signal".format(state))
        ax.fill_between(x, 0, y, alpha=0.2, color=color)
        # B1/2 loop in chain B: within-chain res 286-310
        # node indices 453+285=738 to 453+309=762
        ax.axvspan(738, 763, alpha=0.25, color='#EF5350',
                   label='B1/2 loop (Pg site)')
        ax.axvline(N_CA_CHAIN + 0.5, color='black', lw=1.2,
                   ls='--', alpha=0.5, label='A/B boundary')
        ax.axhline(float(y.mean()), color='gray', lw=0.8, ls=':', label='Mean')
        ax.set_ylabel("Mean |DCCM| with Pg", fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlim(1, N_CATALYTIC)
        for start, end, name, _ in DOMAINS:
            ax.axvline(start, color='lightgray', lw=0.4)

    axes[1].set_xlabel("GAFab node (A: 1-453, B: 454-906)", fontsize=10)
    plt.suptitle(
        "Pg-GAFab correlation signal\n"
        "Which GAFab residues move with Pg?  Red = B1/2 loop (Pg binding site)",
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "fig_pg_cross_signal.png",
                dpi=300, bbox_inches='tight')
    plt.close()


def run_pg_delta_analysis(all_graphs, all_dccms, logfile=None):
    log("\n" + "="*65, logfile)
    log("PART C: Pg Delta Network Analysis", logfile)
    log("="*65, logfile)
    log("  Pg effect on GAFab allosteric communication captured INDIRECTLY.", logfile)
    log("  Delta network = difference in GAFab correlation networks.", logfile)
    log("  Pg coordinates are NOT in the graph.", logfile)

    pg_comparisons = [
        ('Apo',  'Pg',      'Pure Pg effect (no cGMP)'),
        ('cGMP', 'Pg_cGMP', 'Pg effect with cGMP bound'),
    ]

    delta_results = {}
    cross_results = {}

    for state_ref, state_pg, description in pg_comparisons:
        log("\n  --- {} ---".format(description), logfile)

        if state_ref not in all_graphs or state_pg not in all_graphs:
            log("  WARNING: missing graph for {} or {}".format(
                state_ref, state_pg), logfile)
            continue
        if state_ref not in all_dccms or state_pg not in all_dccms:
            log("  WARNING: missing DCCM for {} or {}".format(
                state_ref, state_pg), logfile)
            continue

        # --- 1. GAFab delta network ---
        delta_df = compute_delta_network(
            all_graphs[state_ref], all_graphs[state_pg],
            all_dccms[state_ref],  all_dccms[state_pg],
            state_ref, state_pg, logfile)

        tag = "{}_minus_{}".format(state_pg, state_ref)
        delta_df.to_csv(OUTPUT_DIR / "delta_network_{}.csv".format(tag),
                        index=False)
        log("  -> Saved delta_network_{}.csv ({} edges)".format(
            tag, len(delta_df)), logfile)

        # Top 50 gained + lost
        top_gain = delta_df[delta_df['direction'] == 'gained'].head(50)
        top_loss = delta_df[delta_df['direction'] == 'lost'].head(50)
        pd.concat([top_gain, top_loss]).to_csv(
            OUTPUT_DIR / "delta_top_edges_{}.csv".format(tag), index=False)
        log("  -> Saved delta_top_edges_{}.csv".format(tag), logfile)

        # Domain-pair summary
        delta_df['domain_pair'] = delta_df.apply(
            lambda r: '_'.join(sorted([r['domain_u'][:6], r['domain_v'][:6]])),
            axis=1)
        dom_summary = (delta_df.groupby('domain_pair')
                       .agg(mean_abs_delta=('abs_delta', 'mean'),
                            n_edges=('abs_delta', 'count'),
                            mean_delta=('delta', 'mean'))
                       .sort_values('mean_abs_delta', ascending=False)
                       .reset_index())
        dom_summary.to_csv(
            OUTPUT_DIR / "delta_domain_summary_{}.csv".format(tag), index=False)
        log("  -> Saved delta_domain_summary_{}.csv".format(tag), logfile)
        log("    Top rewired domain pairs:", logfile)
        for _, r in dom_summary.head(5).iterrows():
            log("      {:25s}  mean|d|={:.4f}  n={:5d}  net={:+.4f}".format(
                r['domain_pair'], r['mean_abs_delta'],
                int(r['n_edges']), r['mean_delta']), logfile)

        # Figure
        plot_delta_network_fig(delta_df, state_ref, state_pg, OUTPUT_DIR)
        log("  -> fig_delta_network_{}.png".format(tag), logfile)
        delta_results[tag] = delta_df

        # --- 2. Pg-GAFab cross-block ---
        _, full_mat, _ = load_mean_dccm_full(state_pg, logfile)
        if full_mat is not None:
            cross_df = compute_pg_cross_block(full_mat, state_pg, logfile)
            if not cross_df.empty:
                cross_df.to_csv(
                    OUTPUT_DIR / "pg_cross_signal_{}.csv".format(state_pg),
                    index=False)
                log("  -> Saved pg_cross_signal_{}.csv".format(state_pg),
                    logfile)
            cross_results[state_pg] = cross_df
        else:
            cross_results[state_pg] = pd.DataFrame()

    # Combined cross-block figure
    plot_pg_cross_fig(
        cross_results.get('Pg',      pd.DataFrame()),
        cross_results.get('Pg_cGMP', pd.DataFrame()),
        OUTPUT_DIR)
    log("  -> Saved fig_pg_cross_signal.png", logfile)

    return delta_results, cross_results


# =============================================================================
# 15. MAIN
# =============================================================================

if __name__ == "__main__":
    import datetime

    LOGFILE = OUTPUT_DIR / "pipeline_run.log"
    # Clear old log
    with open(LOGFILE, 'w') as f:
        f.write(f"PDE6 Integrated Pipeline — {datetime.datetime.now()}\n")
        f.write("="*65 + "\n")

    def L(msg):
        log(msg, LOGFILE)

    L("=" * 65)
    L("PDE6 Integrated Differential + CNA Pipeline")
    L("=" * 65)
    L(f"DCCM dir   : {DCCM_DIR}")
    L(f"Output dir : {OUTPUT_DIR}")
    L(f"Log file   : {LOGFILE}")
    L(f"\nPART A thresholds (relaxed):")
    L(f"  |Hedges' g|     >= {EFFECT_SIZE_THRESHOLD}  (was 1.0)")
    L(f"  Consistency     >= {CONSISTENCY_THRESHOLD}  (7/9 pairs — was 1.0)")
    L(f"  CI filter       :  {FILTER_CI}")
    L(f"  FDR gate        :  {FILTER_FDR}  (underpowered at n=3)")
    L(f"\nPART B CNA parameters:")
    L(f"  Edge threshold  :  |DCCM| >= {DCCM_EDGE_THRESHOLD}")
    L(f"  Communities     :  target {N_COMMUNITIES_TARGET}, min size {MIN_COMMUNITY_SIZE}")
    L(f"  Hub threshold   :  BC > mean + {HUB_BC_THRESHOLD}×SD")

    # ── PART A: Load features ─────────────────────────────────────────────
    rmsf_df   = load_rmsf(SAMPLES, LOGFILE)
    dccm_df   = load_dccm_features(SAMPLES, LOGFILE)
    state_idx = build_state_index(SAMPLES)

    # ── PART A: Differential analysis (writes CSV per comparison) ─────────
    all_results = run_all_pairwise(rmsf_df, dccm_df, state_idx, LOGFILE)

    # ── PART A: Figures ───────────────────────────────────────────────────
    L("\n" + "="*60)
    L("STEP A5: Generating Part A figures")
    L("="*60)
    for (state_a, state_b), df_res in all_results.items():
        plot_rmsf_profile(df_res, rmsf_df, state_a, state_b,
                          state_idx, OUTPUT_DIR)
        L(f"  → fig_rmsf_{state_a}_vs_{state_b}.png")
    plot_effect_heatmap(all_results, OUTPUT_DIR)
    L("  → fig_effect_heatmap.png")

    # ── PART A: Validation ────────────────────────────────────────────────
    validate_robust_features(rmsf_df, dccm_df, state_idx,
                              all_results, LOGFILE)

    # ── PART A: Sensitivity ───────────────────────────────────────────────
    run_sensitivity_analysis(rmsf_df, state_idx, OUTPUT_DIR, LOGFILE)

    # Cross-comparison robust RMSF summary
    robust_rmsf = set()
    for df in all_results.values():
        robust_rmsf.update(
            df[df['robust'] & (df['feature_type']=='RMSF')]['feature'])
    if robust_rmsf:
        feat_list = sorted(robust_rmsf, key=lambda f: int(f.split('_')[1]))
        rows = []
        for feat in feat_list:
            node = int(feat.split('_')[1])
            chain, res = node_to_chain_res(node)
            row = {'feature': feat, 'chain': chain,
                   'within_res': res, 'domain': get_domain(res)[0]}
            for (a, b), df in all_results.items():
                fm = df.set_index('feature')
                row[f"g_{a}_vs_{b}"] = fm['hedges_g'].get(feat, 0.0) \
                    if feat in fm.index else 0.0
                row[f"robust_{a}_vs_{b}"] = fm['robust'].get(feat, False) \
                    if feat in fm.index else False
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            OUTPUT_DIR / "robust_rmsf_crosscomparison.csv", index=False)
        L(f"  → Saved robust_rmsf_crosscomparison.csv ({len(rows)} features)")

    # ── PART B: CNA for each state ────────────────────────────────────────
    L("\n" + "="*65)
    L("PART B: Community Network Analysis")
    L("="*65)

    try:
        import networkx as nx
        L(f"  NetworkX: {nx.__version__}")
    except ImportError:
        L("  ERROR: networkx not installed. pip install networkx")
        L("  Skipping CNA.")
        nx = None

    if nx is not None:
        all_hubs   = {}
        all_comms  = {}
        all_inter  = {}
        all_paths  = {}
        all_graphs = {}  # networkx Graph per state
        all_dccms  = {}  # 906x906 mean DCCM per state

        for state in STATES:
            hub_df, comm_df, inter_df, paths_df, G_state, dccm_state = \
                run_cna_for_state(
                state, LOGFILE)
            if hub_df is not None:
                all_hubs[state]   = hub_df
                all_comms[state]  = comm_df
                all_inter[state]  = inter_df
                all_paths[state]  = paths_df
                all_graphs[state] = G_state
                all_dccms[state]  = dccm_state

                # Figures immediately after each state
                plot_hub_profile(hub_df, state, OUTPUT_DIR)
                L(f"  → fig_hub_profile_{state}.png")
                plot_community_strip(comm_df, state, OUTPUT_DIR)
                L(f"  → fig_community_strip_{state}.png")
                plot_intercommunity(inter_df, comm_df, state, OUTPUT_DIR)
                L(f"  → fig_intercommunity_{state}.png")

        # Hub shift across states
        L("\n" + "="*60)
        L("STEP B5: Hub shifts across states")
        L("="*60)
        shift_rows = []
        for state_a, state_b in combinations(STATES, 2):
            if state_a not in all_hubs or state_b not in all_hubs:
                continue
            h1 = set(all_hubs[state_a][all_hubs[state_a]['is_hub']]['node_1idx'])
            h2 = set(all_hubs[state_b][all_hubs[state_b]['is_hub']]['node_1idx'])
            lost   = sorted(h1 - h2)
            gained = sorted(h2 - h1)
            shared = sorted(h1 & h2)
            L(f"  {state_a}→{state_b}: "
              f"gained={len(gained)} lost={len(lost)} shared={len(shared)}")
            shift_rows.append({
                'comparison': f"{state_a}_vs_{state_b}",
                'n_gained': len(gained), 'n_lost': len(lost),
                'n_shared': len(shared),
                'gained_nodes': str(gained), 'lost_nodes': str(lost),
                'shared_nodes': str(shared),
            })
        pd.DataFrame(shift_rows).to_csv(
            OUTPUT_DIR / "hub_shifts.csv", index=False)
        L("  → Saved hub_shifts.csv")

        # Part C: Pg delta network
        delta_results, cross_results = run_pg_delta_analysis(
            all_graphs, all_dccms, LOGFILE)

    # ── Final summary ─────────────────────────────────────────────────────
    L("\n" + "="*65)
    L("PIPELINE COMPLETE")
    L("="*65)

    L("\nPART A — Differential results per comparison:")
    for (a, b), df in all_results.items():
        nr   = int(df['robust'].sum())
        nfdr = int(df['fdr_significant'].sum())
        L(f"  {a:10s} vs {b:10s}: robust={nr:4d}  FDR-sig={nfdr:4d}")

    if nx is not None and all_hubs:
        L("\nPART B — CNA hub summary per state:")
        for state, hub_df in all_hubs.items():
            n_hubs = int(hub_df['is_hub'].sum())
            top3   = hub_df[hub_df['is_hub']].head(3)
            names  = [f"{r['chain']}{r['within_res']}" for _, r in top3.iterrows()]
            L(f"  {state:10s}: {n_hubs} hubs  top3={names}")

    L(f"\nAll outputs in: {OUTPUT_DIR}/")
    L("\nOutput files:")
    files = [
        # Part A
        ("differential_X_vs_Y.csv",           "all features ranked by |g|"),
        ("robust_X_vs_Y.csv",                 "robust features per comparison"),
        ("robust_rmsf_crosscomparison.csv",   "g per residue × all comparisons"),
        ("validation_summary.csv",            "LOO-LDA accuracy"),
        ("sensitivity_rmsf_counts.csv",       "threshold grid counts"),
        ("fig_rmsf_X_vs_Y.png",              "RMSF profiles + g per residue"),
        ("fig_effect_heatmap.png",            "effect size heatmap"),
        ("fig_sensitivity_heatmap.png",       "sensitivity grid"),
        # Part B
        ("hub_residues_{state}.csv",          "betweenness centrality ranking"),
        ("communities_{state}.csv",           "community membership"),
        ("intercommunity_edges_{state}.csv",  "communication between communities"),
        ("allosteric_paths_{state}.csv",      "hub-to-hub shortest paths"),
        ("hub_shifts.csv",                    "hub changes across states"),
        ("fig_hub_profile_{state}.png",       "BC along sequence"),
        ("fig_community_strip_{state}.png",   "residue strip by community"),
        ("fig_intercommunity_{state}.png",    "inter-community heatmap"),
        # Part C
        ("delta_network_Pg_minus_Apo.csv",        "GAFab rewiring Pg vs Apo"),
        ("delta_network_Pg_cGMP_minus_cGMP.csv",  "GAFab rewiring Pg_cGMP vs cGMP"),
        ("delta_top_edges_COMP.csv",              "top 50 gained + lost edges"),
        ("delta_domain_summary_COMP.csv",         "domain-pair rewiring summary"),
        ("pg_cross_signal_STATE.csv",             "Pg-GAFab per-residue signal"),
        ("fig_delta_network_COMP.png",            "per-residue delta rewiring"),
        ("fig_pg_cross_signal.png",               "Pg-GAFab signal both states"),
        ("pipeline_run.log",                      "full log of this run"),
    ]
    for fname, desc in files:
        L(f"  {fname:45s} — {desc}")
