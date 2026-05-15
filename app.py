import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from pathlib import Path

st.set_page_config(page_title='Turbofan Fleet Monitor', layout='wide')
st.title('Turbofan Fleet Health Monitor')
st.caption('NASA CMAPSS FD001 — sensor data → LightGBM quantile RUL predictions, computed on app startup.')

DATA_DIR       = Path(__file__).parent
USEFUL_SENSORS = ['s2', 's3', 's4', 's7', 's8', 's9', 's11',
                  's12', 's13', 's14', 's15', 's17', 's20', 's21']
RUL_CLIP       = 125
N_BASELINE     = 30
ROLL_WINDOW    = 10

col_names = (['unit_id', 'cycle']
             + [f'op{i}' for i in range(1, 4)]
             + [f's{i}'  for i in range(1, 22)])


@st.cache_data
def load_cmapss():
    train = pd.read_csv(DATA_DIR / 'train_FD001.txt', sep=r'\s+', header=None, names=col_names)
    test  = pd.read_csv(DATA_DIR / 'test_FD001.txt',  sep=r'\s+', header=None, names=col_names)
    rul   = pd.read_csv(DATA_DIR / 'RUL_FD001.txt',   sep=r'\s+', header=None, names=['RUL'])
    rul.index = np.arange(1, len(rul) + 1)
    rul.index.name = 'unit_id'
    return train, test, rul


def add_zscores(df):
    early = df[df['cycle'] <= N_BASELINE].groupby('unit_id')[USEFUL_SENSORS]
    means = early.mean()
    stds  = early.std().replace(0, 1)
    for s in USEFUL_SENSORS:
        m  = df['unit_id'].map(means[s])
        sd = df['unit_id'].map(stds[s])
        df[f'{s}_z'] = (df[s] - m) / sd
    return df


def build_features(df):
    df = df.sort_values(['unit_id', 'cycle'])
    z_cols = [f'{s}_z' for s in USEFUL_SENSORS]
    g = df.groupby('unit_id')[z_cols].rolling(window=ROLL_WINDOW, min_periods=1)
    cur   = df[z_cols]
    means = g.mean().reset_index(level=0, drop=True).add_suffix('_mean')
    stds  = g.std().reset_index(level=0, drop=True).add_suffix('_std').fillna(0)
    mins  = g.min().reset_index(level=0, drop=True).add_suffix('_min')
    maxs  = g.max().reset_index(level=0, drop=True).add_suffix('_max')
    return pd.concat([cur, means, stds, mins, maxs], axis=1).loc[df.index]


@st.cache_resource
def train_quantile_models(_train, _test):
    train_z = add_zscores(_train.copy())
    test_z  = add_zscores(_test.copy())
    train_z['RUL'] = (train_z.groupby('unit_id')['cycle']
                            .transform('max') - train_z['cycle']).clip(upper=RUL_CLIP)

    train_sorted = train_z.sort_values(['unit_id', 'cycle'])
    test_sorted  = test_z.sort_values(['unit_id', 'cycle'])
    X_train_df = build_features(train_sorted)
    feature_cols = X_train_df.columns
    X_train = X_train_df.values
    y_train = train_sorted['RUL'].values

    last_idx = test_sorted.groupby('unit_id')['cycle'].idxmax()
    X_test_df = build_features(test_sorted).reindex(columns=feature_cols, fill_value=0)
    X_test = X_test_df.loc[last_idx].values

    common = dict(objective='quantile', n_estimators=300, learning_rate=0.05,
                  num_leaves=31, min_child_samples=20, random_state=42, verbose=-1)
    preds = {}
    for q in [0.1, 0.5, 0.9]:
        m = lgb.LGBMRegressor(alpha=q, **common)
        m.fit(X_train, y_train)
        preds[q] = m.predict(X_test)

    return test_z, preds


with st.spinner('Loading CMAPSS data and training models (~30s on first launch)...'):
    train, test, rul = load_cmapss()
    test_z, preds = train_quantile_models(train, test)

uids     = sorted(test['unit_id'].unique())
p10      = preds[0.1]
p50      = preds[0.5]
p90      = preds[0.9]
true_rul = rul['RUL'].values

st.sidebar.header('Subset: FD001')
st.sidebar.write(f'Test engines: {len(uids)}')
st.sidebar.write(f'Useful sensors: {len(USEFUL_SENSORS)}')

tab1, tab2, tab3 = st.tabs(['Fleet Overview', 'Engine Drilldown', 'Maintenance Plan'])

with tab1:
    st.header('Fleet-wide RUL distribution')

    c1, c2, c3 = st.columns(3)
    c1.metric('Median predicted RUL', f'{np.median(p50):.0f} cycles')
    c2.metric('Engines with median RUL < 50', int((p50 < 50).sum()))
    c3.metric('Mean 80% interval width', f'{(p90 - p10).mean():.1f}')

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(p50, bins=20, edgecolor='black')
    ax.set_xlabel('Predicted RUL (median, cycles)')
    ax.set_ylabel('Number of engines')
    st.pyplot(fig)

    order = np.argsort(p50)
    xs = np.arange(len(p50))
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2.fill_between(xs, p10[order], p90[order], alpha=0.3, label='80% interval')
    ax2.plot(xs, p50[order], lw=1, label='median')
    ax2.scatter(xs, true_rul[order], s=10, c='black', label='true RUL', zorder=3)
    ax2.set_xlabel('Test engine (sorted by predicted RUL)')
    ax2.set_ylabel('RUL (cycles)')
    ax2.legend()
    st.pyplot(fig2)

with tab2:
    unit = st.selectbox('Select engine', uids)
    eng  = test_z[test_z['unit_id'] == unit].sort_values('cycle')
    i = uids.index(unit)

    c1, c2, c3 = st.columns(3)
    c1.metric('Predicted RUL (median)', f'{p50[i]:.0f}')
    c2.metric('80% interval', f'[{p10[i]:.0f}, {p90[i]:.0f}]')
    c3.metric('True RUL (held out)', f'{int(true_rul[i])}')

    sensor_choice = st.multiselect('Sensors to display',
                                   USEFUL_SENSORS,
                                   default=USEFUL_SENSORS[:4])
    if sensor_choice:
        n = len(sensor_choice)
        fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, sensor in zip(axes, sensor_choice):
            ax.plot(eng['cycle'], eng[f'{sensor}_z'], lw=0.8)
            ax.axhline(0, color='gray', lw=0.5)
            ax.set_ylabel(sensor)
        axes[-1].set_xlabel('cycle')
        st.pyplot(fig)

with tab3:
    st.header('Engines sorted by urgency')
    st.caption('Action thresholds: GROUND if 10%-lower-bound RUL < 20, INSPECT if < 50.')

    rows = []
    for i, u in enumerate(uids):
        action = ('GROUND' if p10[i] < 20
                  else 'INSPECT' if p10[i] < 50
                  else 'continue')
        rows.append({
            'unit_id':    u,
            'RUL_median': round(float(p50[i]), 1),
            'RUL_p10':    round(float(p10[i]), 1),
            'RUL_p90':    round(float(p90[i]), 1),
            'true_RUL':   int(true_rul[i]),
            'action':     action,
        })

    priority = {'GROUND': 0, 'INSPECT': 1, 'continue': 2}
    df = (pd.DataFrame(rows)
          .assign(_pri=lambda d: d['action'].map(priority))
          .sort_values(['_pri', 'RUL_p10'])
          .drop(columns='_pri'))

    def color(v):
        if v == 'GROUND':  return 'background-color: #ff7777'
        if v == 'INSPECT': return 'background-color: #ffcc77'
        return ''

    st.dataframe(df.style.applymap(color, subset=['action']),
                 use_container_width=True)
