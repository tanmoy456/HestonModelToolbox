import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from heston.P1P2Heston import heston_call_price, heston_put_price
from heston.QuantlibHeston import heston_option_price
from heston.CosMethodHeston import ChFHestonModel, COS_Heston_Price
from heston.MonteCarloHeston import GeneratePathsHestonAES, GeneratePathsHestonEuler, price_option_mc

from BlackScholes.BSprice import implied_volatility


def get_common_heston_params():
    st.sidebar.header("Heston Parameters")
    kappa = st.sidebar.slider(r"$\kappa$", 0.01, 5.0, 0.1)
    theta = st.sidebar.slider(r"$\theta$", 0.01, 1.0, 0.1)
    sigma = st.sidebar.slider(r"$\sigma$", 0.01, 1.0, 0.1)
    rho = st.sidebar.slider(r"$\rho$", -1.0, 1.0, -0.75)
    v0 = st.sidebar.slider(r"$v_0$", 0.001, 1.0, 0.05)
    st.sidebar.header("Market Parameters")
    r = st.sidebar.slider("Risk-free rate $r$", 0.0, 0.2, 0.05)
    T = st.sidebar.slider("Maturity $T$", 0.01, 5.0, 1.0)
    return kappa, theta, sigma, rho, v0, r, T


def get_common_inputs():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        S0 = st.number_input("Spot price ($S_0$)", value=100.0, format="%.2f")
    with col2:
        K_min = st.number_input("Min strike", value=50)
    with col3:
        K_max = st.number_input("Max strike", value=150)
    with col4:
        K_num = st.number_input("Number of strikes", value=10, step=1, min_value=3)
    strike_array = np.linspace(K_min, K_max, int(K_num))
    return S0, K_min, K_max, K_num, strike_array


def plot_heston_param_variation_plotly(param_name, param_list,
                                      S0, T, r, q,
                                      kappa, theta, sigma, rho, v0, strike_array):
    param_unicode_map = {
        'T': 'T',
        'r': 'r',
        'kappa': 'κ',
        'theta': 'θ',
        'sigma': 'σ',
        'rho': 'ρ',
        'v0': 'v₀'
    }

    
    fig = make_subplots(rows=1, cols=2,) # subplot_titles=("Call Prices", "Implied Volatility"))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'cyan', 'magenta']

    for idx, val in enumerate(param_list):
        p = dict(T=T, r=r, kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0)
        p[param_name] = val
        call_p = []
        iv_c = []

        for K in strike_array:
            call_price = heston_option_price(S0, K, p['r'], q, p['T'], p['v0'],
                                            p['kappa'], p['theta'], p['sigma'], p['rho'], option_type="call")
            call_p.append(call_price)
            iv = implied_volatility(call_price, S0, K, p['T'], p['r'], q, option_type="call")
            iv_c.append(iv)

        label = f"{param_unicode_map.get(param_name, param_name)}={val}"

        fig.add_trace(go.Scatter(
            x=strike_array, y=call_p, mode='lines+markers', name=label,
            line=dict(color=colors[idx % len(colors)]),
            showlegend=True
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=strike_array, y=iv_c, mode='lines+markers', name=label,
            line=dict(color=colors[idx % len(colors)]),
            showlegend=False
        ), row=1, col=2)

    fig.update_xaxes(title_text="Strike", row=1, col=1)
    fig.update_yaxes(title_text="Call Price", row=1, col=1)
    fig.update_xaxes(title_text="Strike", row=1, col=2)
    fig.update_yaxes(title_text="Implied Volatility", row=1, col=2)
    fig.update_layout(height=500, width=950, legend_title_text="Parameters")

    st.plotly_chart(fig, use_container_width=True)


def calculate_heston_prices(S0, strike_array, T, r, NoOfPaths, NoOfSteps,
                            kappa, theta, sigma, rho, v0):
    results_call, results_put = [], []
    q = 0.0  # dividend yield

    for K in strike_array:
        try:
            call_p1p2 = heston_call_price(S0, K, r, T, kappa, theta, sigma, rho, v0)
            put_p1p2 = heston_put_price(S0, K, r, T, kappa, theta, sigma, rho, v0)
        except Exception:
            call_p1p2, put_p1p2 = np.nan, np.nan

        try:
            call_ql = heston_option_price(S0, K, r, q, T, v0, kappa, theta, sigma, rho, option_type="call")
            put_ql  = heston_option_price(S0, K, r, q, T, v0, kappa, theta, sigma, rho, option_type="put")
        except Exception:
            call_ql, put_ql = np.nan, np.nan

        try:
            cf = ChFHestonModel(r, T, kappa, sigma, theta, v0, rho)
            call_cos = COS_Heston_Price(cf, "call", S0, r, T, K)
            put_cos  = COS_Heston_Price(cf, "put", S0, r, T, K)
        except Exception:
            call_cos, put_cos = np.nan, np.nan

        try:
            aes_paths = GeneratePathsHestonAES(NoOfPaths, NoOfSteps, T, r, S0,
                                               kappa, sigma, rho, theta, v0, seed=123)
            call_aes_mc = price_option_mc("call", aes_paths["S"], K, r, T)
            put_aes_mc  = price_option_mc("put", aes_paths["S"], K, r, T)
        except Exception:
            call_aes_mc, put_aes_mc = np.nan, np.nan

        try:
            euler_paths = GeneratePathsHestonEuler(NoOfPaths, NoOfSteps, T, r, S0,
                                                   kappa, sigma, rho, theta, v0, seed=123)
            call_euler_mc = price_option_mc("call", euler_paths["S"], K, r, T)
            put_euler_mc  = price_option_mc("put", euler_paths["S"], K, r, T)
        except Exception:
            call_euler_mc, put_euler_mc = np.nan, np.nan

        results_call.append([K, call_p1p2, call_ql, call_cos, call_aes_mc, call_euler_mc])
        results_put.append([K, put_p1p2, put_ql, put_cos, put_aes_mc, put_euler_mc])

    columns = ["Strike", "P1P2", "QuantLib", "COS", "AES-MC", "EULER-MC"]
    df_call = pd.DataFrame(results_call, columns=columns)
    df_put  = pd.DataFrame(results_put, columns=columns)
    return df_call, df_put


def pricing_methods_page(strike_array, S0, kappa, theta, sigma, rho, v0, r, T):
    st.subheader("Pricing Methods")
    methods = st.multiselect(
        "Select pricing methods to compare",
        ["P1P2", "QuantLib", "COS", "AES-MC", "EULER-MC"],
        default=["P1P2", "QuantLib", "COS"]
    )
    NoOfPaths, NoOfSteps = 10000, 200
    if "AES-MC" in methods or "EULER-MC" in methods:
        with st.expander("Monte Carlo Settings"):
            NoOfPaths = st.number_input("Paths", value=10000)
            NoOfSteps = st.number_input("Steps", value=200)

    df_call, df_put = calculate_heston_prices(S0, strike_array, T, r,
                                              NoOfPaths, NoOfSteps,
                                              kappa, theta, sigma, rho, v0)

    fig_plotly = make_subplots(rows=1, cols=2) #, subplot_titles=("Call Prices", "Put Prices"))
    colors = ["blue", "green", "red", "purple", "orange"]
    markers = ["circle", "diamond", "square", "triangle-up", "cross"]
    for i, m in enumerate(methods):
        fig_plotly.add_trace(go.Scatter(
            x=strike_array, y=df_call[m],
            mode='lines+markers', marker_symbol=markers[i % len(markers)],
            marker=dict(color=colors[i % len(colors)]),
            line=dict(color=colors[i % len(colors)]), name=m
        ), row=1, col=1)
        fig_plotly.add_trace(go.Scatter(
            x=strike_array, y=df_put[m],
            mode='lines+markers', marker_symbol=markers[i % len(markers)],
            marker=dict(color=colors[i % len(colors)]),
            line=dict(color=colors[i % len(colors)]), name=m,
            showlegend=False
        ), row=1, col=2)
    fig_plotly.update_xaxes(title_text="Strike", row=1, col=1)
    fig_plotly.update_yaxes(title_text="Call Price", row=1, col=1)
    fig_plotly.update_xaxes(title_text="Strike", row=1, col=2)
    fig_plotly.update_yaxes(title_text="Put Price", row=1, col=2)
    fig_plotly.update_layout(width=950, height=500)
    st.plotly_chart(fig_plotly)

    if st.checkbox("Show DataFrames"):
        st.subheader("Call Prices DataFrame")
        st.dataframe(df_call[["Strike"] + methods])
        st.subheader("Put Prices DataFrame")
        st.dataframe(df_put[["Strike"] + methods])


def parameter_variation_page(strike_array, S0, kappa, theta, sigma, rho, v0, r, T):
    st.subheader("Parameter Variation")

    c1, c2 = st.columns([2, 3])
    with c1:
        param_name = st.selectbox("Parameter", ['kappa', 'theta', 'sigma', 'rho', 'v0', 'T', 'r'])
    with c2:
        default_vals = {
            'kappa': "0.1,0.5,1.0",
            'theta': "0.05,0.1,0.2",
            'sigma': "0.1,0.3,0.5",
            'rho': "-0.75,-0.5,0.0",
            'v0': "0.01,0.05,0.1",
            'T': "0.5,1.0,2.0",
            'r': "0.01,0.05,0.1"
        }
        param_vals_input = st.text_input("Comma-separated values", default_vals.get(param_name, "0.1,0.5,1.0"))
    try:
        param_list = [float(x.strip()) for x in param_vals_input.split(',')]
    except Exception:
        st.error("Invalid parameter values.")
        return

    plot_heston_param_variation_plotly(param_name, param_list,
                                      S0=S0, T=T, r=r, q=0,
                                      kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0,
                                      strike_array=strike_array)


def main():
    st.title("Heston Model Interactive Explorer")

    # Sidebar Heston and Market parameters that are common
    kappa, theta, sigma, rho, v0, r, T = get_common_heston_params()

    # Common inputs across tabs
    S0, K_min, K_max, K_num, strike_array = get_common_inputs()

    tab1, tab2, tab3 = st.tabs(["Method Comparison", "Parameter Variation", "Heston Model"])

    with tab1:
        pricing_methods_page(strike_array, S0, kappa, theta, sigma, rho, v0, r, T)

    with tab2:
        parameter_variation_page(strike_array, S0, kappa, theta, sigma, rho, v0, r, T)

    with tab3:
        st.subheader("Heston Model: Theory and References")
        st.markdown(r"""
        **Heston Stochastic Volatility Model**  

        The Heston model describes the evolution of the asset price $S_t$ and the stochastic variance $v_t$ as follows:
        $$
        \begin{aligned}
        dS_t &= \mu S_t \, dt + \sqrt{v_t}\, S_t\, dW_t^{(1)} \\
        dv_t &= \kappa (\theta - v_t) \, dt + \sigma \sqrt{v_t} \, dW_t^{(2)}
        \end{aligned}
        $$
        where the two Brownian motions are correlated:
        $$
        dW_t^{(1)} \, dW_t^{(2)} = \rho\, dt
        $$

        $$
        \begin{aligned}
        S_t &: \quad \text{Asset price at time } t \\
        v_t &: \quad \text{Instantaneous variance at time } t \\
        \mu &: \quad \text{Drift (expected growth rate) of the asset price}\\
        \kappa &: \quad \text{Rate at which } v_t \text{ reverts to its long-term mean (mean reversion speed)} \\
        \theta &: \quad \text{Long-term mean (level) of the variance process} \\
        \sigma &: \quad \text{Volatility of volatility (vol-of-vol)} \\
        \rho &: \quad \text{Correlation between the two Brownian motions} \\
        dW_t^{(1)},\ dW_t^{(2)} &: \quad \text{Standard Brownian motions with correlation } \rho
        \end{aligned}
        $$


        **Notes:**

        - The variance process $v_t$ is a Cox-Ingersoll-Ross (CIR) process, which ensures $v_t \geq 0$ if the Feller condition holds: $2\kappa\theta > \sigma^2$.

        -----
                    
        **References:** 
        - [Heston 1993 paper](https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf) 
        - [Heston Model Wiki](https://en.wikipedia.org/wiki/Heston_model) 

        """)


if __name__ == "__main__":
    main()
