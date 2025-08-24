# Advanced Web Calculator (Streamlit)
# Run: streamlit run app.py
import math
import json
import io
import base64
from typing import List, Tuple

import numpy as np
import pandas as pd
import sympy as sp
import streamlit as st
from sympy.parsing.sympy_parser import parse_expr

# ------------------------
# Page config & styling
# ------------------------
st.set_page_config(page_title="Advanced Web Calculator", page_icon="🧮", layout="wide")
st.title("🧮 Advanced Web Calculator")
st.caption("A multi-tool calculator: basic, scientific, graphing, matrices, equations, stats, unit conversions, and programmer mode.")

# Init session state for history
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"tool": str, "expr": str, "result": any}

def add_history(tool: str, expr: str, result):
    try:
        st.session_state.history.append({"tool": tool, "expr": expr, "result": str(result)})
    except Exception:
        pass

# Utility: safe eval using sympy
def eval_expr(expr: str, variables=None):
    variables = variables or {}
    # Provide common names to sympy
    local_dict = {name: getattr(sp, name) for name in dir(sp) if not name.startswith("_")}
    # Add math functions
    local_dict.update({name: getattr(sp, name) for name in ["sin","cos","tan","asin","acos","atan","exp","log","sqrt","pi","E"] if hasattr(sp, name)})
    local_dict.update(variables)
    return parse_expr(expr, local_dict=local_dict, evaluate=True)

# Download helper
def make_download_button(obj, filename: str, label: str):
    if isinstance(obj, (dict, list)):
        data = json.dumps(obj, indent=2).encode("utf-8")
        mime = "application/json"
    elif isinstance(obj, pd.DataFrame):
        data = obj.to_csv(index=False).encode("utf-8")
        mime = "text/csv"
    else:
        data = str(obj).encode("utf-8")
        mime = "text/plain"
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime};base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

# ------------------------
# Sidebar: About & Settings
# ------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    theme = st.selectbox("Theme", ["Auto (Streamlit)", "Light", "Dark"], index=0)
    st.write("Tips:")
    st.markdown("- Use **Scientific** tab for constants and trig.\n- Use **Graphing** to plot f(x).\n- Use **Matrices** for linear algebra.\n- Use **Equations** to solve symbolically.\n- Use **Programmer** for base/bitwise ops.")
    st.divider()
    if st.button("🧹 Clear History"):
        st.session_state.history = []

if theme == "Light":
    st.markdown("<style>body{color:#111;background:#fff}</style>", unsafe_allow_html=True)
elif theme == "Dark":
    st.markdown("<style>body{color:#eee;background:#111}</style>", unsafe_allow_html=True)

# ------------------------
# Tabs
# ------------------------
tabs = st.tabs([
    "Standard", "Scientific", "Graphing", "Matrices", "Equations", "Statistics", "Units", "Programmer", "History"
])

# ---------- Standard ----------
with tabs[0]:
    st.subheader("Standard Calculator")
    col1, col2 = st.columns([2,1])
    with col1:
        expr = st.text_input("Expression", value="(25 + 15)/5 + 2*3", help="Use + - * / // % ** ( )")
        if st.button("Calculate", key="std_calc"):
            try:
                # Use Python eval in a limited scope for standard arithmetic
                # but convert to float/int; avoid names
                allowed = {"__builtins__": None}
                result = eval(expr, allowed, {})
                st.success(f"Result: {result}")
                add_history("Standard", expr, result)
            except Exception as e:
                st.error(f"Error: {e}")
    with col2:
        st.write("Quick Ops")
        a = st.number_input("A", value=10.0)
        b = st.number_input("B", value=3.0)
        st.write(f"A + B = {a + b}")
        st.write(f"A - B = {a - b}")
        st.write(f"A × B = {a * b}")
        st.write(f"A ÷ B = {a / b if b != 0 else '∞'}")
        st.write(f"A ^ B = {a ** b}")

# ---------- Scientific ----------
with tabs[1]:
    st.subheader("Scientific Calculator (SymPy)")
    c1, c2 = st.columns([2,1])
    with c1:
        sexpr = st.text_input("Enter expression",
                              value="sin(pi/3) + log(E) + sqrt(2)**2")
        if st.button("Evaluate", key="sci_eval"):
            try:
                res = eval_expr(sexpr)
                st.success(f"Exact: {sp.simplify(res)}")
                st.info(f"Numeric: {sp.N(res, 12)}")
                add_history("Scientific", sexpr, res)
            except Exception as e:
                st.error(f"Error: {e}")
    with c2:
        st.write("Constants")
        cols = st.columns(3)
        consts = {"π": sp.pi, "e": sp.E, "φ": (1+sp.sqrt(5))/2}
        for i,(k,v) in enumerate(consts.items()):
            with cols[i%3]:
                st.code(f"{k} = {sp.N(v, 12)}")

        st.write("Functions")
        st.markdown("`sin, cos, tan, asin, acos, atan, exp, log, sqrt, abs, floor, ceiling`")

# ---------- Graphing ----------
with tabs[2]:
    st.subheader("Graph f(x)")
    gcol1, gcol2 = st.columns([2,1])
    with gcol1:
        f_expr = st.text_input("f(x) =", value="sin(x) * exp(-x/5)")
        x_min, x_max = st.slider("x-range", -20.0, 20.0, (-10.0, 10.0), step=0.5)
        samples = st.slider("Samples", 100, 5000, 500, step=100)
        if st.button("Plot f(x)"):
            try:
                x = sp.symbols('x')
                f = sp.lambdify(x, eval_expr(f_expr, {"x": x}), "numpy")
                xs = np.linspace(x_min, x_max, samples)
                ys = f(xs)
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot(xs, ys)  # (No specific colors/styles as requested)
                ax.set_xlabel("x")
                ax.set_ylabel("f(x)")
                ax.grid(True, which="both", linestyle="--", alpha=0.3)
                st.pyplot(fig, clear_figure=True)
                add_history("Graphing", f"f(x)={f_expr}", "plotted")
            except Exception as e:
                st.error(f"Plot error: {e}")
    with gcol2:
        st.info("Supports any SymPy-friendly function of x. Examples:")
        st.code("sin(x), cos(2*x), exp(-x/3)*sin(5*x), (x**2-1)/(x-1)")

# ---------- Matrices ----------
with tabs[3]:
    st.subheader("Matrix Calculator")
    st.caption("Enter matrices as rows separated by semicolons. Example: 1 2; 3 4")
    A_str = st.text_input("Matrix A", value="1 2; 3 4")
    B_str = st.text_input("Matrix B", value="5 6; 7 8")
    op = st.selectbox("Operation", ["A + B", "A - B", "A × B", "A · Aᵀ", "det(A)", "inv(A)", "rank(A)", "eig(A)"])
    def parse_matrix(s: str) -> np.ndarray:
        rows = [r.strip() for r in s.split(";") if r.strip()]
        data = [list(map(float, r.replace(",", " ").split())) for r in rows]
        width = max(len(r) for r in data)
        # normalize rows
        data = [r + [0.0]*(width-len(r)) for r in data]
        return np.array(data, dtype=float)

    if st.button("Compute Matrix"):
        try:
            A = parse_matrix(A_str)
            B = parse_matrix(B_str) if "B" in op else None
            if op == "A + B": res = A + B
            elif op == "A - B": res = A - B
            elif op == "A × B": res = A @ B
            elif op == "A · Aᵀ": res = A @ A.T
            elif op == "det(A)": res = np.linalg.det(A)
            elif op == "inv(A)": res = np.linalg.inv(A)
            elif op == "rank(A)": res = np.linalg.matrix_rank(A)
            elif op == "eig(A)":
                vals, vecs = np.linalg.eig(A)
                st.write("Eigenvalues:", vals)
                st.write("Eigenvectors:")
                st.write(pd.DataFrame(vecs))
                add_history("Matrices", op, {"eigvals": vals.tolist()})
                res = None
            if res is not None:
                st.write(pd.DataFrame(res))
                add_history("Matrices", op, np.array(res).tolist())
        except Exception as e:
            st.error(f"Matrix error: {e}")

# ---------- Equations ----------
with tabs[4]:
    st.subheader("Solve Equations")
    st.caption("Enter equations with '=' (e.g., x^2 - 5*x + 6 = 0). Separate multiple equations by newline.")
    text = st.text_area("Equations", value="x^2 - 5*x + 6 = 0")
    vars_text = st.text_input("Variables (comma-separated)", value="x")
    if st.button("Solve"):
        try:
            var_names = [v.strip() for v in vars_text.split(",") if v.strip()]
            symbols = sp.symbols(var_names)
            equations = []
            for line in text.splitlines():
                if not line.strip():
                    continue
                if "=" in line:
                    left, right = line.split("=", 1)
                    eq = sp.Eq(eval_expr(left, dict(zip(var_names, symbols))),
                               eval_expr(right, dict(zip(var_names, symbols))))
                else:
                    eq = sp.Eq(eval_expr(line, dict(zip(var_names, symbols))), 0)
                equations.append(eq)
            sol = sp.solve(equations, symbols, dict=True)
            st.success(sol)
            add_history("Equations", text, sol)
        except Exception as e:
            st.error(f"Solve error: {e}")

# ---------- Statistics ----------
with tabs[5]:
    st.subheader("Basic Statistics")
    st.caption("Paste numbers separated by commas or spaces.")
    data_str = st.text_area("Data", value="12 15 14 10 12 18 20 14 16")
    if st.button("Analyze"):
        try:
            data = np.array([float(x) for x in data_str.replace(",", " ").split() if x.strip()], dtype=float)
            if data.size == 0:
                raise ValueError("No data provided.")
            df = pd.DataFrame({"x": data})
            st.write(df.describe())
            st.write("Median:", float(np.median(data)))
            st.write("Mode(s):", pd.Series(data).mode().tolist())
            # Histogram
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.hist(data, bins=min(20, max(3, int(np.sqrt(len(data))))))
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            st.pyplot(fig, clear_figure=True)
            add_history("Statistics", "describe/hist", df.describe().to_dict())
        except Exception as e:
            st.error(f"Stats error: {e}")

# ---------- Units ----------
with tabs[6]:
    st.subheader("Unit Converter")
    # Simple curated unit sets
    unit_sets = {
        "Length": {"m":1.0, "cm":0.01, "mm":0.001, "km":1000.0, "inch":0.0254, "ft":0.3048, "yd":0.9144, "mile":1609.344},
        "Mass": {"kg":1.0, "g":0.001, "mg":1e-6, "lb":0.45359237, "oz":0.028349523125},
        "Time": {"s":1.0, "ms":0.001, "min":60.0, "hr":3600.0, "day":86400.0},
        "Temperature": {"C":("C",), "F":("F",), "K":("K",)},
        "Data": {"bit":1.0, "byte":8.0, "KB":8.0*1024, "MB":8.0*1024**2, "GB":8.0*1024**3},
    }
    cat = st.selectbox("Category", list(unit_sets.keys()))
    if cat == "Temperature":
        val = st.number_input("Value", value=25.0)
        from_u = st.selectbox("From", ["C","F","K"], index=0)
        to_u = st.selectbox("To", ["C","F","K"], index=2)
        def convert_temp(v, from_u, to_u):
            if from_u == to_u: return v
            # Convert to Celsius
            if from_u == "C": c = v
            elif from_u == "F": c = (v - 32)*5/9
            elif from_u == "K": c = v - 273.15
            # Celsius to target
            if to_u == "C": return c
            elif to_u == "F": return c*9/5 + 32
            elif to_u == "K": return c + 273.15
        if st.button("Convert", key="temp_conv"):
            try:
                res = convert_temp(val, from_u, to_u)
                st.success(f"{val} {from_u} = {res} {to_u}")
                add_history("Units", f"{val} {from_u} -> {to_u}", res)
            except Exception as e:
                st.error(f"Conversion error: {e}")
    else:
        val = st.number_input("Value", value=1.0)
        units = unit_sets[cat]
        from_u = st.selectbox("From", list(units.keys()))
        to_u = st.selectbox("To", list(units.keys()), index=1)
        if st.button("Convert", key="unit_conv"):
            try:
                base = val * units[from_u]
                res = base / units[to_u]
                st.success(f"{val} {from_u} = {res} {to_u}")
                add_history("Units", f"{val} {from_u} -> {to_u}", res)
            except Exception as e:
                st.error(f"Conversion error: {e}")

# ---------- Programmer ----------
with tabs[7]:
    st.subheader("Programmer Calculator")
    n = st.text_input("Number (dec/hex/bin)", value="255", help="Examples: 255, 0xff, 0b1010")
    try:
        val = int(n, 0)  # auto base
        bits = st.slider("Bits", 8, 64, 16, step=8)
        st.write(f"Dec: {val}")
        st.write(f"Hex: {hex(val)}")
        st.write(f"Bin: {bin(val)}")
        st.write(f"Oct: {oct(val)}")
        # Bitwise with a second number
        m = st.text_input("Other (for bitwise ops)", value="15")
        mval = int(m, 0)
        st.write(f"A & B = {val & mval}")
        st.write(f"A | B = {val | mval}")
        st.write(f"A ^ B = {val ^ mval}")
        st.write(f"~A (within {bits} bits) = {val ^ ((1<<bits)-1)}")
        add_history("Programmer", f"{n}", f"dec={val}")
    except Exception as e:
        st.info("Enter a valid integer like 255, 0xff, or 0b1010.")

# ---------- History ----------
with tabs[8]:
    st.subheader("History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True, hide_index=True)
        make_download_button(df, "calc_history.csv", "⬇️ Download History (CSV)")
        make_download_button(st.session_state.history, "calc_history.json", "⬇️ Download History (JSON)")
    else:
        st.info("No history yet. Calculate something!")
