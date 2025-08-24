# Advanced Web Calculator (Streamlit)
# Run with: streamlit run app.py
import math
import json
import base64
from typing import List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
import streamlit as st
from sympy.parsing.sympy_parser import parse_expr
from PIL import Image

# ------------------------
# Page config
# ------------------------
st.set_page_config(
    page_title="Advanced Web Calculator created by Vishwajit Sen",
    page_icon="🧮",
    layout="wide"
)

# ------------------------
# Load Logo
# ------------------------
LOGO_FILENAME = "vishwajit.jpg"
_logo_img = None
if Path(LOGO_FILENAME).exists():
    try:
        _logo_img = Image.open(LOGO_FILENAME)
    except Exception:
        _logo_img = None

# ------------------------
# Custom CSS (Gradient + Bold Text + Contrast)
# ------------------------
st.markdown(
    """
    <style>
        /* Background */
        .stApp {
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            min-height: 100vh;
        }
        /* Main container card */
        .block-container {
            background: rgba(255,255,255,0.96);
            border-radius: 16px;
            padding: 20px 25px;
            box-shadow: 0 12px 28px rgba(0,0,0,0.22);
        }
        /* Titles */
        h1, h2, h3, h4 {
            font-weight: bold;
            color: #0a66c2;
        }
        label, .stMarkdown, .stTextInput, .stSelectbox, .stNumberInput, .stTextArea {
            font-weight: bold !important;
            color: #111 !important;
        }
        /* Buttons */
        .stButton>button {
            background: #0a66c2;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 8px 14px;
        }
        .stButton>button:hover {
            background: #084a91;
            color: #fff;
        }
        /* Results */
        .result-box {
            font-size: 20px;
            font-weight: bold;
            color: #facc15;
            background-color: #1e293b;
            padding: 12px;
            border-radius: 10px;
            text-align: center;
            margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------
# Header
# ------------------------
header_cols = st.columns([1, 9, 1])
with header_cols[0]:
    if _logo_img:
        st.image(_logo_img, width=70)
with header_cols[1]:
    st.markdown(
        "<h1>🧮 Advanced Web Calculator created by Vishwajit Sen</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='font-weight:bold;color:#334155;'>"
        "A multi-tool calculator: Standard, Scientific, Graphing, Matrices, Equations, Statistics, Units & Programmer tools."
        "</div>",
        unsafe_allow_html=True,
    )
with header_cols[2]:
    st.write("")

st.markdown("---")

# ------------------------
# History state
# ------------------------
if "history" not in st.session_state:
    st.session_state.history = []

def add_history(tool: str, expr: str, result):
    try:
        st.session_state.history.append(
            {"tool": tool, "expr": expr, "result": str(result)}
        )
    except Exception:
        pass

# ------------------------
# Expression evaluator
# ------------------------
def eval_expr(expr: str, variables=None):
    variables = variables or {}
    local_dict = {name: getattr(sp, name) for name in dir(sp) if not name.startswith("_")}
    local_dict.update(
        {name: getattr(sp, name) for name in
         ["sin","cos","tan","asin","acos","atan","exp","log","sqrt","pi","E"]
         if hasattr(sp, name)}
    )
    local_dict.update(variables)
    return parse_expr(expr, local_dict=local_dict, evaluate=True)

# ------------------------
# Download helper
# ------------------------
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
    href = f'<a href="data:{mime};base64,{b64}" download="{filename}"><b>{label}</b></a>'
    st.markdown(href, unsafe_allow_html=True)

# ------------------------
# Sidebar
# ------------------------
with st.sidebar:
    if _logo_img:
        st.image(_logo_img, caption="Vishwajit Sen", width=200)
    else:
        st.write("**Vishwajit Sen**")
    st.header("⚙️ Settings")
    theme = st.selectbox("Theme", ["Auto (Streamlit)", "Light", "Dark"], index=0)
    st.markdown(
        "**Tips:**\n- Use Scientific tab for symbolic math\n"
        "- Graphing to plot f(x)\n"
        "- Matrices for linear algebra\n"
        "- Equations for solving\n"
        "- Programmer for base & bitwise ops"
    )
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
    "Standard", "Scientific", "Graphing", "Matrices",
    "Equations", "Statistics", "Units", "Programmer", "History"
])

# ---------- Standard ----------
with tabs[0]:
    st.subheader("Standard Calculator")
    col1, col2 = st.columns([2,1])
    with col1:
        expr = st.text_input("Expression", value="(25 + 15)/5 + 2*3")
        if st.button("Calculate", key="std_calc"):
            try:
                result = eval(expr, {"__builtins__": None}, {})
                st.markdown(f'<div class="result-box">Result: {result}</div>', unsafe_allow_html=True)
                add_history("Standard", expr, result)
            except Exception as e:
                st.error(f"Error: {e}")
    with col2:
        a = st.number_input("A", value=10.0)
        b = st.number_input("B", value=3.0)
        st.write(f"**A + B = {a + b}**")
        st.write(f"**A - B = {a - b}**")
        st.write(f"**A × B = {a * b}**")
        st.write(f"**A ÷ B = {a / b if b != 0 else '∞'}**")
        st.write(f"**A ^ B = {a ** b}**")

# ---------- Scientific ----------
with tabs[1]:
    st.subheader("Scientific Calculator (SymPy)")
    sexpr = st.text_input("Enter expression", value="sin(pi/3) + log(E) + sqrt(2)**2")
    if st.button("Evaluate", key="sci_eval"):
        try:
            res = eval_expr(sexpr)
            st.markdown(f'<div class="result-box">Exact: {sp.simplify(res)}</div>', unsafe_allow_html=True)
            st.info(f"Numeric: {sp.N(res, 12)}")
            add_history("Scientific", sexpr, res)
        except Exception as e:
            st.error(f"Error: {e}")

# ---------- Graphing ----------
with tabs[2]:
    st.subheader("Graph f(x)")
    f_expr = st.text_input("f(x) =", value="sin(x) * exp(-x/5)")
    x_min, x_max = st.slider("x-range", -20.0, 20.0, (-10.0, 10.0), step=0.5)
    if st.button("Plot f(x)"):
        try:
            x = sp.symbols('x')
            f = sp.lambdify(x, eval_expr(f_expr, {"x": x}), "numpy")
            xs = np.linspace(x_min, x_max, 400)
            ys = f(xs)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(xs, ys)
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)
            add_history("Graphing", f"f(x)={f_expr}", "plotted")
        except Exception as e:
            st.error(f"Plot error: {e}")

# ---------- Matrices ----------
with tabs[3]:
    st.subheader("Matrix Calculator")
    A_str = st.text_input("Matrix A", value="1 2; 3 4")
    B_str = st.text_input("Matrix B", value="5 6; 7 8")
    op = st.selectbox("Operation", ["A + B","A - B","A × B","det(A)","inv(A)","rank(A)"])
    def parse_matrix(s): return np.array([list(map(float,r.split())) for r in s.split(";")])
    if st.button("Compute Matrix"):
        try:
            A = parse_matrix(A_str)
            B = parse_matrix(B_str)
            if op=="A + B": res=A+B
            elif op=="A - B": res=A-B
            elif op=="A × B": res=A@B
            elif op=="det(A)": res=np.linalg.det(A)
            elif op=="inv(A)": res=np.linalg.inv(A)
            elif op=="rank(A)": res=np.linalg.matrix_rank(A)
            st.write(pd.DataFrame(res if isinstance(res,np.ndarray) else [[res]]))
            add_history("Matrices", op, str(res))
        except Exception as e:
            st.error(f"Matrix error: {e}")

# ---------- Equations ----------
with tabs[4]:
    st.subheader("Solve Equations")
    text = st.text_area("Equations", value="x^2 - 5*x + 6 = 0")
    if st.button("Solve"):
        try:
            x = sp.symbols('x')
            if "=" in text:
                left,right=text.split("=")
                eq=sp.Eq(eval_expr(left,{"x":x}),eval_expr(right,{"x":x}))
            else:
                eq=sp.Eq(eval_expr(text,{"x":x}),0)
            sol=sp.solve(eq,x)
            st.markdown(f'<div class="result-box">{sol}</div>', unsafe_allow_html=True)
            add_history("Equations", text, sol)
        except Exception as e:
            st.error(f"Solve error: {e}")

# ---------- Statistics ----------
with tabs[5]:
    st.subheader("Statistics")
    data_str = st.text_area("Data", value="12 15 14 10 12 18 20 14 16")
    if st.button("Analyze"):
        try:
            data=np.array([float(x) for x in data_str.replace(","," ").split()])
            df=pd.DataFrame(data,columns=["x"])
            st.write(df.describe())
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots()
            ax.hist(data,bins=10)
            st.pyplot(fig)
            add_history("Statistics","analysis",df.describe().to_dict())
        except Exception as e:
            st.error(f"Stats error: {e}")

# ---------- Units ----------
with tabs[6]:
    st.subheader("Unit Converter")
    cat="Length"
    units={"m":1,"cm":0.01,"km":1000}
    val=st.number_input("Value",1.0)
    from_u=st.selectbox("From",list(units.keys()))
    to_u=st.selectbox("To",list(units.keys()))
    if st.button("Convert"):
        res=val*units[from_u]/units[to_u]
        st.markdown(f'<div class="result-box">{val} {from_u} = {res} {to_u}</div>', unsafe_allow_html=True)
        add_history("Units",f"{val}{from_u}->{to_u}",res)

# ---------- Programmer ----------
with tabs[7]:
    st.subheader("Programmer Calculator")
    n=st.text_input("Number (dec/hex/bin)",value="255")
    try:
        val=int(n,0)
        st.write(f"**Dec:** {val}")
        st.write(f"**Hex:** {hex(val)}")
        st.write(f"**Bin:** {bin(val)}")
    except:
        st.error("Invalid number")

# ---------- History ----------
with tabs[8]:
    st.subheader("History")
    if st.session_state.history:
        df=pd.DataFrame(st.session_state.history)
        st.dataframe(df)
        make_download_button(df,"history.csv","⬇️ Download CSV")
    else:
        st.info("No history yet.")
