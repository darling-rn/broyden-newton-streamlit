import streamlit as st
import numpy as np

st.title("Método de Newton y Broyden - Ejercicio 2.32 (Sauer)")

def F(x):
    u, v = x
    return np.array([
        v - u**3,
        u**2 + v**2 - 1
    ])

def J(x):
    u, v = x
    return np.array([
        [-3*u**2, 1],
        [2*u,     2*v]
    ])

def newton(x0, tol=1e-8, max_iter=50):
    x = x0.copy()
    for i in range(max_iter):
        fx = F(x)
        if np.linalg.norm(fx) < tol:
            return x, i
        dx = np.linalg.solve(J(x), -fx)
        x += dx
    return x, max_iter

def broyden(x0, tol=1e-8, max_iter=50):
    x = x0.copy()
    n = len(x)
    B = np.eye(n)
    fx = F(x)
    for i in range(max_iter):
        if np.linalg.norm(fx) < tol:
            return x, i
        dx = np.linalg.solve(B, -fx)
        x_new = x + dx
        fx_new = F(x_new)
        delta_x = x_new - x
        delta_f = fx_new - fx
        B += np.outer((delta_f - B @ delta_x), delta_x) / (delta_x @ delta_x)
        x, fx = x_new, fx_new
    return x, max_iter

# Inputs
u0 = st.number_input("Valor inicial u", value=1.0)
v0 = st.number_input("Valor inicial v", value=1.0)
x0 = np.array([u0, v0])

if st.button("Ejecutar Métodos"):
    xn, iter_n = newton(x0)
    xb, iter_b = broyden(x0)
    st.write("### Resultados:")
    st.write(f"**Newton:** u = {xn[0]:.6f}, v = {xn[1]:.6f} (iteraciones: {iter_n})")
    st.write(f"**Broyden:** u = {xb[0]:.6f}, v = {xb[1]:.6f} (iteraciones: {iter_b})")
