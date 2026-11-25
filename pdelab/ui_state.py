"""Explicit navigation for actions that produce a result in another section."""

import streamlit as st


def select_section(key, label):
    st.session_state[key] = label
