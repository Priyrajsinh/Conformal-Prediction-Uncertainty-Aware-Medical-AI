"""Glassmorphism indigo/purple theme for Gradio (rules C14, C44)."""

import gradio as gr


def get_theme() -> gr.themes.Base:
    """Return a custom indigo/purple Gradio theme."""
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
    )


def get_css() -> str:
    """Return glassmorphism CSS string for Gradio Blocks."""
    return """
    body, .gradio-container {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
        min-height: 100vh;
    }
    .hero {
        text-align: center;
        padding: 2rem 1rem 1rem;
        animation: slideUp 0.6s ease-out;
    }
    .hero h1 {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #818cf8, #67e8f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .hero p {
        color: #c4b5fd;
        font-size: 1rem;
        margin-bottom: 0.75rem;
    }
    .hero a {
        color: #818cf8;
        text-decoration: none;
        font-weight: 600;
    }
    .hero a:hover { text-decoration: underline; }
    .gr-box, .gr-form, .gr-panel, .block {
        background: rgba(255,255,255,0.07) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 16px !important;
    }
    button.primary {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important;
        font-weight: 700;
        letter-spacing: 0.05em;
        transition: box-shadow 0.2s;
    }
    button.primary:hover {
        box-shadow: 0 0 24px rgba(139, 92, 246, 0.7) !important;
    }
    .result-reveal {
        animation: slideUp 0.4s ease-out;
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    label, .label-wrap span { color: #e0e7ff !important; font-weight: 600; }
    """
