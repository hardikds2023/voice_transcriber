"""
app.py

Streamlit web application for voice memo transcription and analysis.

Run with:
    streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from transcriber import SUPPORTED_EXTENSIONS, format_timestamp, load_model, transcribe
from analyzer import analyze

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Voice Memo Transcriber",
    page_icon="🎙️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .key-point {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 0 4px 4px 0;
    }
    .category-header {
        background-color: #e8f4f8;
        padding: 6px 12px;
        border-radius: 4px;
        margin-top: 12px;
        font-weight: 600;
    }
    .category-item {
        padding: 4px 12px 4px 24px;
    }
    .transcript-segment {
        padding: 4px 0;
    }
    .timestamp {
        color: #6c757d;
        font-family: monospace;
        font-size: 0.85em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    model_size = st.selectbox(
        "Whisper Model",
        options=["tiny", "base", "small", "medium", "large"],
        index=1,
        help="Larger models are more accurate but slower.",
    )

    st.divider()

    api_key = st.text_input(
        "OpenAI API Key (optional)",
        type="password",
        help=(
            "Provide an API key for GPT-powered key point extraction and "
            "categorization. Without a key, a basic extractive approach is used."
        ),
    )

    use_openai = st.checkbox(
        "Use OpenAI for analysis",
        value=True,
        help="Uncheck to always use the basic (local) analysis.",
    )

    st.divider()
    st.caption(
        "**How it works**\n\n"
        "1. Upload an audio file (.m4a, .caf, .wav, .mp3)\n"
        "2. Whisper transcribes the audio locally\n"
        "3. The transcript is analyzed for key points\n"
        "4. Notes are organized by category"
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("Voice Memo Transcriber")
st.markdown("Upload an Apple Voice Memo or audio file to get a full transcript, highlighted key points, and categorized notes.")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
    help="Supported formats: M4A, CAF, WAV, MP3",
)

if uploaded_file is not None:
    # Show audio player
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

    if st.button("Transcribe & Analyze", type="primary", use_container_width=True):
        # Save upload to temp file
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # ---- Step 1: Transcription ----
        with st.status("Transcribing audio...", expanded=True) as status:
            st.write(f"Loading Whisper model ({model_size})...")
            try:
                model = load_model(model_size)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

            st.write("Running transcription...")
            result = transcribe(model, tmp_path)
            status.update(label="Transcription complete!", state="complete")

        # ---- Step 2: Analysis ----
        with st.status("Analyzing transcript...", expanded=True) as status:
            st.write("Extracting key points and categorizing...")
            analysis = analyze(
                result["text"],
                api_key=api_key if api_key else None,
                use_openai=use_openai,
            )
            status.update(label="Analysis complete!", state="complete")

        # ---- Display results ----
        st.divider()

        tab_summary, tab_key_points, tab_categories, tab_transcript, tab_export = st.tabs(
            ["Summary", "Key Points", "Categorized Notes", "Full Transcript", "Export"]
        )

        # -- Summary tab --
        with tab_summary:
            st.subheader("Summary")
            st.write(analysis.summary)
            st.caption(f"Language detected: **{result['language']}** | Segments: **{len(result['segments'])}**")

        # -- Key Points tab --
        with tab_key_points:
            st.subheader("Key Points")
            if analysis.key_points:
                for i, point in enumerate(analysis.key_points, 1):
                    st.markdown(
                        f'<div class="key-point"><strong>{i}.</strong> {point}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No key points extracted.")

        # -- Categories tab --
        with tab_categories:
            st.subheader("Categorized Notes")
            if analysis.categories:
                for category, items in analysis.categories.items():
                    st.markdown(
                        f'<div class="category-header">{category}</div>',
                        unsafe_allow_html=True,
                    )
                    for item in items:
                        st.markdown(f"- {item}")
            else:
                st.info("No categories identified.")

        # -- Full Transcript tab --
        with tab_transcript:
            st.subheader("Full Transcript")

            show_timestamps = st.checkbox("Show timestamps", value=True)

            if show_timestamps and result["segments"]:
                for seg in result["segments"]:
                    ts = format_timestamp(seg["start"])
                    st.markdown(
                        f'<div class="transcript-segment">'
                        f'<span class="timestamp">[{ts}]</span> {seg["text"]}'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.write(result["text"])

        # -- Export tab --
        with tab_export:
            st.subheader("Export")

            # Markdown export
            md_content = f"# Voice Memo Notes\n\n**Source:** {uploaded_file.name}\n\n"
            md_content += analysis.to_markdown()
            md_content += "\n---\n\n## Full Transcript\n\n"
            if result["segments"]:
                for seg in result["segments"]:
                    ts = format_timestamp(seg["start"])
                    md_content += f"**[{ts}]** {seg['text']}\n\n"
            else:
                md_content += result["text"]

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download as Markdown",
                    data=md_content,
                    file_name=f"{Path(uploaded_file.name).stem}_notes.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col2:
                import json

                export_data = {
                    "source_file": uploaded_file.name,
                    "language": result["language"],
                    "transcription": result["text"],
                    "segments": result["segments"],
                    "analysis": analysis.to_dict(),
                }
                st.download_button(
                    label="Download as JSON",
                    data=json.dumps(export_data, indent=2, ensure_ascii=False),
                    file_name=f"{Path(uploaded_file.name).stem}_notes.json",
                    mime="application/json",
                    use_container_width=True,
                )

            st.divider()
            st.subheader("Preview (Markdown)")
            st.markdown(md_content)

        # Cleanup temp file
        Path(tmp_path).unlink(missing_ok=True)

else:
    # Empty state
    st.info("Upload an audio file to get started. Supported formats: `.m4a`, `.caf`, `.wav`, `.mp3`")
