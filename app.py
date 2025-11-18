import streamlit as st
import pandas as pd
from visualizer import draw_graph
from sentirenet_analyzer import SentireNetAnalyzer
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Project SentireNet",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- State Management ---
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'simulation_summary' not in st.session_state:
    st.session_state.simulation_summary = ""

# --- Caching ---
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            return None
    return None

# --- Sidebar ---
st.sidebar.title("SentireNet Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Use sample data if no file uploaded
if not uploaded_file and os.path.exists("sample_data.csv"):
    st.sidebar.info("Using sample_data.csv")
    df = pd.read_csv("sample_data.csv")
elif uploaded_file:
    df = load_data(uploaded_file)
else:
    df = None

# --- Main Layout ---
st.title("Project SentireNet")
st.markdown("### The Semantic & Emotional Web Analyzer")

tab1, tab2 = st.tabs(["🕸️ Network Visualizer", "🔮 Rhetorical Simulation"])

with tab1:
    if df is not None:
        text_col = st.selectbox("Select Text Column", df.columns)
        threshold = st.slider("Similarity Threshold (Edges)", 0.1, 0.9, 0.4)
        
        if st.button("Build Graph"):
            with st.spinner("Analyzing text and building graph..."):
                try:
                    analyzer = SentireNetAnalyzer(df, text_col)
                    # Build the graph
                    graph = analyzer.build_graph(similarity_threshold=threshold)
                    
                    # Store in session state
                    st.session_state.analyzer = analyzer
                    st.session_state.graph = graph
                    st.success(f"Graph built with {len(graph.nodes)} concepts and {len(graph.edges)} connections.")
                except Exception as e:
                    st.error(f"An error occurred during graph building: {e}")

    # Draw Graph (if exists)
    if st.session_state.graph:
        fig = draw_graph(st.session_state.graph)
        # FIX: Replaced use_container_width with width="stretch" as requested by the warning
        try:
            st.plotly_chart(fig, width="stretch")
        except Exception:
            # Fallback for older Streamlit versions
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload data and build the graph to view visualizations.")

with tab2:
    st.subheader("Rhetorical Pathways Simulation")
    if not st.session_state.graph:
        st.warning("Please build the graph in Tab 1 first.")
    else:
        user_idea = st.text_input("Enter a new idea/phrase:", "Artificial intelligence empowers creativity")
        if st.button("Simulate"):
            analyzer = st.session_state.analyzer
            try:
                paths, summary = analyzer.simulate_pathway(user_idea)
                st.session_state.simulation_summary = summary
            except ValueError as e:
                st.error(f"Simulation math error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
            
        if st.session_state.simulation_summary:
            st.text_area("Simulation Results", st.session_state.simulation_summary, height=300)