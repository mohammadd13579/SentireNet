import streamlit as st
import pandas as pd
from sentirenet_analyzer import SentireNetAnalyzer
from visualizer import draw_graph
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Project SentireNet",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- State Management ---
# Initialize session state variables
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'simulation_summary' not in st.session_state:
    st.session_state.simulation_summary = ""

# --- Caching ---
@st.cache_data
def load_data(uploaded_file):
    """Loads data from the uploaded file."""
    if uploaded_file:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            return None
    return None

@st.cache_resource
def get_analyzer(df, text_column):
    """Initializes the SentireNetAnalyzer."""
    return SentireNetAnalyzer(df, text_column)

# --- Sidebar (Controls) ---
st.sidebar.title("SentireNet Controls")
st.sidebar.markdown("Upload your text data and configure the analysis.")

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

df = None
if uploaded_file:
    df = load_data(uploaded_file)
elif os.path.exists("sample_data.csv"):
    st.sidebar.success("Loaded default `sample_data.csv`. Upload your own to explore!")
    df = load_data("sample_data.csv")

if df is not None:
    # --- Column Selection ---
    available_columns = df.columns
    text_column = st.sidebar.selectbox(
        "Which column contains the text to analyze?",
        available_columns,
        index=len(available_columns) - 1 if len(available_columns) > 0 else 0
    )
    
    # --- Analysis Parameters ---
    st.sidebar.subheader("Analysis Parameters")
    sample_size = st.sidebar.slider(
        "Number of Samples",
        min_value=10,
        max_value=min(2000, len(df)),
        value=min(100, len(df)),
        help="Number of text entries to process. Larger numbers are slower but more accurate."
    )
    
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold (Min)",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Minimum semantic similarity (0-1) to draw a line between two concepts."
    )

    # --- Run Analysis Button ---
    if st.sidebar.button("Build Semantic Web", type="primary"):
        if text_column:
            with st.spinner("Analyzing text and building graph... This may take a moment."):
                try:
                    # Initialize analyzer
                    analyzer = get_analyzer(df, text_column)
                    
                    # Build graph
                    graph = analyzer.build_graph(sample_size, similarity_threshold)
                    
                    # Store in session state
                    st.session_state.analyzer = analyzer
                    st.session_state.graph = graph
                    
                    if graph is None or not graph.nodes:
                         st.error("Could not build graph. Try adjusting parameters or check your data.")
                    else:
                        st.success(f"Graph built successfully with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    # You might want to log the full traceback here
                    print(f"Error: {e}")
        else:
            st.sidebar.error("Please select a valid text column.")

# --- Main Page Layout ---
st.title("🕸️ Project SentireNet")
st.markdown("### The Semantic & Emotional Web Analyzer")

# --- Tabbed Interface ---
tab1, tab2 = st.tabs(["Semantic Web Visualizer", "Rhetorical Pathways Simulation"])

with tab1:
    st.header("Semantic Web Visualization")
    st.markdown("This graph shows the connections between key concepts in your text. Node size represents frequency, and color represents sentiment (Red=Negative, Blue=Positive).")
    
    # Display the graph
    if st.session_state.graph:
        with st.spinner("Generating interactive visualization..."):
            fig = draw_graph(st.session_state.graph)
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("Click 'Build Semantic Web' in the sidebar to generate a visualization.")

with tab2:
    st.header("Rhetorical Pathways Simulation")
    st.markdown("Curious how a new idea would be received? Type in a sentence and see the likely path it would take through the existing web of ideas.")
    
    if not st.session_state.graph:
        st.info("You must build a semantic web first (see Tab 1) before running a simulation.")
    else:
        start_idea = st.text_input(
            "Enter a new idea, phrase, or sentence:",
            value="This technology is all about creative freedom."
        )
        
        if st.button("Simulate Pathway", type="primary"):
            analyzer = st.session_state.analyzer
            if analyzer and start_idea:
                with st.spinner("Simulating..."):
                    try:
                        paths, summary = analyzer.simulate_pathway(start_idea, steps=5, path_count=3)
                        st.session_state.simulation_summary = summary
                    except Exception as e:
                        st.error(f"Error during simulation: {e}")
                        print(f"Simulation Error: {e}")
            else:
                st.error("Analyzer not ready or no idea provided.")
        
        # Display simulation results
        if st.session_state.simulation_summary:
            st.subheader("Simulation Results")
            st.text_area("Likely Pathways:", st.session_state.simulation_summary, height=300)

        else:
            st.info("Please upload a CSV file (or use the default `sample_data.csv`) to begin.")