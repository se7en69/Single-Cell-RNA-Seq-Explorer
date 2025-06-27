import streamlit as st
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import os
import time
import hashlib
from io import StringIO
import altair as alt
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Single-Cell Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'adata' not in st.session_state:
    st.session_state.adata = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'clustered' not in st.session_state:
    st.session_state.clustered = False
if 'current_file_hash' not in st.session_state:
    st.session_state.current_file_hash = None

# Custom hashing function for AnnData objects
def hash_adata(adata):
    if adata is None:
        return None
    m = hashlib.sha256()
    m.update(str(adata.shape).encode())
    m.update(str(adata.obs.shape).encode())
    m.update(str(adata.var.shape).encode())
    if adata.n_obs > 0 and adata.n_vars > 0:
        m.update(adata.X[:10,:10].tobytes())
    return m.hexdigest()

# Cache settings with custom hashing
@st.cache_resource(show_spinner=False, max_entries=3)
def load_data_cached(uploaded_file, example_dataset):
    return load_data(uploaded_file, example_dataset)

@st.cache_resource(show_spinner=False, hash_funcs={"_AnnData": hash_adata})
def preprocess_data_cached(_adata, normalize, log_transform, scale, min_genes, min_cells):
    return preprocess_data(_adata, normalize, log_transform, scale, min_genes, min_cells)

@st.cache_resource(show_spinner=False, hash_funcs={"_AnnData": hash_adata})
def run_dim_reduction_cached(_adata, n_pcs, n_neighbors, perplexity):
    return run_dim_reduction(_adata, n_pcs, n_neighbors, perplexity)

@st.cache_resource(show_spinner=False, hash_funcs={"_AnnData": hash_adata})
def run_clustering_cached(_adata, resolution):
    return run_clustering(_adata, resolution)

@st.cache_resource(show_spinner=False, hash_funcs={"_AnnData": hash_adata})
def run_diff_exp_cached(_adata, cluster1, cluster2):
    return run_diff_exp(_adata, cluster1, cluster2)

# Data loading function
def load_data(uploaded_file, example_dataset):
    if uploaded_file is not None:
        current_file_hash = hash(uploaded_file.getvalue())
        if st.session_state.current_file_hash == current_file_hash and st.session_state.adata is not None:
            return st.session_state.adata
            
        try:
            progress_bar = st.progress(0, text="Loading data...")
            
            if uploaded_file.name.endswith('.h5ad'):
                with st.spinner("Reading h5ad file..."):
                    adata = sc.read_h5ad(uploaded_file)
                st.session_state.file_type = "h5ad"
            else:
                with st.spinner("Reading CSV/TSV file..."):
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file, index_col=0)
                    else:
                        df = pd.read_csv(uploaded_file, sep='\t', index_col=0)
                
                with st.spinner("Creating AnnData object..."):
                    adata = sc.AnnData(df.T if df.shape[0] < df.shape[1] else df)
                st.session_state.file_type = "matrix"
            
            progress_bar.progress(100, text="Data loaded successfully!")
            time.sleep(0.5)
            progress_bar.empty()
            
            st.session_state.adata = adata
            st.session_state.current_file_hash = current_file_hash
            return adata
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    elif example_dataset != "None":
        try:
            progress_bar = st.progress(0, text="Loading example dataset...")
            
            with st.spinner(f"Loading {example_dataset} dataset..."):
                if example_dataset == "pbmc3k":
                    adata = sc.datasets.pbmc3k()
                elif example_dataset == "pancreas":
                    adata = sc.datasets.pancreas()
                elif example_dataset == "tabula_muris":
                    adata = sc.datasets.tabulamuris()
            
            progress_bar.progress(100, text="Example dataset loaded!")
            time.sleep(0.5)
            progress_bar.empty()
            
            st.session_state.adata = adata
            st.session_state.file_type = "example"
            return adata
        
        except Exception as e:
            st.error(f"Error loading example dataset: {str(e)}")
            return None
    
    else:
        return None

# Preprocessing function
def preprocess_data(adata, normalize=True, log_transform=True, scale=True, min_genes=200, min_cells=3):
    if adata is None:
        return None
    
    progress_bar = st.progress(0, text="Preprocessing data...")
    
    try:
        with st.spinner("Filtering cells and genes..."):
            sc.pp.filter_cells(adata, min_genes=min_genes)
            sc.pp.filter_genes(adata, min_cells=min_cells)
            progress_bar.progress(20, text="Filtering complete")
        
        if normalize:
            with st.spinner("Normalizing data..."):
                sc.pp.normalize_total(adata, target_sum=1e4)
                progress_bar.progress(40, text="Normalization complete")
        
        if log_transform:
            with st.spinner("Log transforming..."):
                sc.pp.log1p(adata)
                progress_bar.progress(60, text="Log transform complete")
        
        with st.spinner("Finding highly variable genes..."):
            sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
            progress_bar.progress(80, text="Variable genes identified")
        
        if scale:
            with st.spinner("Scaling data..."):
                sc.pp.scale(adata, max_value=10)
                progress_bar.progress(100, text="Scaling complete")
        
        st.session_state.processed = True
        return adata
    
    except Exception as e:
        st.error(f"Error during preprocessing: {str(e)}")
        return None
    finally:
        time.sleep(0.5)
        progress_bar.empty()

# Dimensionality reduction function
def run_dim_reduction(adata, n_pcs=50, n_neighbors=15, perplexity=30):
    if adata is None:
        return None
    
    progress_bar = st.progress(0, text="Running dimensionality reduction...")
    
    try:
        with st.spinner("Running PCA..."):
            sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs)
            progress_bar.progress(30, text="PCA completed")
        
        with st.spinner("Computing neighborhood graph..."):
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
            progress_bar.progress(60, text="Neighborhood graph computed")
        
        with st.spinner("Running UMAP..."):
            sc.tl.umap(adata)
            progress_bar.progress(80, text="UMAP completed")
        
        with st.spinner("Running t-SNE..."):
            sc.tl.tsne(adata, perplexity=perplexity, n_pcs=n_pcs)
            progress_bar.progress(100, text="t-SNE completed")
        
        return adata
    
    except Exception as e:
        st.error(f"Error during dimensionality reduction: {str(e)}")
        return None
    finally:
        time.sleep(0.5)
        progress_bar.empty()

# Clustering function (Leiden only)
def run_clustering(adata, resolution=0.5):
    if adata is None:
        return None
    
    progress_bar = st.progress(0, text="Running clustering...")
    
    try:
        with st.spinner("Running Leiden clustering..."):
            sc.tl.leiden(adata, resolution=resolution)
            progress_bar.progress(100, text="Leiden clustering completed")
            st.session_state.clustered = True
            return adata
        
    except Exception as e:
        st.error(f"Error during clustering: {str(e)}")
        return None
    finally:
        time.sleep(0.5)
        progress_bar.empty()

# Differential expression function
def run_diff_exp(adata, cluster1, cluster2=None):
    if adata is None:
        return None
    
    cluster_key = 'leiden'
    progress_bar = st.progress(0, text="Running differential expression...")
    
    try:
        with st.spinner("Identifying marker genes..."):
            if cluster2 is None:
                sc.tl.rank_genes_groups(adata, cluster_key, groups=[cluster1], reference='rest')
            else:
                sc.tl.rank_genes_groups(adata, cluster_key, groups=[cluster1], reference=cluster2)
            progress_bar.progress(100, text="Differential expression completed")
        
        return adata
    
    except Exception as e:
        st.error(f"Error during differential expression: {str(e)}")
        return None
    finally:
        time.sleep(0.5)
        progress_bar.empty()

# Visualization functions
def create_interactive_plot(adata, plot_type, color_by=None, gene=None):
    try:
        # Get the correct coordinates based on plot type
        if plot_type == "UMAP":
            coords = adata.obsm['X_umap']
            title = "UMAP Projection"
        elif plot_type == "t-SNE":
            coords = adata.obsm['X_tsne']  # Note: 'tsne' without hyphen
            title = "t-SNE Projection"
        else:  # PCA
            coords = adata.obsm['X_pca'][:, :2]
            title = "PCA Projection (First 2 PCs)"
        
        plot_df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1]
        })
        
        if color_by is not None and color_by != "None":
            if color_by == "Expression":
                plot_df['color'] = adata[:, gene].X.toarray().flatten()
                color_label = f"Expression: {gene}"
                color_scale = 'Viridis'
            else:
                plot_df['color'] = adata.obs[color_by].values
                color_label = color_by
                if pd.api.types.is_numeric_dtype(plot_df['color']):
                    color_scale = 'Viridis'
                else:
                    color_scale = None
            
            fig = px.scatter(plot_df, x='x', y='y', color='color',
                           color_continuous_scale=color_scale,
                           title=f"{title} - Colored by {color_label}",
                           labels={'color': color_label},
                           hover_data={'x': False, 'y': False, 'color': True})
        else:
            fig = px.scatter(plot_df, x='x', y='y', title=title)
        
        fig.update_layout(
            hovermode='closest',
            plot_bgcolor='rgba(240,240,240,0.9)',
            paper_bgcolor='rgba(240,240,240,0.5)',
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title=f"{plot_type} 1",
            yaxis_title=f"{plot_type} 2"
        )
        
        fig.update_traces(
            marker=dict(size=4, line=dict(width=0.5, color='DarkSlateGrey')),
            selector=dict(mode='markers')
        )
        
        return fig
    
    except KeyError as e:
        st.error(f"Could not find coordinates for {plot_type}. Please run dimensionality reduction first.")
        return None
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None
# Main app function
def main():
    # Sidebar navigation
    st.sidebar.image('images/logo.png', width=300)
    st.sidebar.title("Single-Cell Explorer 🧬")
    st.sidebar.write("**Interactive analysis of single-cell RNA-seq data** 🔍")
    st.sidebar.markdown("---")
    
    # App introduction
    st.sidebar.header("Introduction")
    st.sidebar.write(
        """
        Welcome to **Single-Cell Explorer**!
        This app provides interactive visualization and analysis of single-cell RNA sequencing data.
        Upload your data or explore example datasets to perform clustering, dimensionality reduction,
        and differential expression analysis.
        """
    )
    st.sidebar.markdown("---")
    
    # Navigation options
    app_mode = st.sidebar.radio(
        "Navigate the App",
        ["Data Upload", "Preprocessing", "Visualization", "Analysis", "Help"]
    )
    st.sidebar.markdown("---")
    
    # Data input section (shown for all analysis tabs)
    if app_mode in ["Data Upload", "Preprocessing", "Visualization", "Analysis"]:
        st.sidebar.header("Data Input")
        uploaded_file = st.sidebar.file_uploader(
            "Upload scRNA-seq data",
            type=['csv', 'tsv', 'h5ad'],
            help="Upload gene expression matrix (CSV/TSV) or AnnData object (h5ad)"
        )
        
        example_dataset = st.sidebar.selectbox(
            "Or use example dataset",
            ["None", "pbmc3k", "pancreas", "tabula_muris"],
            help="Select an example dataset to explore"
        )
    
    # Conditional sections based on navigation
    if app_mode == "Preprocessing":
        st.sidebar.header("Preprocessing Options")
        with st.sidebar.expander("Settings"):
            normalize = st.checkbox("Normalize data (CPM)", True)
            log_transform = st.checkbox("Log transform (log1p)", True)
            scale = st.checkbox("Scale data (z-score)", True)
            min_genes = st.number_input("Minimum genes per cell", 200, 5000, 200)
            min_cells = st.number_input("Minimum cells per gene", 3, 100, 3)

    elif app_mode == "Visualization":
        st.sidebar.header("Visualization Settings")
        with st.sidebar.expander("Dimensionality Reduction"):
            n_pcs = st.slider("Number of PCs", 10, 100, 50)
            n_neighbors = st.slider("Number of neighbors", 5, 100, 15)
            perplexity = st.slider("t-SNE perplexity", 5, 100, 30)

    elif app_mode == "Analysis":
        st.sidebar.header("Analysis Parameters")
        with st.sidebar.expander("Clustering"):
            resolution = st.slider("Resolution", 0.1, 2.0, 0.5, 0.1)
    
    st.sidebar.markdown("---")
    
    # About section
    st.sidebar.header("About")
    st.sidebar.write(
        """
        This tool is designed for researchers to explore single-cell RNA sequencing data.
        For technical support or feature requests, please contact us.
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.write("Created by [Abdul Rehman Ikram]. For feedback, contact: [hanzo7n@gmail.com](mailto:hanzo7n@gmail.com)")
    
    # Main content area
    if app_mode == "Data Upload":
        st.header("Data Upload")
        st.write("Upload your single-cell RNA-seq data to begin analysis.")
        
        adata = load_data_cached(uploaded_file, example_dataset)
        
        if st.session_state.adata is not None:
            adata = st.session_state.adata
            
            with st.expander("Dataset Information", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Number of cells", adata.n_obs)
                with col2:
                    st.metric("Number of genes", adata.n_vars)
                with col3:
                    st.metric("Size in memory", f"{adata.__sizeof__()/1e6:.2f} MB")
                
                if adata.obs.shape[1] > 0:
                    st.subheader("Cell Metadata Preview")
                    st.dataframe(adata.obs.head())
        
        else:
            st.warning("Please upload data or select an example dataset to begin analysis")
    
    elif app_mode == "Preprocessing":
        st.header("Data Preprocessing")
        
        if st.session_state.adata is None:
            st.warning("Please upload data first in the Data Upload section")
            return
        
        if st.button("Run Preprocessing", help="Apply normalization, filtering, and scaling"):
            with st.spinner("Preprocessing data..."):
                adata = preprocess_data_cached(
                    st.session_state.adata,
                    normalize=normalize,
                    log_transform=log_transform,
                    scale=scale,
                    min_genes=min_genes,
                    min_cells=min_cells
                )
                st.session_state.processed = True
                st.session_state.adata = adata
                st.success("Data preprocessing completed!")
        
        if st.session_state.processed:
            st.subheader("Preprocessing Results")
            st.write("Your data has been successfully preprocessed. You can now proceed to visualization.")
    
    elif app_mode == "Visualization":
        st.header("Data Visualization")
        
        if st.session_state.adata is None:
            st.warning("Please upload and preprocess data first")
            return
        
        if not st.session_state.processed:
            st.warning("Please preprocess your data first in the Preprocessing section")
            return
        
        if st.button("Run Dimensionality Reduction", help="Compute PCA, UMAP, and t-SNE embeddings"):
            with st.spinner("Running dimensionality reduction..."):
                adata = run_dim_reduction_cached(
                    st.session_state.adata,
                    n_pcs=n_pcs,
                    n_neighbors=n_neighbors,
                    perplexity=perplexity
                )
                st.session_state.adata = adata
                st.success("Dimensionality reduction completed!")
        
        # Check which dimensionality reduction results are available
        has_umap = 'X_umap' in st.session_state.adata.obsm
        has_tsne = 'X_tsne' in st.session_state.adata.obsm
        has_pca = 'X_pca' in st.session_state.adata.obsm
        
        if not (has_umap or has_tsne or has_pca):
            st.warning("No dimensionality reduction results found. Please run dimensionality reduction first.")
            return
        
        viz_col1, viz_col2 = st.columns([1, 3])
        
        with viz_col1:
            st.subheader("Plot Settings")
            
            # Only show available plot types
            available_plots = []
            if has_umap: available_plots.append("UMAP")
            if has_tsne: available_plots.append("t-SNE")
            if has_pca: available_plots.append("PCA")
            
            plot_type = st.selectbox(
                "Plot type",
                available_plots,
                key="plot_type_select"
            )
            
            color_by = st.selectbox(
                "Color by",
                ["None"] + list(st.session_state.adata.obs.columns) + ["Expression"],
                key="color_by_select"
            )
            
            if color_by == "Expression":
                gene = st.selectbox(
                    "Select gene",
                    st.session_state.adata.var_names.tolist(),
                    key="gene_select"
                )
            else:
                gene = None
            
            st.markdown("**Download Options**")
            
            # Mapping between display names and obsm keys
            plot_key_map = {
                "UMAP": "umap",
                "t-SNE": "tsne",  # Note: 'tsne' without hyphen
                "PCA": "pca"
            }
            
            plot_key = f'X_{plot_key_map[plot_type]}'
            
            if plot_key in st.session_state.adata.obsm:
                st.download_button(
                    label="📥 Download Coordinates",
                    data=pd.DataFrame(st.session_state.adata.obsm[plot_key]).to_csv(),
                    file_name=f"{plot_type}_coordinates.csv",
                    mime="text/csv"
                )
        
        with viz_col2:
            fig = create_interactive_plot(st.session_state.adata, plot_type, color_by, gene)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

    elif app_mode == "Analysis":
        st.header("Data Analysis")
        
        if st.session_state.adata is None:
            st.warning("Please upload and preprocess data first")
            return
        
        if not st.session_state.processed:
            st.warning("Please preprocess your data first in the Preprocessing section")
            return
        
        if st.button("Run Clustering", help="Perform cell clustering using Leiden algorithm"):
            with st.spinner("Running Leiden clustering..."):
                adata = run_clustering_cached(
                    st.session_state.adata,
                    resolution=resolution
                )
                st.session_state.clustered = True
                st.session_state.adata = adata
                st.success("Clustering completed!")
        
        if st.session_state.clustered:
            cluster_key = 'leiden'
            
            st.subheader("Cluster Analysis")
            cluster_counts = st.session_state.adata.obs[cluster_key].value_counts().sort_index()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.altair_chart(alt.Chart(pd.DataFrame({
                    'Cluster': cluster_counts.index.astype(str),
                    'Count': cluster_counts.values
                })).mark_bar().encode(
                    x='Cluster:N',
                    y='Count:Q',
                    color='Cluster:N'
                ), use_container_width=True)
                
                st.download_button(
                    label="📥 Download Cluster Data",
                    data=cluster_counts.to_csv(),
                    file_name="cluster_counts.csv",
                    mime="text/csv"
                )
            
            with col2:
                cluster_fig = create_interactive_plot(st.session_state.adata, "UMAP", cluster_key)
                st.plotly_chart(cluster_fig, use_container_width=True)
            
            st.subheader("Differential Expression Analysis")

            # Form for input parameters only
            with st.form("diff_exp_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    cluster1 = st.selectbox(
                        "Select cluster",
                        sorted(st.session_state.adata.obs[cluster_key].unique()),
                        key="cluster1_select"
                    )
                
                with col2:
                    compare_option = st.radio(
                        "Comparison type",
                        ["vs all other cells", "vs specific cluster"],
                        key="compare_option"
                    )
                    
                    if compare_option == "vs specific cluster":
                        cluster2 = st.selectbox(
                            "Select comparison cluster",
                            sorted(st.session_state.adata.obs[cluster_key].unique()),
                            key="cluster2_select"
                        )
                    else:
                        cluster2 = None
                
                submitted = st.form_submit_button("Run Differential Expression")

            # Results display and downloads (outside the form)
            if submitted:
                with st.spinner("Running differential expression analysis..."):
                    adata = run_diff_exp_cached(st.session_state.adata, cluster1, cluster2)
                    st.session_state.adata = adata
                    st.success("Analysis completed!")
                    
                    tab1, tab2, tab3 = st.tabs(["Results Table", "Volcano Plot", "Heatmap"])
                    
                    with tab1:
                        if compare_option == "vs all other cells":
                            results = sc.get.rank_genes_groups_df(adata, group=cluster1)
                        else:
                            results = sc.get.rank_genes_groups_df(adata, group=cluster1, reference=cluster2)
                        
                        results = results[results['pvals_adj'] < 0.05]
                        results = results.sort_values('scores', ascending=False)
                        
                        st.dataframe(results.head(20))
                        
                        # CSV Download Button (now outside form)
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="diff_exp_results.csv",
                            mime="text/csv"
                        )
                    with tab2:
                        st.subheader("Volcano Plot")
                        
                        # Add transformed column
                        results['neg_log10_pvals_adj'] = -np.log10(results['pvals_adj'])
                        
                        volcano_fig = px.scatter(
                            results,
                            x='logfoldchanges',
                            y='neg_log10_pvals_adj',
                            hover_name='names',
                            color='scores',
                            color_continuous_scale='Viridis',
                            title="Volcano Plot of Differential Expression",
                            labels={
                                'logfoldchanges': 'log2(Fold Change)',
                                'neg_log10_pvals_adj': '-log10(Adjusted p-value)'
                            }
                        )
                        st.plotly_chart(volcano_fig, use_container_width=True)                    
                    with tab3:
                        st.subheader("Heatmap of Top Genes")
                        
                        top_genes = results.head(10)['names'].tolist()
                        temp_adata = adata[:, top_genes].copy()
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(
                            temp_adata.X.T,
                            index=temp_adata.var_names,
                            columns=temp_adata.obs[cluster_key]
                        )
                        
                        # Plot with Seaborn
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.heatmap(df, ax=ax, cmap='viridis')
                        
                        # Display and download
                        st.pyplot(fig)
                        
                        buf = BytesIO()
                        fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
                        buf.seek(0)
                        
                        st.download_button(
                            label="Download Heatmap as PNG",
                            data=buf,
                            file_name="diff_exp_heatmap.png",
                            mime="image/png"
                        )
                        
                        plt.close(fig)
    elif app_mode == "Help":
        st.header("Help & Documentation")
        
        st.markdown("""
        ## Getting Started
        
        1. **Upload your data** or select an example dataset
        2. **Preprocess** your data (normalization, filtering, scaling)
        3. **Visualize** your data using dimensionality reduction techniques
        4. **Analyze** clusters and perform differential expression
        
        ## File Formats Supported
        
        - **CSV/TSV**: Gene expression matrix (genes × cells or cells × genes)
        - **h5ad**: AnnData format (preferred for metadata preservation)
        
        ## Example Datasets
        
        - **pbmc3k**: 2,700 PBMCs from a healthy donor
        - **pancreas**: Human pancreatic islet scRNA-seq data
        - **tabula_muris**: 100,000 cells from 20 mouse organs and tissues
        
        ## Troubleshooting
        
        - If you encounter errors during preprocessing, try adjusting the filtering parameters
        - For visualization issues, try reducing the number of principal components
        - Clustering results can be adjusted using the resolution parameter
        
        ## Contact
        
        For technical support or feature requests, please contact [hanzo7n@gmail.com](mailto:hanzo7n@gmail.com)
        """)

if __name__ == "__main__":
    main()