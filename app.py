import streamlit as st
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
import time
import logging
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log Streamlit version
logger.debug("Streamlit version: %s", st.__version__)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .stButton>button {
        background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 20px; font-size: 16px;
    }
    .stButton>button:hover { background-color: #45a049; }
    .stTextInput>div>input { border-radius: 8px; border: 1px solid #ddd; padding: 10px; }
    .stFileUploader>div>label { color: #333; font-size: 16px; }
    h1, h2, h3 { color: #2c3e50; }
    .sidebar .sidebar-content { background-color: #ffffff; border-right: 1px solid #ddd; }
    .stAlert { border-radius: 8px; }
    .stExpander { background-color: #111111; border-radius: 8px; border: 1px solid #ddd; }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("🛠️ Tunnel Stability Predictor")
st.markdown("""
Welcome to the **Tunnel Stability Predictor**! This app allows you to:
- Predict the stability number ratio (N) using a pre-trained Random Forest model.
- Upload a dataset to train a new model, visualize results, and make predictions.
- Explore dataset statistics, metrics, hyperparameters, and plots.
""")

# How to Use
with st.expander("📖 How to Use", expanded=False):
    st.markdown("""
    ### Getting Started
    1. **Select Mode**: Choose **Prediction** or **Train New Model** in the sidebar.
    2. **Prediction Mode**:
        - Enter positive values for `Sigma Ci`, `GSI`, `mi`, `C/H`, `r/B`, `e/B`.
        - Click **Predict** to get the stability number ratio (N).
    3. **Train New Model Mode**:
        - Upload a CSV with columns: `Sigma Ci`, `GSI`, `mi`, `C/H`, `r/B`, `e/B`, `N`.
        - View **Dataset Description** (statistics, sample, correlation heatmap).
        - The app trains a Random Forest model and displays:
            - Metrics (R², MAE, MSE, RMSE).
            - Hyperparameters from Optuna (50 trials, 10-fold CV).
            - Plots: Feature Importance, Actual vs Predicted, Residual, Error Distribution, Learning Curves.
        - Predict with the new model using the form (positive values only).
        - Models are saved as `random_forest_model_<timestamp>.pkl`.
    4. **Tips**:
        - Ensure no missing values in the dataset.
        - Use realistic positive values for predictions.
    """)

# Sidebar
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Prediction", "Train New Model"])
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)

# Initialize session state
def initialize_session_state():
    defaults = {
        'app_state': "idle",
        'uploaded_file': None,
        'best_rf': None,
        'scaler': None,
        'poly': None,
        'feature_names': None,
        'file_uploader_key': 0,
        'training_lock': False,
        'render_phase': "idle",
        'cached_results': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    logger.debug("Session state initialized: %s", {k: str(v)[:50] for k, v in st.session_state.items()})

initialize_session_state()

# Preprocess data
@st.cache_data
def preprocess_data(data):
    logger.debug("Preprocessing data, shape=%s", data.shape)
    try:
        required_columns = ['Sigma Ci', 'GSI', 'mi', 'C/H', 'r/B', 'e/B', 'N']
        if not all(col in data.columns for col in required_columns):
            logger.error("Missing columns: %s", required_columns)
            st.error(f"Dataset must contain: {', '.join(required_columns)}")
            return None, None, None, None, None
        
        if data[required_columns].isnull().any().any():
            logger.error("NaN values detected")
            st.error("Dataset contains NaN values.")
            return None, None, None, None, None
        
        if len(data) < 20:
            logger.error("Dataset too small: %s rows", len(data))
            st.error("Dataset must have at least 20 rows.")
            return None, None, None, None, None
        
        if data['N'].nunique() <= 1:
            logger.error("Target 'N' has no variance")
            st.error("Target 'N' must have varying values.")
            return None, None, None, None, None
        
        data = data[data['N'] < 100].copy()
        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        X = data[['Sigma Ci', 'GSI', 'mi', 'C/H', 'r/B', 'e/B']].copy()
        
        if X.empty:
            logger.error("Empty features after outlier removal")
            st.error("No data remains after outlier removal.")
            return None, None, None, None, None
        
        if (data['C/H'] <= 0).any() or (data['r/B'] <= 0).any():
            logger.warning("Zero/negative values in 'C/H' or 'r/B'")
            st.warning("Zero/negative values in 'C/H' or 'r/B'. Replacing with 1e-6.")
            data['C/H'] = data['C/H'].replace(0, 1e-6).clip(lower=1e-6)
            data['r/B'] = data['r/B'].replace(0, 1e-6).clip(lower=1e-6)
        
        data['Sigma Ci'] = data['Sigma Ci'].clip(lower=1e-6, upper=1e6)
        data['GSI'] = data['GSI'].clip(lower=1e-6, upper=1e6)
        
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(['Sigma Ci', 'GSI', 'mi', 'C/H', 'r/B', 'e/B'])
        X_poly_df = pd.DataFrame(X_poly, columns=feature_names)
        
        X_poly_df['log_C/H'] = np.log1p(data['C/H'])
        X_poly_df['log_r/B'] = np.log1p(data['r/B'])
        X_poly_df['exp_C/H'] = np.exp(-data['C/H'])
        X_poly_df['exp_Sigma_Ci'] = np.exp(-data['Sigma Ci'] / (data['Sigma Ci'].max() + 1e-6))
        X_poly_df['GSI_over_Sigma_Ci'] = data['GSI'] / (data['Sigma Ci'] + 1e-6)
        X_poly_df['mi_times_C/H'] = data['mi'] * data['C/H']
        
        if X_poly_df.isnull().any().any() or np.isinf(X_poly_df.values).any():
            nan_columns = X_poly_df.columns[X_poly_df.isnull().any()].tolist()
            inf_columns = X_poly_df.columns[np.isinf(X_poly_df.values).any(axis=0)].tolist()
            logger.warning("NaN/infinite values after transformation. NaN columns: %s, Inf columns: %s", nan_columns, inf_columns)
            st.warning("NaN/infinite values detected. Dropping rows.")
            initial_rows = len(X_poly_df)
            X_poly_df = X_poly_df.dropna()
            y = data.loc[X_poly_df.index, 'N'].copy()
            st.info(f"Dropped {initial_rows - len(X_poly_df)} rows.")
        else:
            y = data['N'].copy()
        
        if y.empty:
            logger.error("Empty target data")
            st.error("Target data is empty.")
            return None, None, None, None, None
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_poly_df)
        
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            logger.error("NaN/infinite values in scaled data")
            st.error("Invalid values in scaled data.")
            return None, None, None, None, None
        
        logger.debug("Preprocessing successful, X_scaled shape=%s", X_scaled.shape)
        return X_scaled, y, scaler, poly, feature_names
    except Exception as e:
        logger.error("Preprocessing failed: %s", str(e))
        st.error(f"Preprocessing error: {str(e)}")
        return None, None, None, None, None

# Prediction function
def predict_stability_number(sigma_ci, gsi, mi, c_h, r_b, e_b, model, scaler, poly, feature_names):
    logger.debug("Predicting with inputs: sigma_ci=%s, gsi=%s, mi=%s, c_h=%s, r_b=%s, e_b=%s", 
                 sigma_ci, gsi, mi, c_h, r_b, e_b)
    try:
        sigma_ci = min(max(float(sigma_ci), 1e-6), 1e6)
        gsi = min(max(float(gsi), 1e-6), 1e6)
        c_h = max(float(c_h), 1e-6)
        r_b = max(float(r_b), 1e-6)
        e_b = max(float(e_b), 1e-6)
        mi = max(float(mi), 1e-6)
        
        input_data = pd.DataFrame([[sigma_ci, gsi, mi, c_h, r_b, e_b]],
                                 columns=['Sigma Ci', 'GSI', 'mi', 'C/H', 'r/B', 'e/B'])
        
        if input_data.isnull().any().any():
            raise ValueError("Input contains NaN values.")
        if (input_data <= 0).any().any():
            raise ValueError("Inputs must be positive.")
        
        input_poly = poly.transform(input_data)
        input_poly_df = pd.DataFrame(input_poly, columns=feature_names)
        
        input_poly_df['log_C/H'] = np.log1p(input_data['C/H'])
        input_poly_df['log_r/B'] = np.log1p(input_data['r/B'])
        input_poly_df['exp_C/H'] = np.exp(-input_data['C/H'])
        input_poly_df['exp_Sigma_Ci'] = np.exp(-input_data['Sigma Ci'] / (input_data['Sigma Ci'].max() + 1e-6))
        input_poly_df['GSI_over_Sigma_Ci'] = input_data['GSI'] / (input_data['Sigma Ci'] + 1e-6)
        input_poly_df['mi_times_C/H'] = input_data['mi'] * input_data['C/H']
        
        if input_poly_df.isnull().any().any() or np.isinf(input_poly_df.values).any():
            raise ValueError("Transformed inputs contain NaN/infinite values.")
        
        input_scaled = scaler.transform(input_poly_df)
        
        if np.isnan(input_scaled).any() or np.isinf(input_scaled).any():
            raise ValueError("Scaled inputs contain NaN/infinite values.")
        
        predicted_n = model.predict(input_scaled)[0]
        logger.debug("Prediction successful: %s", predicted_n)
        return predicted_n
    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        st.error(f"Prediction error: {str(e)}")
        return None

# Debug logs
if debug_mode:
    st.sidebar.subheader("Debug Logs")
    try:
        with open('app.log', 'r') as f:
            logs = f.read()
        st.sidebar.text_area("Recent Logs", logs, height=200)
    except FileNotFoundError:
        st.sidebar.write("No logs available.")
    
    st.sidebar.subheader("Session State")
    state_info = {
        "app_state": st.session_state.get('app_state', 'Not set'),
        "render_phase": st.session_state.get('render_phase', 'Not set'),
        "uploaded_file": "Set" if st.session_state.get('uploaded_file') else "None",
        "best_rf": "Set" if st.session_state.get('best_rf') is not None else "None",
        "scaler": "Set" if st.session_state.get('scaler') is not None else "None",
        "poly": "Set" if st.session_state.get('poly') is not None else "None",
        "feature_names": f"{len(st.session_state.get('feature_names', []))} features" if st.session_state.get('feature_names') is not None else "None",
        "file_uploader_key": st.session_state.get('file_uploader_key', 'Not set'),
        "training_lock": st.session_state.get('training_lock', 'Not set'),
        "cached_results": "Set" if st.session_state.get('cached_results') is not None else "None"
    }
    if state_info['render_phase'] == 'Not set':
        logger.warning("render_phase missing in session state")
    st.sidebar.json(state_info)

# Function to render cached results
def render_cached_results(cached_results):
    logger.debug("Rendering cached results")
    if not cached_results:
        st.warning("No training results available.")
        return
    
    # Dataset Description
    st.subheader("📈 Dataset Description")
    st.write(f"**Dataset Shape**: {cached_results['dataset_shape'][0]} rows, {cached_results['dataset_shape'][1]} columns")
    st.write("**Feature Statistics**:")
    st.table(cached_results['stats_df'])
    st.write("**Sample of Dataset (First 5 Rows)**:")
    st.dataframe(cached_results['sample_df'])
    if cached_results['corr_fig'] is not None:
        st.plotly_chart(cached_results['corr_fig'], use_container_width=True)
    else:
        st.warning("Correlation matrix is invalid.")
    
    # Hyperparameters
    if cached_results['params_df'] is not None:
        st.subheader("Best Hyperparameters")
        st.table(cached_results['params_df'])
        st.write(f"Best Cross-Validation R² Score: {cached_results['best_cv_r2']:.4f}")
    
    # Metrics
    if cached_results['metrics'] is not None:
        st.subheader("Model Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{cached_results['metrics']['r2']:.4f}")
            st.metric("Mean Absolute Error (MAE)", f"{cached_results['metrics']['mae']:.4f}")
        with col2:
            st.metric("Mean Squared Error (MSE)", f"{cached_results['metrics']['mse']:.4f}")
            st.metric("Root Mean Squared Error (RMSE)", f"{cached_results['metrics']['rmse']:.4f}")
    
    # Plots
    if cached_results['fig_importance'] is not None:
        st.plotly_chart(cached_results['fig_importance'], use_container_width=True)
    else:
        st.warning("Feature Importance plot unavailable.")
    
    if cached_results['fig_pred'] is not None:
        st.plotly_chart(cached_results['fig_pred'], use_container_width=True)
    else:
        st.warning("Actual vs Predicted plot unavailable.")
    
    if cached_results['fig_residual'] is not None:
        st.plotly_chart(cached_results['fig_residual'], use_container_width=True)
    else:
        st.warning("Residual Plot unavailable.")
    
    if cached_results['fig_error_dist'] is not None:
        st.plotly_chart(cached_results['fig_error_dist'], use_container_width=True)
    else:
        st.warning("Prediction Error Distribution plot unavailable.")
    
    if cached_results['fig_learning'] is not None:
        st.plotly_chart(cached_results['fig_learning'], use_container_width=True)
    else:
        st.warning("Learning Curves plot unavailable.")
    
    if cached_results['model_saved']:
        st.success(f"Model saved as {cached_results['model_filename']}.")

# Prediction Mode
if app_mode == "Prediction":
    logger.debug("Entering Prediction mode")
    st.header("🔍 Predict Stability Number")
    st.markdown("Enter positive feature values to predict the stability number ratio (N).")
    
    try:
        model = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        poly = joblib.load('poly_transformer.pkl')
        feature_names = poly.get_feature_names_out(['Sigma Ci', 'GSI', 'mi', 'C/H', 'r/B', 'e/B'])
    except FileNotFoundError:
        logger.error("Pre-trained model files not found")
        st.error("Pre-trained model files not found.")
        st.stop()
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            sigma_ci = st.number_input("Sigma Ci (MPa)", min_value=1e-6, value=50.0, step=0.0001,format="%.4f", key="sigma_ci")
            gsi = st.number_input("GSI", min_value=1e-6, value=50.0, step=0.0001,format="%.4f", key="gsi")
            mi = st.number_input("mi", min_value=1e-6, value=10.0, step=0.0001,format="%.4f", key="mi")
        with col2:
            c_h = st.number_input("C/H", min_value=1e-6, value=1.0, step=0.0001,format="%.4f", key="c_h")
            r_b = st.number_input("r/B", min_value=1e-6, value=0.5, step=0.0001,format="%.4f", key="r_b")
            e_b = st.number_input("e/B", min_value=1e-6, value=0.1, step=0.0001,format="%.4f", key="e_b")
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            logger.debug("Prediction form submitted")
            with st.spinner("Predicting..."):
                predicted_n = predict_stability_number(sigma_ci, gsi, mi, c_h, r_b, e_b, model, scaler, poly, feature_names)
                if predicted_n is not None:
                    st.success(f"**Predicted Stability Number Ratio (N):** {predicted_n:.6f}")

# Train New Model Mode
elif app_mode == "Train New Model":
    logger.debug("Entering Train New Model mode, app_state=%s, render_phase=%s, training_lock=%s", 
                 st.session_state.app_state, st.session_state.render_phase, st.session_state.training_lock)
    st.header("📊 Train a New Model")
    st.markdown("Upload a CSV to train a Random Forest model and visualize results.")
    
    # Display cached results if trained
    if st.session_state.app_state == "trained" and st.session_state.render_phase == "trained":
        st.info("Model trained. Displaying results below.")
        render_cached_results(st.session_state.cached_results)
    
    # File uploader for new training
    if st.session_state.app_state in ["idle", "training"] and st.session_state.render_phase == "idle" and not st.session_state.training_lock:
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], key=f"uploader_{st.session_state.file_uploader_key}")
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.app_state = "training"
            st.session_state.render_phase = "training"
            st.session_state.training_lock = True
            logger.debug("File uploaded, app_state=training, render_phase=training, training_lock=True")
    
    # Training process
    if st.session_state.app_state == "training" and st.session_state.uploaded_file:
        logger.debug("Starting training process")
        cached_results = {}
        try:
            data = pd.read_csv(st.session_state.uploaded_file)
            logger.debug("Dataset loaded, shape=%s", data.shape)
            
            # Dataset Description
            try:
                cached_results['dataset_shape'] = data.shape
                stats_df = data[['Sigma Ci', 'GSI', 'mi', 'C/H', 'r/B', 'e/B', 'N']].describe().T[['mean', 'std', 'min', 'max']].round(4)
                stats_df.columns = ['Mean', 'Std Dev', 'Min', 'Max']
                cached_results['stats_df'] = stats_df
                cached_results['sample_df'] = data[['Sigma Ci', 'GSI', 'mi', 'C/H', 'r/B', 'e/B', 'N']].head(5)
                
                corr_matrix = data[['Sigma Ci', 'GSI', 'mi', 'C/H', 'r/B', 'e/B', 'N']].corr()
                if corr_matrix.isnull().all().all():
                    logger.warning("Correlation matrix is all NaN")
                    cached_results['corr_fig'] = None
                else:
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='Viridis',
                        zmin=-1, zmax=1,
                        text=corr_matrix.values.round(2),
                        texttemplate="%{text}",
                        showscale=True))
                    fig_corr.update_layout(title="Correlation Matrix Heatmap", width=600, height=600)
                    cached_results['corr_fig'] = fig_corr
                
                st.subheader("📈 Dataset Description")
                st.write(f"**Dataset Shape**: {data.shape[0]} rows, {data.shape[1]} columns")
                st.write("**Feature Statistics**:")
                st.table(stats_df)
                st.write("**Sample of Dataset (First 5 Rows)**:")
                st.dataframe(cached_results['sample_df'])
                if cached_results['corr_fig']:
                    st.plotly_chart(cached_results['corr_fig'], use_container_width=True)
                logger.debug("Dataset Description rendered")
                
                if debug_mode:
                    st.sidebar.subheader("Dataset Debug Info")
                    debug_info = {
                        "Rows": data.shape[0],
                        "Columns": data.shape[1],
                        "N Unique Values": data['N'].nunique(),
                        "NaN Count": data.isnull().sum().sum(),
                        "Sigma Ci Range": f"[{data['Sigma Ci'].min():.2f}, {data['Sigma Ci'].max():.2f}]",
                        "GSI Range": f"[{data['GSI'].min():.2f}, {data['GSI'].max():.2f}]"
                    }
                    st.sidebar.json(debug_info)
            except Exception as e:
                logger.error("Dataset Description failed: %s", str(e))
                st.warning(f"Dataset Description error: {str(e)}")
            
            # Preprocessing
            X_scaled, y, scaler, poly, feature_names = preprocess_data(data)
            if X_scaled is None:
                logger.error("Preprocessing failed")
                st.session_state.app_state = "idle"
                st.session_state.render_phase = "idle"
                st.session_state.uploaded_file = None
                st.session_state.file_uploader_key += 1
                st.session_state.training_lock = False
                st.rerun()
            
            # Data split
            try:
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                logger.debug("Data split, X_train shape=%s", X_train.shape)
            except Exception as e:
                logger.error("Data split failed: %s", str(e))
                st.error(f"Data split error: {str(e)}")
                st.session_state.app_state = "idle"
                st.session_state.render_phase = "idle"
                st.session_state.uploaded_file = None
                st.session_state.file_uploader_key += 1
                st.session_state.training_lock = False
                st.rerun()
            
            # Optuna optimization
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 5, 50),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                }
                if params['bootstrap']:
                    params['max_samples'] = trial.suggest_float('max_samples', 0.5, 1.0)
                
                rf = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                cv_scores = cross_val_score(rf, X_train, y_train, cv=10, scoring='r2', n_jobs=-1)
                if np.isnan(cv_scores).any():
                    logger.warning("NaN in CV scores")
                    return -float('inf')
                return cv_scores.mean()
            
            logger.debug("Initializing progress bar")
            total_trials = 50
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            def progress_callback(study, trial):
                completed_trials = trial.number + 1
                progress = completed_trials / total_trials
                progress_bar.progress(min(progress, 1.0))
                progress_text.text(f"Optimizing: {completed_trials}/{total_trials} trials ({int(progress * 100)}%)")
            
            with st.spinner("Optimizing model with Optuna..."):
                try:
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=total_trials, callbacks=[progress_callback])
                    logger.debug("Optuna optimization completed, best_value=%s", study.best_value)
                except Exception as e:
                    logger.error("Optimization failed: %s", str(e))
                    st.error(f"Optimization failed: {str(e)}")
                    st.session_state.app_state = "idle"
                    st.session_state.render_phase = "idle"
                    st.session_state.uploaded_file = None
                    st.session_state.file_uploader_key += 1
                    st.session_state.training_lock = False
                    st.rerun()
            
            st.session_state.render_phase = "rendering"
            logger.debug("Transition to render_phase=rendering")
            
            # Hyperparameters
            try:
                best_params = study.best_params
                final_params = best_params.copy()
                if not final_params['bootstrap']:
                    final_params.pop('max_samples', None)
                
                if study.best_value <= -float('inf'):
                    logger.error("Optimization produced invalid results")
                    st.warning("Optimization failed to produce valid hyperparameters.")
                    cached_results['params_df'] = None
                    cached_results['best_cv_r2'] = None
                else:
                    cached_results['params_df'] = pd.DataFrame(list(final_params.items()), columns=['Parameter', 'Value'])
                    cached_results['best_cv_r2'] = study.best_value
                    st.subheader("Best Hyperparameters")
                    st.table(cached_results['params_df'])
                    st.write(f"Best Cross-Validation R² Score: {study.best_value:.4f}")
                    logger.debug("Hyperparameters displayed")
            except Exception as e:
                logger.error("Hyperparameter display failed: %s", str(e))
                st.warning(f"Hyperparameter display error: {str(e)}")
                cached_results['params_df'] = None
                cached_results['best_cv_r2'] = None
            
            # Train final model
            try:
                best_rf = RandomForestRegressor(**final_params, random_state=42, n_jobs=-1)
                best_rf.fit(X_train, y_train)
                logger.debug("Final model trained")
            except Exception as e:
                logger.error("Model training failed: %s", str(e))
                st.warning(f"Model training error: {str(e)}")
                best_rf = None
            
            # Test predictions
            try:
                y_pred = best_rf.predict(X_test) if best_rf is not None else None
                logger.debug("Test predictions completed, y_pred shape=%s", y_pred.shape if y_pred is not None else "None")
            except Exception as e:
                logger.error("Test predictions failed: %s", str(e))
                st.warning(f"Prediction error: {str(e)}")
                y_pred = None
            
            # Metrics
            try:
                if y_pred is not None and not np.isnan(y_pred).any():
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    
                    if np.isnan([r2, mae, mse, rmse]).any():
                        logger.error("Invalid metrics: r2=%s, mae=%s, mse=%s, rmse=%s", r2, mae, mse, rmse)
                        st.warning("Invalid performance metrics.")
                        cached_results['metrics'] = None
                    else:
                        cached_results['metrics'] = {'r2': r2, 'mae': mae, 'mse': mse, 'rmse': rmse}
                        st.subheader("Model Performance Metrics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("R² Score", f"{r2:.4f}")
                            st.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
                        with col2:
                            st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
                            st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
                        logger.debug("Metrics displayed")
                else:
                    logger.warning("Skipping metrics due to invalid predictions")
                    st.warning("Cannot compute metrics due to invalid predictions.")
                    cached_results['metrics'] = None
            except Exception as e:
                logger.error("Metrics calculation failed: %s", str(e))
                st.warning(f"Metrics calculation error: {str(e)}")
                cached_results['metrics'] = None
            
            # Feature Importance
            try:
                if best_rf is not None:
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': best_rf.feature_importances_
                    }).sort_values(by='Importance', ascending=False)
                    
                    if feature_importance.empty or feature_importance['Importance'].isnull().any():
                        logger.error("Invalid feature importance data")
                        st.warning("Feature importance data is invalid.")
                        cached_results['fig_importance'] = None
                    else:
                        fig_importance = px.bar(feature_importance.head(10), x='Importance', y='Feature', orientation='h',
                                                title="Top 10 Feature Importance",
                                                color='Importance', color_continuous_scale='Viridis')
                        cached_results['fig_importance'] = fig_importance
                        st.plotly_chart(fig_importance, use_container_width=True)
                        logger.debug("Feature Importance plot rendered")
                else:
                    logger.warning("Skipping Feature Importance due to no model")
                    st.warning("Feature Importance plot skipped: No model available.")
                    cached_results['fig_importance'] = None
            except Exception as e:
                logger.error("Feature Importance plot failed: %s", str(e))
                st.warning(f"Feature Importance plot error: {str(e)}")
                cached_results['fig_importance'] = None
            
            # Actual vs Predicted
            try:
                if y_pred is not None and not np.isnan(y_pred).any():
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions',
                                                 marker=dict(color='blue', size=8)))
                    fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                                 mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
                    fig_pred.update_layout(title="Actual vs Predicted Stability Number",
                                          xaxis_title="Actual N", yaxis_title="Predicted N",
                                          showlegend=True)
                    cached_results['fig_pred'] = fig_pred
                    st.plotly_chart(fig_pred, use_container_width=True)
                    logger.debug("Actual vs Predicted plot rendered")
                else:
                    logger.warning("Skipping Actual vs Predicted due to invalid predictions")
                    st.warning("Actual vs Predicted plot skipped: Invalid predictions.")
                    cached_results['fig_pred'] = None
            except Exception as e:
                logger.error("Actual vs Predicted plot failed: %s", str(e))
                st.warning(f"Actual vs Predicted plot error: {str(e)}")
                cached_results['fig_pred'] = None
            
            # Residual Plot
            try:
                if y_pred is not None and not np.isnan(y_pred).any():
                    residuals = y_test - y_pred
                    fig_residual = go.Figure()
                    fig_residual.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                                                     name='Residuals', marker=dict(color='purple', size=8)))
                    fig_residual.add_trace(go.Scatter(x=[y_pred.min(), y_pred.max()], y=[0, 0],
                                                     mode='lines', name='Zero Line', line=dict(color='black')))
                    fig_residual.update_layout(title="Residual Plot",
                                              xaxis_title="Predicted N", yaxis_title="Residuals",
                                              showlegend=True)
                    cached_results['fig_residual'] = fig_residual
                    st.plotly_chart(fig_residual, use_container_width=True)
                    logger.debug("Residual plot rendered")
                else:
                    logger.warning("Skipping Residual Plot due to invalid predictions")
                    st.warning("Residual Plot skipped: Invalid predictions.")
                    cached_results['fig_residual'] = None
            except Exception as e:
                logger.error("Residual plot failed: %s", str(e))
                st.warning(f"Residual plot error: {str(e)}")
                cached_results['fig_residual'] = None
            
            # Prediction Error Distribution
            try:
                if y_pred is not None and not np.isnan(y_pred).any():
                    residuals = y_test - y_pred
                    fig_error_dist = px.histogram(residuals, nbins=30, title="Prediction Error Distribution",
                                                 color_discrete_sequence=['teal'])
                    fig_error_dist.update_layout(xaxis_title="Prediction Error", yaxis_title="Count",
                                                showlegend=False)
                    cached_results['fig_error_dist'] = fig_error_dist
                    st.plotly_chart(fig_error_dist, use_container_width=True)
                    logger.debug("Prediction Error Distribution plot rendered")
                else:
                    logger.warning("Skipping Prediction Error Distribution due to invalid predictions")
                    st.warning("Prediction Error Distribution skipped: Invalid predictions.")
                    cached_results['fig_error_dist'] = None
            except Exception as e:
                logger.error("Prediction Error Distribution plot failed: %s", str(e))
                st.warning(f"Prediction Error Distribution plot error: {str(e)}")
                cached_results['fig_error_dist'] = None
            
            # Learning Curves
            try:
                if best_rf is not None and len(X_train) >= 20:
                    train_sizes, train_scores, test_scores = learning_curve(
                        best_rf, X_train, y_train, cv=3, scoring='r2', n_jobs=1,
                        train_sizes=np.linspace(0.1, 1.0, 5))
                    
                    train_mean = np.mean(train_scores, axis=1)
                    train_std = np.std(train_scores, axis=1)
                    test_mean = np.mean(test_scores, axis=1)
                    test_std = np.std(test_scores, axis=1)
                    
                    if np.any(np.isnan([train_mean, train_std, test_mean, test_std])):
                        logger.error("Invalid learning curve data")
                        st.warning("Learning curve data is invalid.")
                        cached_results['fig_learning'] = None
                    else:
                        fig_learning = go.Figure()
                        fig_learning.add_trace(go.Scatter(x=train_sizes, y=train_mean, name='Training Score',
                                                         mode='lines', line=dict(color='blue')))
                        fig_learning.add_trace(go.Scatter(x=train_sizes, y=test_mean, name='Validation Score',
                                                         mode='lines', line=dict(color='orange')))
                        fig_learning.add_trace(go.Scatter(
                            x=np.concatenate([train_sizes, train_sizes[::-1]]),
                            y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
                            fill='toself', fillcolor='rgba(0, 0, 255, 0.1)', line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip", showlegend=False))
                        fig_learning.add_trace(go.Scatter(
                            x=np.concatenate([train_sizes, train_sizes[::-1]]),
                            y=np.concatenate([test_mean + test_std, (test_mean - test_std)[::-1]]),
                            fill='toself', fillcolor='rgba(255, 165, 0, 0.1)', line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip", showlegend=False))
                        fig_learning.update_layout(title="Learning Curves",
                                                  xaxis_title="Training Examples", yaxis_title="R² Score",
                                                  showlegend=True)
                        cached_results['fig_learning'] = fig_learning
                        st.plotly_chart(fig_learning, use_container_width=True)
                        logger.debug("Learning Curves plot rendered")
                else:
                    logger.warning("Skipping Learning Curves: insufficient data or no model")
                    st.warning("Learning Curves skipped: Insufficient data or no model.")
                    cached_results['fig_learning'] = None
            except Exception as e:
                logger.error("Learning Curves plot failed: %s", str(e))
                st.warning(f"Learning Curves plot error: {str(e)}")
                cached_results['fig_learning'] = None
            
            # Save model
            try:
                if best_rf is not None and scaler is not None and poly is not None and feature_names is not None:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    model_filename = f"random_forest_model_{timestamp}.pkl"
                    joblib.dump(best_rf, model_filename)
                    joblib.dump(scaler, f'scaler_{timestamp}.pkl')
                    joblib.dump(poly, f'poly_transformer_{timestamp}.pkl')
                    cached_results['model_saved'] = True
                    cached_results['model_filename'] = f"'random_forest_model_{timestamp}.pkl', 'scaler_{timestamp}.pkl', 'poly_transformer_{timestamp}.pkl'"
                    st.success(f"Model saved as {cached_results['model_filename']}.")
                    logger.debug("Model saved")
                else:
                    logger.warning("Skipping model saving: incomplete components")
                    st.warning("Model saving skipped: Incomplete components.")
                    cached_results['model_saved'] = False
                    cached_results['model_filename'] = None
            except Exception as e:
                logger.error("Model saving failed: %s", str(e))
                st.warning(f"Model saving error: {str(e)}")
                cached_results['model_saved'] = False
                cached_results['model_filename'] = None
            
            # Update session state
            try:
                if best_rf is not None and scaler is not None and poly is not None and feature_names is not None:
                    st.session_state.best_rf = best_rf
                    st.session_state.scaler = scaler
                    st.session_state.poly = poly
                    st.session_state.feature_names = feature_names
                    st.session_state.cached_results = cached_results
                    st.session_state.app_state = "trained"
                    st.session_state.render_phase = "trained"
                    st.session_state.uploaded_file = None
                    st.session_state.file_uploader_key += 1
                    st.session_state.training_lock = False
                    logger.debug("Session state updated, app_state=trained, render_phase=trained, training_lock=False")
                else:
                    logger.warning("Skipping session state update: incomplete components")
                    st.warning("Session state not updated: Incomplete components.")
                    st.session_state.app_state = "idle"
                    st.session_state.render_phase = "idle"
                    st.session_state.uploaded_file = None
                    st.session_state.file_uploader_key += 1
                    st.session_state.training_lock = False
                    st.rerun()
            except Exception as e:
                logger.error("Session state update failed: %s", str(e))
                st.warning(f"Session state update error: {str(e)}")
        
        except Exception as e:
            logger.error("Training process failed: %s", str(e))
            st.error(f"Critical training error: {str(e)}")
            st.session_state.app_state = "idle"
            st.session_state.render_phase = "idle"
            st.session_state.uploaded_file = None
            st.session_state.best_rf = None
            st.session_state.scaler = None
            st.session_state.poly = None
            st.session_state.feature_names = None
            st.session_state.file_uploader_key += 1
            st.session_state.training_lock = False
            st.rerun()
    
   # Train New Model Mode - Prediction Form
if st.session_state.app_state == "trained" and st.session_state.render_phase == "trained":
    logger.debug("Rendering prediction form")
    if st.button("Reset to Train New Model"):
        st.session_state.app_state = "idle"
        st.session_state.render_phase = "idle"
        st.session_state.uploaded_file = None
        st.session_state.best_rf = None
        st.session_state.scaler = None
        st.session_state.poly = None
        st.session_state.feature_names = None
        st.session_state.cached_results = None
        st.session_state.file_uploader_key += 1
        st.session_state.training_lock = False
        logger.debug("Reset button clicked, app_state=idle, render_phase=idle")
        st.rerun()
    
    st.subheader("🔍 Predict with New Model")
    st.markdown("Enter positive feature values to predict with the new model.")
    
    with st.form("new_model_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_sigma_ci = st.number_input("Sigma Ci (MPa)", min_value=1e-6, value=50.0, step=0.0001,format="%.4f", key="new_sigma_ci")
            new_gsi = st.number_input("GSI", min_value=1e-6, value=50.0, step=0.0001, format="%.4f", key="new_gsi")
            new_mi = st.number_input("mi", min_value=1e-6, value=10.0, step=0.0001,format="%.4f", key="new_mi")
        with col2:
            new_c_h = st.number_input("C/H", min_value=1e-6, value=1.0, step=0.0001,format="%.4f", key="new_c_h")
            new_r_b = st.number_input("r/B", min_value=1e-6, value=0.5, step=0.0001,format="%.4f", key="new_r_b")
            new_e_b = st.number_input("e/B", min_value=1e-6, value=0.1, step=0.0001,format="%.4f", key="new_e_b")
        
        new_submitted = st.form_submit_button("Predict with New Model")
        
        if new_submitted:
            logger.debug("Predict with New Model button clicked")
            with st.spinner("Predicting with new model..."):
                model_valid = st.session_state.best_rf is not None
                scaler_valid = st.session_state.scaler is not None
                poly_valid = st.session_state.poly is not None
                features_valid = st.session_state.feature_names is not None and len(st.session_state.feature_names) > 0
                
                if not (model_valid and scaler_valid and poly_valid and features_valid):
                    logger.error("Invalid session state: model=%s, scaler=%s, poly=%s, features=%s",
                                 model_valid, scaler_valid, poly_valid, features_valid)
                    st.error("No trained model available. Please train a model first.")
                    st.session_state.app_state = "idle"
                    st.session_state.render_phase = "idle"
                    st.session_state.uploaded_file = None
                    st.session_state.best_rf = None
                    st.session_state.scaler = None
                    st.session_state.poly = None
                    st.session_state.feature_names = None
                    st.session_state.cached_results = None
                    st.session_state.file_uploader_key += 1
                    st.session_state.training_lock = False
                    st.rerun()
                else:
                    predicted_n = predict_stability_number(
                        new_sigma_ci, new_gsi, new_mi, new_c_h, new_r_b, new_e_b,
                        st.session_state.best_rf, st.session_state.scaler,
                        st.session_state.poly, st.session_state.feature_names
                    )
                    if predicted_n is not None:
                        st.success(f"**Predicted Stability Number Ratio (N):** {predicted_n:.6f}")
                        logger.debug("Prediction displayed successfully")

# Footer
st.markdown("---")
st.markdown("**Designed and Developed by Rishabh Kashyap**")
