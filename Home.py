# forest_change_accuracy_assessment.py
import ee
import geemap
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import json
from shapely.geometry import Point
import random
from datetime import datetime
import tempfile
import os
import zipfile

# Initialize Earth Engine
service_account = st.secrets.get("EE_SERVICE_ACCOUNT") if hasattr(st, 'secrets') else None
private_key = st.secrets.get("EE_PRIVATE_KEY") if hasattr(st, 'secrets') else None

def initialize_ee():
    """Initialize Earth Engine with service account or user credentials"""
    try:
        if service_account and private_key:
            # For deployment with service account
            credentials = ee.ServiceAccountCredentials(service_account, key_data=private_key)
            ee.Initialize(credentials)
        else:
            # For local development
            ee.Initialize(project='ee-your-project-id')
        return True
    except Exception as e:
        st.error(f"Earth Engine initialization failed: {e}")
        return False

# Set page config
st.set_page_config(page_title="Forest Change Accuracy Assessment", layout="wide")

# Title and description
st.title("Forest Change Time-Series Accuracy Assessment")
st.markdown("""
This dashboard follows the Olofsson (2014) methodology for accuracy assessment of deforestation, degradation, 
and regeneration detection using NICFI high-resolution data.
""")

# Sidebar for configuration
st.sidebar.header("Assessment Configuration")

# Initialize Earth Engine
if not ee.data._initialized:
    if initialize_ee():
        st.sidebar.success("Earth Engine initialized!")
    else:
        st.sidebar.warning("Earth Engine not initialized. Please check credentials.")

# 1. Upload TS classification results
st.sidebar.subheader("1. Time-Series Classification Results")
uploaded_ts = st.sidebar.file_uploader("Upload TS classification file", 
                                       type=['gpkg', 'shp', 'shx', 'dbf', 'prj', 'csv', 'geojson'])

# Handle shapefile components
uploaded_files = {}
if uploaded_ts:
    if uploaded_ts.name.endswith('.shp'):
        uploaded_files['shp'] = uploaded_ts
    elif uploaded_ts.name.endswith('.shx'):
        uploaded_files['shx'] = uploaded_ts
    elif uploaded_ts.name.endswith('.dbf'):
        uploaded_files['dbf'] = uploaded_ts
    elif uploaded_ts.name.endswith('.prj'):
        uploaded_files['prj'] = uploaded_ts

# 2. Define classes
st.sidebar.subheader("2. Define Classes")
default_classes = "Stable Forest\nDeforestation\nDegradation\nRegeneration\nStable Non-Forest"
classes = st.sidebar.text_area("Enter class names (one per line):", default_classes)
class_list = [c.strip() for c in classes.split('\n') if c.strip()]

# 3. Sampling parameters
st.sidebar.subheader("3. Sampling Parameters")
sample_method = st.sidebar.selectbox("Sampling method:", ["Stratified Random", "Simple Random", "Systematic"])
sample_size = st.sidebar.number_input("Total sample size:", min_value=50, max_value=2000, value=250)
min_samples_per_class = st.sidebar.number_input("Minimum samples per class:", min_value=5, value=30)
buffer_distance = st.sidebar.number_input("Buffer distance (meters):", min_value=10, max_value=1000, value=100)

# 4. NICFI year selection
st.sidebar.subheader("4. NICFI Imagery Year")
nicfi_years = list(range(2016, 2025))
nicfi_year = st.sidebar.selectbox("Select NICFI year:", nicfi_years, index=nicfi_years.index(2020))

# 5. Additional options
st.sidebar.subheader("5. Additional Options")
show_ts_layer = st.sidebar.checkbox("Show TS classification layer", value=True)
ts_opacity = st.sidebar.slider("TS layer opacity:", 0.0, 1.0, 0.5) if show_ts_layer else 0.0
show_ndvi = st.sidebar.checkbox("Show NDVI layer", value=False)

# 6. Initialize assessment button
st.sidebar.subheader("6. Initialize Assessment")
init_button = st.sidebar.button("Initialize Assessment Points", type="primary")

# Main content area
col1, col2 = st.columns([2, 1])

# Initialize session state for points
if 'assessment_points' not in st.session_state:
    st.session_state.assessment_points = None
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'user_assessments' not in st.session_state:
    st.session_state.user_assessments = {}
if 'ts_gdf' not in st.session_state:
    st.session_state.ts_gdf = None

# Function to load TS classification
def load_ts_classification(file, uploaded_files=None):
    """Load time-series classification results"""
    try:
        # Create temporary directory for shapefile components
        with tempfile.TemporaryDirectory() as tmp_dir:
            if file.name.endswith('.zip'):
                # Extract zip file
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)
                
                # Find shapefile in extracted files
                shapefile = None
                for f in os.listdir(tmp_dir):
                    if f.endswith('.shp'):
                        shapefile = os.path.join(tmp_dir, f)
                        break
                
                if shapefile:
                    gdf = gpd.read_file(shapefile)
                else:
                    st.error("No shapefile found in zip archive")
                    return None
                    
            elif file.name.endswith('.shp'):
                # Save all shapefile components
                shp_path = os.path.join(tmp_dir, file.name)
                with open(shp_path, 'wb') as f:
                    f.write(file.getvalue())
                
                # Check for other components
                for ext in ['.shx', '.dbf', '.prj']:
                    comp_name = file.name.replace('.shp', ext)
                    if comp_name in uploaded_files:
                        comp_path = os.path.join(tmp_dir, comp_name)
                        with open(comp_path, 'wb') as f:
                            f.write(uploaded_files[ext].getvalue())
                
                gdf = gpd.read_file(shp_path)
                
            elif file.name.endswith('.gpkg'):
                gpkg_path = os.path.join(tmp_dir, file.name)
                with open(gpkg_path, 'wb') as f:
                    f.write(file.getvalue())
                gdf = gpd.read_file(gpkg_path)
                
            elif file.name.endswith('.geojson') or file.name.endswith('.json'):
                gdf = gpd.read_file(file)
                
            elif file.name.endswith('.csv'):
                df = pd.read_csv(file)
                # Assuming coordinates are in columns 'lon' and 'lat'
                if 'lon' in df.columns and 'lat' in df.columns:
                    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
                    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
                else:
                    st.error("CSV must contain 'lon' and 'lat' columns")
                    return None
            else:
                st.error("Unsupported file format")
                return None
            
            # Check required columns
            if 'class' not in gdf.columns:
                st.error("Data must contain a 'class' column")
                return None
            
            return gdf
            
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Function to generate stratified random points
def generate_stratified_points(gdf, sample_size, min_per_class, method="Stratified Random"):
    """Generate random points based on selected method"""
    points_list = []
    
    if method == "Stratified Random":
        # Get area for each class (if geometry is polygon)
        if gdf.geometry.type.str.contains('Polygon').any():
            gdf['area'] = gdf.geometry.area
            class_areas = gdf.groupby('class')['area'].sum()
        else:
            class_areas = gdf.groupby('class').size()
        
        total_area = class_areas.sum()
        
        for class_name in class_list:
            if class_name in class_areas.index:
                class_area = class_areas[class_name]
                proportion = class_area / total_area
                
                # Proportional allocation with minimum
                n_points = max(min_per_class, int(sample_size * proportion))
                
                # Get features of this class
                class_features = gdf[gdf['class'] == class_name]
                
                # Generate points proportional to feature area/size
                for _, feature in class_features.iterrows():
                    if feature.geometry.type == 'Polygon':
                        feature_area = feature.geometry.area
                        feature_proportion = feature_area / class_area
                        n_feature_points = max(1, int(n_points * feature_proportion))
                        
                        for _ in range(n_feature_points):
                            # Generate random point within polygon
                            minx, miny, maxx, maxy = feature.geometry.bounds
                            attempts = 0
                            while attempts < 100:  # Safety limit
                                random_point = Point(
                                    random.uniform(minx, maxx),
                                    random.uniform(miny, maxy)
                                )
                                if feature.geometry.contains(random_point):
                                    points_list.append({
                                        'geometry': random_point,
                                        'class_ts': class_name,
                                        'class_user': None,
                                        'confidence': None,
                                        'notes': None,
                                        'difficulty': None,
                                        'timestamp': None
                                    })
                                    break
                                attempts += 1
                    
                    elif feature.geometry.type == 'Point':
                        # For point data, just use the point
                        points_list.append({
                            'geometry': feature.geometry,
                            'class_ts': class_name,
                            'class_user': None,
                            'confidence': None,
                            'notes': None,
                            'difficulty': None,
                            'timestamp': None
                        })
    
    elif method == "Simple Random":
        # Simple random sampling
        n_points = min(sample_size, len(gdf))
        random_indices = random.sample(range(len(gdf)), n_points)
        
        for idx in random_indices:
            feature = gdf.iloc[idx]
            if feature.geometry.type == 'Polygon':
                # Generate point within polygon
                minx, miny, maxx, maxy = feature.geometry.bounds
                attempts = 0
                while attempts < 100:
                    random_point = Point(
                        random.uniform(minx, maxx),
                        random.uniform(miny, maxy)
                    )
                    if feature.geometry.contains(random_point):
                        points_list.append({
                            'geometry': random_point,
                            'class_ts': feature['class'],
                            'class_user': None,
                            'confidence': None,
                            'notes': None,
                            'difficulty': None,
                            'timestamp': None
                        })
                        break
                    attempts += 1
            else:
                points_list.append({
                    'geometry': feature.geometry,
                    'class_ts': feature['class'],
                    'class_user': None,
                    'confidence': None,
                    'notes': None,
                    'difficulty': None,
                    'timestamp': None
                })
    
    elif method == "Systematic":
        # Systematic sampling (grid-based)
        bounds = gdf.total_bounds
        xmin, ymin, xmax, ymax = bounds
        
        # Calculate grid spacing based on sample size
        area = (xmax - xmin) * (ymax - ymin)
        spacing = np.sqrt(area / sample_size)
        
        # Generate grid points
        x_coords = np.arange(xmin, xmax, spacing)
        y_coords = np.arange(ymin, ymax, spacing)
        
        grid_points = []
        for x in x_coords:
            for y in y_coords:
                point = Point(x, y)
                # Check if point is within any polygon
                for _, row in gdf.iterrows():
                    if row.geometry.contains(point):
                        grid_points.append({
                            'geometry': point,
                            'class_ts': row['class'],
                            'class_user': None,
                            'confidence': None,
                            'notes': None,
                            'difficulty': None,
                            'timestamp': None
                        })
                        break
        
        # Take required number of points
        points_list = grid_points[:sample_size]
    
    return gpd.GeoDataFrame(points_list, crs=gdf.crs)

# Function to get NICFI imagery
def get_nicfi_image(year, point, buffer=100):
    """Get NICFI basemap for given year and location"""
    try:
        # NICFI Planet Basemaps in Earth Engine
        nicfi = ee.ImageCollection('projects/planet-nicfi/assets/basemaps/africa')
        
        # Filter by date for the selected year
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        
        nicfi_filtered = nicfi.filterDate(start_date, end_date).mosaic()
        
        # Convert point to EE geometry
        point_geom = ee.Geometry.Point([point.x, point.y])
        buffer_geom = point_geom.buffer(buffer)
        
        # RGB visualization
        vis_params = {
            'bands': ['R', 'G', 'B'],
            'min': 64,
            'max': 5454,
            'gamma': 1.8
        }
        
        # NDVI if requested
        ndvi = None
        if show_ndvi:
            ndvi = nicfi_filtered.normalizedDifference(['N', 'R']).rename('NDVI')
            ndvi_vis = {
                'min': -0.2,
                'max': 0.8,
                'palette': ['blue', 'white', 'green']
            }
        
        return nicfi_filtered.clip(buffer_geom), vis_params, ndvi
        
    except Exception as e:
        st.error(f"Error getting NICFI imagery: {e}")
        return None, None, None

# Function to create assessment map
def create_assessment_map(point, ts_gdf):
    """Create interactive map for assessment"""
    try:
        # Create map centered on point
        Map = geemap.Map(center=[point.geometry.y, point.geometry.x], 
                        zoom=16,
                        height=600)
        
        # Get NICFI imagery
        nicfi_image, vis_params, ndvi = get_nicfi_image(nicfi_year, point.geometry, buffer_distance)
        
        if nicfi_image:
            Map.addLayer(nicfi_image, vis_params, f'NICFI {nicfi_year}')
            
            # Add NDVI if requested
            if ndvi and show_ndvi:
                ndvi_vis = {
                    'min': -0.2,
                    'max': 0.8,
                    'palette': ['blue', 'white', 'green']
                }
                Map.addLayer(ndvi, ndvi_vis, 'NDVI', False)
        
        # Add TS classification if requested
        if show_ts_layer and ts_gdf is not None:
            # Buffer around point
            buffer_geom = point.geometry.buffer(buffer_distance)
            
            # Clip TS classification to buffer
            ts_clipped = ts_gdf[ts_gdf.geometry.intersects(buffer_geom)].copy()
            
            if not ts_clipped.empty:
                # Define colors for classes
                class_colors = {
                    'Stable Forest': '#228B22',
                    'Deforestation': '#FF0000',
                    'Degradation': '#FFA500',
                    'Regeneration': '#00BFFF',
                    'Stable Non-Forest': '#A9A9A9'
                }
                
                # Add each class as a separate layer
                for class_name, color in class_colors.items():
                    class_features = ts_clipped[ts_clipped['class'] == class_name]
                    if not class_features.empty:
                        # Convert to GeoJSON
                        geojson = class_features.__geo_interface__
                        
                        # Create EE feature collection
                        fc = ee.FeatureCollection(geojson)
                        
                        # Add to map
                        Map.addLayer(
                            fc.style(**{'fillColor': color, 'color': color, 'fillOpacity': ts_opacity}),
                            {},
                            f"TS: {class_name}",
                            True
                        )
        
        # Add current point marker
        Map.add_marker(location=[point.geometry.y, point.geometry.x],
                      popup=f"Assessment Point\nTS Class: {point['class_ts']}",
                      draggable=False)
        
        return Map
        
    except Exception as e:
        st.error(f"Error creating map: {e}")
        return None

# Main application logic
if uploaded_ts:
    # Load TS classification
    if st.session_state.ts_gdf is None or st.sidebar.button("Reload Data"):
        with st.spinner("Loading classification data..."):
            st.session_state.ts_gdf = load_ts_classification(uploaded_ts, uploaded_files)
        
        if st.session_state.ts_gdf is not None:
            st.sidebar.success(f"Loaded {len(st.session_state.ts_gdf)} features")
            
            # Show data preview
            with st.expander("Data Preview"):
                st.dataframe(st.session_state.ts_gdf.head())
                st.write(f"Classes found: {st.session_state.ts_gdf['class'].unique().tolist()}")
                st.write(f"CRS: {st.session_state.ts_gdf.crs}")

with col1:
    st.subheader("Interactive Assessment Map")
    
    if st.session_state.ts_gdf is not None:
        if init_button:
            with st.spinner("Generating assessment points..."):
                assessment_points = generate_stratified_points(
                    st.session_state.ts_gdf, 
                    sample_size, 
                    min_samples_per_class,
                    sample_method
                )
                st.session_state.assessment_points = assessment_points
                st.session_state.current_index = 0
                st.session_state.user_assessments = {}
                
                st.success(f"Generated {len(assessment_points)} assessment points using {sample_method} sampling")
        
        if st.session_state.assessment_points is not None:
            # Display progress
            total_points = len(st.session_state.assessment_points)
            assessed_points = len([v for v in st.session_state.user_assessments.values() 
                                  if v.get('class_user')])
            progress = assessed_points / total_points if total_points > 0 else 0
            
            # Progress bar and stats
            col_prog1, col_prog2, col_prog3 = st.columns(3)
            with col_prog1:
                st.metric("Total Points", total_points)
            with col_prog2:
                st.metric("Assessed", assessed_points)
            with col_prog3:
                st.metric("Progress", f"{progress:.1%}")
            
            st.progress(progress)
            
            # Get current point
            current_idx = st.session_state.current_index
            current_point = st.session_state.assessment_points.iloc[current_idx]
            
            # Create and display map
            assessment_map = create_assessment_map(current_point, st.session_state.ts_gdf)
            if assessment_map:
                # Use geemap's streamlit support
                assessment_map.to_streamlit()
            
            # Quick navigation
            st.write("### Quick Navigation")
            nav_cols = st.columns(5)
            
            with nav_cols[0]:
                if st.button("⏮️ First", use_container_width=True) and current_idx > 0:
                    st.session_state.current_index = 0
                    st.experimental_rerun()
            
            with nav_cols[1]:
                if st.button("⬅️ Previous", use_container_width=True) and current_idx > 0:
                    st.session_state.current_index -= 1
                    st.experimental_rerun()
            
            with nav_cols[2]:
                jump_to = st.number_input("Go to:", 
                                        min_value=0, 
                                        max_value=total_points-1, 
                                        value=current_idx,
                                        label_visibility="collapsed")
                if st.button("Go", use_container_width=True) and jump_to != current_idx:
                    st.session_state.current_index = int(jump_to)
                    st.experimental_rerun()
            
            with nav_cols[3]:
                if st.button("Next ➡️", use_container_width=True) and current_idx < total_points - 1:
                    st.session_state.current_index += 1
                    st.experimental_rerun()
            
            with nav_cols[4]:
                if st.button("Last ⏭️", use_container_width=True) and current_idx < total_points - 1:
                    st.session_state.current_index = total_points - 1
                    st.experimental_rerun()

with col2:
    st.subheader("Assessment Interface")
    
    if st.session_state.assessment_points is not None:
        current_idx = st.session_state.current_index
        current_point = st.session_state.assessment_points.iloc[current_idx]
        
        # Display point info
        st.info(f"**Point {current_idx + 1} of {total_points}**")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("TS Class", current_point['class_ts'])
        with col_info2:
            st.metric("Coordinates", f"{current_point.geometry.y:.4f}, {current_point.geometry.x:.4f}")
        
        # Previous assessment if exists
        if current_idx in st.session_state.user_assessments:
            prev_assessment = st.session_state.user_assessments[current_idx]
            st.warning(f"**Previously assessed as:** {prev_assessment.get('class_user', 'Not assessed')}")
        
        st.markdown("---")
        st.write("### Your Assessment")
        
        # User assessment form
        user_class = st.selectbox(
            "Reference Class:",
            options=["Select class..."] + class_list,
            index=0,
            key=f"class_select_{current_idx}"
        )
        
        confidence = st.radio(
            "Confidence:",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: f"{x} - {'Low' if x <= 2 else 'Medium' if x == 3 else 'High'}",
            horizontal=True,
            key=f"confidence_{current_idx}"
        )
        
        difficulty = st.select_slider(
            "Difficulty:",
            options=["Easy", "Moderate", "Difficult", "Uncertain"],
            value="Moderate",
            key=f"difficulty_{current_idx}"
        )
        
        notes = st.text_area(
            "Notes/Observations:",
            placeholder="Describe what you see, any issues with imagery, etc...",
            height=100,
            key=f"notes_{current_idx}"
        )
        
        # Action buttons
        col_act1, col_act2, col_act3 = st.columns(3)
        
        with col_act1:
            if st.button("💾 Save", use_container_width=True, type="primary"):
                if user_class != "Select class...":
                    st.session_state.user_assessments[current_idx] = {
                        'class_user': user_class,
                        'confidence': confidence,
                        'difficulty': difficulty,
                        'notes': notes,
                        'timestamp': datetime.now().isoformat()
                    }
                    st.success("Assessment saved!")
                    
                    # Auto-advance if not last point
                    if current_idx < total_points - 1 and st.checkbox("Auto-advance after save", value=True):
                        st.session_state.current_index += 1
                        st.experimental_rerun()
                else:
                    st.error("Please select a reference class")
        
        with col_act2:
            if st.button("⏭️ Save & Next", use_container_width=True):
                if user_class != "Select class...":
                    st.session_state.user_assessments[current_idx] = {
                        'class_user': user_class,
                        'confidence': confidence,
                        'difficulty': difficulty,
                        'notes': notes,
                        'timestamp': datetime.now().isoformat()
                    }
                    if current_idx < total_points - 1:
                        st.session_state.current_index += 1
                        st.experimental_rerun()
                else:
                    st.error("Please select a reference class")
        
        with col_act3:
            if st.button("🚫 Skip", use_container_width=True):
                if current_idx < total_points - 1:
                    st.session_state.current_index += 1
                    st.experimental_rerun()
        
        st.markdown("---")
        
        # Export results section
        st.write("### Export Results")
        
        if st.button("📊 Generate Accuracy Report"):
            # Compile results
            results = []
            for idx, point in st.session_state.assessment_points.iterrows():
                assessment = st.session_state.user_assessments.get(idx, {})
                results.append({
                    'point_id': idx,
                    'x_coord': point.geometry.x,
                    'y_coord': point.geometry.y,
                    'class_ts': point['class_ts'],
                    'class_user': assessment.get('class_user'),
                    'confidence': assessment.get('confidence'),
                    'difficulty': assessment.get('difficulty'),
                    'notes': assessment.get('notes'),
                    'timestamp': assessment.get('timestamp')
                })
            
            results_df = pd.DataFrame(results)
            
            # Calculate accuracy metrics
            assessed_df = results_df[results_df['class_user'].notna()]
            
            if not assessed_df.empty:
                # Confusion matrix
                confusion_data = pd.crosstab(
                    assessed_df['class_ts'],
                    assessed_df['class_user'],
                    rownames=['Time-Series Classification'],
                    colnames=['User Reference'],
                    margins=True,
                    margins_name="Total"
                )
                
                # Calculate accuracy metrics
                overall_accuracy = np.diag(confusion_data.iloc[:-1, :-1]).sum() / confusion_data.iloc[:-1, :-1].sum().sum()
                
                # User's accuracy (commission error)
                users_accuracy = pd.Series(index=class_list, dtype=float)
                for class_name in class_list:
                    if class_name in confusion_data.columns and class_name in confusion_data.index:
                        correct = confusion_data.loc[class_name, class_name]
                        total_user = confusion_data.loc['Total', class_name] if 'Total' in confusion_data.index else confusion_data[class_name].sum()
                        if total_user > 0:
                            users_accuracy[class_name] = correct / total_user
                
                # Producer's accuracy (omission error)
                producers_accuracy = pd.Series(index=class_list, dtype=float)
                for class_name in class_list:
                    if class_name in confusion_data.index and class_name in confusion_data.columns:
                        correct = confusion_data.loc[class_name, class_name]
                        total_producer = confusion_data.loc[class_name, 'Total'] if 'Total' in confusion_data.columns else confusion_data.loc[class_name].sum()
                        if total_producer > 0:
                            producers_accuracy[class_name] = correct / total_producer
                
                # Display metrics
                st.success(f"**Overall Accuracy:** {overall_accuracy:.2%}")
                
                col_met1, col_met2 = st.columns(2)
                with col_met1:
                    st.write("**User's Accuracy:**")
                    st.dataframe(users_accuracy.round(3))
                
                with col_met2:
                    st.write("**Producer's Accuracy:**")
                    st.dataframe(producers_accuracy.round(3))
                
                st.write("**Confusion Matrix:**")
                st.dataframe(confusion_data)
                
                # Export options
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # CSV export
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name=f"accuracy_assessment_{timestamp}.csv",
                    mime="text/csv"
                )
                
                # GeoJSON export
                results_gdf = gpd.GeoDataFrame(
                    results_df,
                    geometry=[Point(xy) for xy in zip(results_df['x_coord'], results_df['y_coord'])],
                    crs=st.session_state.assessment_points.crs
                )
                
                geojson_str = results_gdf.to_json()
                st.download_button(
                    label="🗺️ Download GeoJSON",
                    data=geojson_str,
                    file_name=f"accuracy_assessment_{timestamp}.geojson",
                    mime="application/json"
                )
                
                # Summary report
                report = f"""
                ACCURACY ASSESSMENT REPORT
                ===========================
                Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                Total Points: {total_points}
                Assessed Points: {assessed_points}
                
                Sampling Method: {sample_method}
                Sample Size: {sample_size}
                Minimum per Class: {min_samples_per_class}
                
                OVERALL ACCURACY: {overall_accuracy:.2%}
                
                Class-wise Accuracy:
                """
                
                for class_name in class_list:
                    if class_name in users_accuracy.index and class_name in producers_accuracy.index:
                        report += f"\n{class_name}:"
                        report += f"\n  User's Accuracy: {users_accuracy[class_name]:.2%}"
                        report += f"\n  Producer's Accuracy: {producers_accuracy[class_name]:.2%}"
                
                st.download_button(
                    label="📋 Download Report",
                    data=report,
                    file_name=f"accuracy_report_{timestamp}.txt",
                    mime="text/plain"
                )
                
            else:
                st.warning("No assessments have been completed yet.")

