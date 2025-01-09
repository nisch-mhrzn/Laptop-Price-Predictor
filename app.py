import streamlit as st
import pickle
import numpy as np

# Load the model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Page configuration
st.set_page_config(page_title="Laptop Price Predictor", page_icon="\U0001F4BB", layout="wide")

# App Title and Description
st.title("Laptop Price Predictor \U0001F4BB")
st.markdown("""
Predict the price of a laptop based on its specifications. 
Fill in the details below to get an estimate of the laptop's price.
""")

# Inputs with improved design
with st.expander("Laptop Specifications", expanded=True):
    # Brand
    company = st.selectbox('Select Laptop Brand', df['Company'].unique(), help="Choose the manufacturer of the laptop.")

    # Laptop Type
    type = st.selectbox('Select Laptop Type', df['TypeName'].unique(), help="Select the type of laptop, e.g., Ultrabook, Gaming.")

    # RAM
    ram = st.selectbox('Select RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64], help="Choose the amount of RAM the laptop has.")

    # Weight
    weight = st.number_input('Enter Laptop Weight (in kg)', min_value=0.5, max_value=5.0, step=0.1, value=1.5, help="Enter the weight of the laptop in kilograms.")

    # Touchscreen
    touchscreen = st.radio('Touchscreen Feature', ['No', 'Yes'], help="Does the laptop have a touchscreen?")

    # IPS
    ips = st.radio('IPS Display', ['No', 'Yes'], help="Does the laptop have an IPS display?")

    # Screen Size
    screen_size = st.slider('Select Screen Size (in inches)', 10.0, 18.0, 15.6, help="Choose the screen size of the laptop.")

    # Resolution
    resolution = st.selectbox(
        'Select Screen Resolution',
        ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'],
        help="Choose the screen resolution.")

    # CPU
    cpu = st.selectbox('Select CPU', df['Cpu_brand'].unique(), help="Choose the processor brand.")

    # HDD
    hdd = st.selectbox('Select HDD Capacity (in GB)', [0, 128, 256, 512, 1024, 2048], help="Choose the size of the Hard Disk Drive.")

    # SSD
    ssd = st.selectbox('Select SSD Capacity (in GB)', [0, 128, 256, 512, 1024], help="Choose the size of the Solid State Drive.")

    # GPU
    gpu = st.selectbox('Select GPU', df['Gpu_brand'].unique(), help="Choose the Graphics Processing Unit (GPU) brand.")

    # OS
    os = st.selectbox('Select Operating System', df['os'].unique(), help="Choose the operating system.")

# Predict Price Button
if st.button('Predict Price'):
    try:
        # Calculate Pixels Per Inch (PPI)
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

        # Prepare query
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0
        query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, -1)

        # Predict price
        predicted_price_inr = int(np.exp(pipe.predict(query)[0]))
        predicted_price_npr = int(predicted_price_inr * 1.6)  # Approximate conversion rate

        # Display result
        st.success(f"The predicted price of this configuration is: ₹{predicted_price_inr:,} (INR) / NPR {predicted_price_npr:,}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown("""
---
**Note**: This prediction is based on machine learning models and may not be 100% accurate.

Checkout my Github [Nischal Maharjan](https://github.com/nisch-mhrzn)
""")