import streamlit as st
import pandas as pd
import time
import os

st.set_page_config(page_title="Network Flow Analysis", layout="wide")
st.title("Suspicious Network Behavior Detection")

csv_file = "flow_predictions.csv"

# Function to load data
def load_data():
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            return df
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return pd.DataFrame(columns=['Timestamp', 'Source IP', 'Destination IP', 'Domain Name', 'Protocol', 'Prediction', 'Confidence'])
    else:
        return pd.DataFrame(columns=['Timestamp', 'Source IP', 'Destination IP', 'Domain Name', 'Protocol', 'Prediction', 'Confidence'])

placeholder = st.empty()

while True:
    df = load_data()
    with placeholder.container():
        st.subheader("Live Network Flows Predictions")
        
        # Add basic stats
        if not df.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Flows Analyzed", len(df))
            with col2:
                # Count alerts
                if 'Prediction' in df.columns:
                    predictions_counts = df['Prediction'].value_counts()
                    st.write("Predictions counts:")
                    st.dataframe(predictions_counts)
            
            # Select desired columns to display
            display_columns = ['Source IP', 'Destination IP', 'Domain Name', 'Protocol', 'Prediction', 'Confidence']
            
            # Show the table honoring exact requested columns
            if all(col in df.columns for col in display_columns):
                st.dataframe(df[display_columns].tail(100).sort_index(ascending=False), use_container_width=True)
            else:
                st.dataframe(df.tail(100).sort_index(ascending=False), use_container_width=True)
        else:
            st.info("No network flows detected yet. Please make sure the sniffer is running.")
            
    time.sleep(2)
