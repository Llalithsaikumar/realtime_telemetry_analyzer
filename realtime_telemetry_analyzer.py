import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from robot.api import logger
from robot.api.deco import keyword, library
import time
import threading
import queue
import random
import datetime
from collections import deque

class RealTimeDataSimulator:
    
    def __init__(self):
        self.running = False
        self.data_queue = queue.Queue()
        
    def generate_telemetry_data(self):
        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'Latitude': random.uniform(-90, 90),
            'Longitude': random.uniform(-180, 180),
            'Altitude': random.uniform(0, 1000),
            'Speed': random.uniform(0, 100),
            'Acceleration': random.uniform(-20, 20),
            'Deceleration': random.uniform(-20, 20),
            'Roll': random.uniform(-180, 180),
            'Pitch': random.uniform(-90, 90),
            'Yaw': random.uniform(-180, 180)
        }
        
    def start(self):
        self.running = True
        threading.Thread(target=self._run_simulator).start()
        
    def stop(self):
        self.running = False
        
    def _run_simulator(self):
        while self.running:
            data = self.generate_telemetry_data()
            self.data_queue.put(data)
            time.sleep(1)  # Generate data every second
            
    def get_data(self):
        return self.data_queue.get() if not self.data_queue.empty() else None

@library
class RoboticTelemetryAnalyzer:
    
    ROBOT_LIBRARY_SCOPE = 'TEST SUITE'
    
    def __init__(self, buffer_size=100, update_interval=1.0):
        self.anomaly_threshold = 70  # Risk threshold
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.data_buffer = deque(maxlen=buffer_size)  # Using deque for efficient buffer management
        self.running = False
        self.data_source = RealTimeDataSimulator()
        self.visualization_thread = None
        self.analysis_results = queue.Queue()
        self.is_scaler_fitted = False 
        self.fig, self.axes = None, None  # Store plot objects for proper closing
        
    @keyword('Start Real-Time Analysis')
    def start_analysis(self):
        """Start real-time telemetry analysis"""
        self.running = True
        self.data_source.start()
        self.visualization_thread = threading.Thread(target=self._run_analysis_loop)
        self.visualization_thread.start()
        logger.info("Started real-time telemetry analysis")
        
    @keyword('Stop Real-Time Analysis')
    def stop_analysis(self):
        self.running = False
        self.data_source.stop()
        if self.visualization_thread:
            self.visualization_thread.join()
        logger.info("Stopped real-time telemetry analysis")
        
    def _update_data_buffer(self, new_data):
        self.data_buffer.append(new_data)  # Efficiently append to deque
        
    def calculate_risk_score(self, df):
        if len(df) < 2:  # Need at least 2 samples for meaningful analysis
            return df
            
        features = ['Latitude', 'Longitude', 'Altitude', 
                   'Speed', 'Acceleration', 'Deceleration', 
                   'Roll', 'Pitch', 'Yaw']
        
        if not self.is_scaler_fitted:
            self.scaler.fit(df[features])  
            self.is_scaler_fitted = True  
        
        scaled_data = self.scaler.transform(df[features])  
        anomaly_labels = self.isolation_forest.fit_predict(scaled_data)
        
        deviations = np.abs(scaled_data - np.mean(scaled_data, axis=0))
        anomaly_scores = np.max(deviations, axis=1)
        
        risk_scores = anomaly_scores * 100
        risk_levels = ['Low Risk' if score <= self.anomaly_threshold else 'High Risk' for score in risk_scores]
        
        df['RiskScore'] = risk_scores
        df['RiskLevel'] = risk_levels
        
        return df
        
    def _create_real_time_plot(self):
        plt.ion()  # Enable interactive mode
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.canvas.manager.set_window_title('Real-time Telemetry Analysis')
        return fig, axes
        
    def _update_plot(self, fig, axes, df):
        df_to_plot = df.tail(100)  # Display only the last 100 points
        
        for i, feature in enumerate(['Latitude', 'Longitude', 'Altitude', 'Speed', 'Acceleration']):
            row, col = divmod(i, 3)
            ax = axes[row, col]
            ax.clear()
            
            # Highlight high-risk values in red
            ax.plot(df_to_plot.index, df_to_plot[feature], 
                    color='blue' if df_to_plot[feature].iloc[-1] <= self.anomaly_threshold else 'red',  
                    label=feature) 
            
            ax.set_title(feature)
            ax.set_xlabel('Time')
            ax.set_ylabel(feature)
            ax.legend()
            
            # Add a horizontal line for the risk level
            ax.axhline(y=self.anomaly_threshold, color='black', linestyle='--', label='Risk Threshold')
            
            ax.legend()

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
    def _run_analysis_loop(self):

        last_update = time.time()
        
        self.fig, self.axes = self._create_real_time_plot()

        try:
            while self.running:
                current_time = time.time()
                
                # Get new data
                new_data = self.data_source.get_data()
                if new_data:
                    self._update_data_buffer(new_data)
                    
                    # Perform analysis and update visualization at specified interval
                    if current_time - last_update >= self.update_interval:
                        analyzed_df = self.calculate_risk_score(pd.DataFrame(self.data_buffer))
                        
                        # Store analysis results
                        result = {
                            'timestamp': new_data['timestamp'],
                            'high_risk_count': len(analyzed_df[analyzed_df['RiskLevel'] == 'High Risk']),
                            'total_count': len(analyzed_df),
                            'latest_risk_score': analyzed_df['RiskScore'].iloc[-1],
                            'latest_risk_level': analyzed_df['RiskLevel'].iloc[-1]
                        }
                        self.analysis_results.put(result)
                        
                        last_update = current_time
                        
                        # Update plot in the main thread
                        self._update_plot(self.fig, self.axes, analyzed_df)  # Use stored objects
                        
                        # Print threat alert based on risk level
                        if result['latest_risk_level'] == 'High Risk':
                            print(f"Threat Alert: High risk detected at {result['timestamp']}")
                        
                time.sleep(0.1) #sleep for 0.1 seconds
                
        except Exception as e:
            logger.error(f"Error in analysis loop: {e}")
            
        finally:
            if self.fig:
                plt.close(self.fig) 
            
    @keyword('Get Latest Analysis Results')
    def get_latest_results(self):
        if not self.analysis_results.empty():
            return self.analysis_results.get()
        return None

if __name__ == "__main__":
    analyzer = RoboticTelemetryAnalyzer(buffer_size=100, update_interval=1.0)
    
    try:
        analyzer.start_analysis()
        print("Real-time analysis started. Press Ctrl+C to stop...")
        
        while True:
            results = analyzer.get_latest_results()
            if results:
                print(f"\rLatest Results - Time: {results['timestamp']}, "
                      f"Risk Level: {results['latest_risk_level']}, "
                      f"Risk Score: {results['latest_risk_score']:.2f}", end='')

            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping analysis...")
        analyzer.stop_analysis()