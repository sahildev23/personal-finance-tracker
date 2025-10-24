"""
Personal Finance Tracker with Time-Series Forecasting
Automated transaction analysis, anomaly detection, and spending prediction
using ARIMA and Prophet models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# For time-series forecasting
try:
    from statsmodels.tsa.arima.model import ARIMA
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("Prophet not installed. Install with: pip install prophet")
    PROPHET_AVAILABLE = False

class FinanceTracker:
    def __init__(self, transactions_file=None):
        """Initialize the finance tracker."""
        self.transactions_file = transactions_file
        self.df = None
        self.monthly_summary = None
        self.anomalies = None
        
    def load_transactions(self):
        """Load and clean transaction data."""
        print("Loading transactions...")
        self.df = pd.read_csv(self.transactions_file)
        
        # Convert date column to datetime
        date_col = [col for col in self.df.columns if 'date' in col.lower()][0]
        self.df['date'] = pd.to_datetime(self.df[date_col])
        
        # Standardize amount column
        amount_col = [col for col in self.df.columns if 'amount' in col.lower()][0]
        self.df['amount'] = pd.to_numeric(self.df[amount_col], errors='coerce')
        
        # Remove null values
        self.df.dropna(subset=['date', 'amount'], inplace=True)
        
        # Sort by date
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} transactions")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        
        return self.df
    
    def clean_and_categorize(self):
        """Automated data cleaning and categorization."""
        print("\nCleaning and categorizing transactions...")
        
        # Add time-based features
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['quarter'] = self.df['date'].dt.quarter
        
        # Categorize transaction types (if description column exists)
        if 'description' in [c.lower() for c in self.df.columns]:
            desc_col = [col for col in self.df.columns if 'description' in col.lower()][0]
            self.df['category'] = self.df[desc_col].apply(self._categorize_transaction)
        else:
            self.df['category'] = 'Uncategorized'
        
        # Flag income vs expense
        self.df['type'] = self.df['amount'].apply(lambda x: 'Income' if x > 0 else 'Expense')
        self.df['abs_amount'] = self.df['amount'].abs()
        
        print(f"Categorized into {self.df['category'].nunique()} categories")
        
    def _categorize_transaction(self, description):
        """Simple rule-based categorization."""
        desc = str(description).lower()
        
        categories = {
            'Food & Dining': ['restaurant', 'cafe', 'food', 'dining', 'grocery', 'doordash', 'ubereats'],
            'Transportation': ['gas', 'uber', 'lyft', 'parking', 'transit', 'auto'],
            'Shopping': ['amazon', 'store', 'retail', 'shopping', 'walmart', 'target'],
            'Bills & Utilities': ['electric', 'internet', 'phone', 'utility', 'water', 'rent'],
            'Entertainment': ['movie', 'concert', 'game', 'entertainment', 'spotify', 'netflix'],
            'Health': ['pharmacy', 'doctor', 'hospital', 'health', 'medical'],
            'Income': ['salary', 'paycheck', 'deposit', 'income', 'payment received']
        }
        
        for category, keywords in categories.items():
            if any(keyword in desc for keyword in keywords):
                return category
        
        return 'Other'
    
    def detect_anomalies(self):
        """Detect anomalous transactions using Isolation Forest."""
        print("\nDetecting anomalies...")
        
        # Prepare features for anomaly detection
        features = self.df[['abs_amount', 'day_of_week', 'month']].copy()
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.df['anomaly'] = iso_forest.fit_predict(features_scaled)
        
        # -1 indicates anomaly, 1 indicates normal
        self.anomalies = self.df[self.df['anomaly'] == -1].copy()
        
        print(f"Detected {len(self.anomalies)} anomalous transactions")
        print(f"Anomaly rate: {len(self.anomalies)/len(self.df)*100:.2f}%")
        
        return self.anomalies
    
    def calculate_monthly_summary(self):
        """Calculate monthly spending summary."""
        print("\nCalculating monthly summary...")
        
        self.df['year_month'] = self.df['date'].dt.to_period('M')
        
        self.monthly_summary = self.df.groupby('year_month').agg({
            'amount': ['sum', 'mean', 'count'],
            'abs_amount': ['sum', 'mean']
        }).reset_index()
        
        self.monthly_summary.columns = ['year_month', 'net_amount', 'avg_transaction', 
                                        'transaction_count', 'total_spending', 'avg_spending']
        
        # Convert period to timestamp for forecasting
        self.monthly_summary['date'] = self.monthly_summary['year_month'].dt.to_timestamp()
        
        return self.monthly_summary
    
    def forecast_arima(self, periods=6):
        """Forecast spending using ARIMA model."""
        print(f"\nForecasting next {periods} months with ARIMA...")
        
        # Prepare time series
        ts_data = self.monthly_summary.set_index('date')['total_spending']
        
        # Fit ARIMA model
        model = ARIMA(ts_data, order=(1, 1, 1))
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=periods)
        
        # Calculate error metrics on historical data
        predictions = fitted_model.predict(start=1, end=len(ts_data))
        mae = np.mean(np.abs(predictions - ts_data[1:]))
        mape = np.mean(np.abs((ts_data[1:] - predictions) / ts_data[1:])) * 100
        
        print(f"ARIMA Model Performance:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return forecast, fitted_model
    
    def forecast_prophet(self, periods=6):
        """Forecast spending using Prophet model."""
        if not PROPHET_AVAILABLE:
            print("Prophet not available. Using ARIMA only.")
            return None, None
        
        print(f"\nForecasting next {periods} months with Prophet...")
        
        # Prepare data for Prophet
        prophet_df = self.monthly_summary[['date', 'total_spending']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Fit Prophet model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(prophet_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq='MS')
        forecast = model.predict(future)
        
        # Calculate error metrics
        predictions = forecast.set_index('ds').loc[prophet_df['ds'], 'yhat']
        mae = np.mean(np.abs(predictions.values - prophet_df['y'].values))
        mape = np.mean(np.abs((prophet_df['y'].values - predictions.values) / prophet_df['y'].values)) * 100
        
        print(f"Prophet Model Performance:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return forecast, model
    
    def visualize_dashboard(self):
        """Create comprehensive visualization dashboard."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Monthly Spending Trend
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(self.monthly_summary['date'], self.monthly_summary['total_spending'], 
                marker='o', linewidth=2, markersize=6, label='Actual')
        ax1.set_title('Monthly Spending Trend', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Total Spending ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Category Breakdown (Pie Chart)
        ax2 = fig.add_subplot(gs[0, 2])
        category_totals = self.df[self.df['amount'] < 0].groupby('category')['abs_amount'].sum()
        ax2.pie(category_totals.values, labels=category_totals.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Spending by Category', fontsize=14, fontweight='bold')
        
        # 3. Daily Spending Pattern
        ax3 = fig.add_subplot(gs[1, 0])
        daily_avg = self.df.groupby('day_of_week')['abs_amount'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax3.bar(range(7), daily_avg.values, color='steelblue', alpha=0.7)
        ax3.set_xticks(range(7))
        ax3.set_xticklabels(days, rotation=45)
        ax3.set_title('Avg Spending by Day of Week', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Average Amount ($)')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Transaction Count Over Time
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(self.monthly_summary['date'], self.monthly_summary['transaction_count'], 
                marker='s', color='coral', linewidth=2)
        ax4.set_title('Transaction Count Trend', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Count')
        ax4.grid(True, alpha=0.3)
        
        # 5. Top Anomalies
        ax5 = fig.add_subplot(gs[1, 2])
        top_anomalies = self.anomalies.nlargest(10, 'abs_amount')[['date', 'abs_amount']]
        ax5.barh(range(len(top_anomalies)), top_anomalies['abs_amount'].values, color='red', alpha=0.6)
        ax5.set_yticks(range(len(top_anomalies)))
        ax5.set_yticklabels([d.strftime('%Y-%m-%d') for d in top_anomalies['date']], fontsize=8)
        ax5.set_title('Top 10 Anomalous Transactions', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Amount ($)')
        
        # 6. Spending Forecast
        ax6 = fig.add_subplot(gs[2, :])
        
        # Plot historical data
        ax6.plot(self.monthly_summary['date'], self.monthly_summary['total_spending'], 
                marker='o', linewidth=2, label='Historical', color='steelblue')
        
        # ARIMA forecast
        arima_forecast, _ = self.forecast_arima(periods=6)
        last_date = self.monthly_summary['date'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='MS')
        ax6.plot(forecast_dates, arima_forecast, marker='s', linewidth=2, 
                linestyle='--', label='ARIMA Forecast', color='green')
        
        # Prophet forecast (if available)
        if PROPHET_AVAILABLE:
            prophet_forecast, _ = self.forecast_prophet(periods=6)
            prophet_future = prophet_forecast[prophet_forecast['ds'] > last_date]
            ax6.plot(prophet_future['ds'], prophet_future['yhat'], marker='^', 
                    linewidth=2, linestyle='--', label='Prophet Forecast', color='orange')
            ax6.fill_between(prophet_future['ds'], prophet_future['yhat_lower'], 
                            prophet_future['yhat_upper'], alpha=0.2, color='orange')
        
        ax6.set_title('Spending Forecast (6 Months)', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Total Spending ($)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.savefig('finance_dashboard.png', dpi=300, bbox_inches='tight')
        print("\nDashboard saved as 'finance_dashboard.png'")
        plt.show()
    
    def generate_report(self):
        """Generate detailed financial report."""
        print("\n" + "="*70)
        print("FINANCIAL ANALYSIS REPORT")
        print("="*70)
        
        total_income = self.df[self.df['amount'] > 0]['amount'].sum()
        total_expenses = self.df[self.df['amount'] < 0]['abs_amount'].sum()
        net_savings = total_income - total_expenses
        
        print(f"\nOverall Summary:")
        print(f"  Total Income:    ${total_income:,.2f}")
        print(f"  Total Expenses:  ${total_expenses:,.2f}")
        print(f"  Net Savings:     ${net_savings:,.2f}")
        print(f"  Savings Rate:    {(net_savings/total_income)*100:.2f}%")
        
        print(f"\nTop Spending Categories:")
        category_totals = self.df[self.df['amount'] < 0].groupby('category')['abs_amount'].sum().sort_values(ascending=False)
        for cat, amount in category_totals.head(5).items():
            print(f"  {cat:20s} ${amount:,.2f}")
        
        print(f"\nMonthly Averages:")
        print(f"  Avg Monthly Spending: ${self.monthly_summary['total_spending'].mean():,.2f}")
        print(f"  Avg Transaction Count: {self.monthly_summary['transaction_count'].mean():.0f}")
        
        print(f"\nAnomaly Detection:")
        print(f"  Total Anomalies: {len(self.anomalies)}")
        print(f"  Avg Anomaly Amount: ${self.anomalies['abs_amount'].mean():,.2f}")
    
    def run_full_analysis(self):
        """Execute complete finance tracking pipeline."""
        self.load_transactions()
        self.clean_and_categorize()
        self.detect_anomalies()
        self.calculate_monthly_summary()
        self.visualize_dashboard()
        self.generate_report()


# Generate sample transaction data for demonstration
def generate_sample_data(n_transactions=5000):
    """Generate sample transaction data."""
    np.random.seed(42)
    
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 1095)) for _ in range(n_transactions)]
    
    categories = ['Food & Dining', 'Transportation', 'Shopping', 'Bills & Utilities', 
                  'Entertainment', 'Health', 'Other']
    
    transactions = []
    for date in dates:
        # 90% expenses, 10% income
        if np.random.random() < 0.9:
            amount = -np.random.lognormal(3, 1)  # Expenses
            category = np.random.choice(categories)
            desc = f"{category} transaction"
        else:
            amount = np.random.uniform(2000, 5000)  # Income
            desc = "Salary deposit"
        
        transactions.append({
            'date': date,
            'amount': amount,
            'description': desc
        })
    
    df = pd.DataFrame(transactions)
    df.to_csv('sample_transactions.csv', index=False)
    print(f"Generated {n_transactions} sample transactions")


# Example usage
if __name__ == "__main__":
    print("Personal Finance Tracker")
    print("="*70)
    
    # Generate sample data
    generate_sample_data(n_transactions=5000)
    
    # Run analysis
    tracker = FinanceTracker('sample_transactions.csv')
    tracker.run_full_analysis()
    
    print("\n" + "="*70)
    print("Analysis completed successfully!")
    print("="*70)
