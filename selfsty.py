"""
Sales Prediction Project
========================
A comprehensive data science project for predicting sales using multiple machine learning models.

Features:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Multiple ML Models (Linear Regression, Random Forest, XGBoost, LightGBM)
- Model Evaluation and Comparison
- Visualization and Insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SalesPredictionPipeline:
    """
    End-to-end pipeline for sales prediction
    """
    
    def __init__(self, sales_path, products_path, inventory_path, suppliers_path):
        """Initialize with data paths"""
        self.sales_path = sales_path
        self.products_path = products_path
        self.inventory_path = inventory_path
        self.suppliers_path = suppliers_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_and_merge_data(self):
        """Load all datasets and merge them"""
        print("Loading datasets...")
        
        # Load all datasets
        sales = pd.read_csv(self.sales_path)
        products = pd.read_csv(self.products_path)
        inventory = pd.read_csv(self.inventory_path)
        suppliers = pd.read_csv(self.suppliers_path)
        
        print(f"Sales records: {len(sales)}")
        print(f"Products: {len(products)}")
        print(f"Inventory items: {len(inventory)}")
        print(f"Suppliers: {len(suppliers)}")
        
        # Merge datasets
        self.df = sales.merge(products, on='sku_id', how='left')
        self.df = self.df.merge(inventory, on='sku_id', how='left')
        self.df = self.df.merge(suppliers, left_on='supplier_id_x', right_on='supplier_id', how='left')
        
        print(f"\nMerged dataset shape: {self.df.shape}")
        return self.df
    
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic info
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nBasic Statistics:")
        print(self.df.describe())
        
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0])
        
        # Sales distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Units sold distribution
        axes[0, 0].hist(self.df['units_sold'], bins=50, edgecolor='black')
        axes[0, 0].set_title('Distribution of Units Sold')
        axes[0, 0].set_xlabel('Units Sold')
        axes[0, 0].set_ylabel('Frequency')
        
        # Gross revenue distribution
        axes[0, 1].hist(self.df['gross_revenue'], bins=50, edgecolor='black', color='green')
        axes[0, 1].set_title('Distribution of Gross Revenue')
        axes[0, 1].set_xlabel('Gross Revenue')
        axes[0, 1].set_ylabel('Frequency')
        
        # Sales by category
        if 'category' in self.df.columns:
            category_sales = self.df.groupby('category')['units_sold'].sum().sort_values(ascending=True)
            axes[1, 0].barh(category_sales.index, category_sales.values)
            axes[1, 0].set_title('Total Sales by Category')
            axes[1, 0].set_xlabel('Total Units Sold')
        
        # Sales trend over time
        if 'date' in self.df.columns:
            self.df['date_parsed'] = pd.to_datetime(self.df['date'], format='%m/%d/%y')
            daily_sales = self.df.groupby('date_parsed')['units_sold'].sum()
            axes[1, 1].plot(daily_sales.index, daily_sales.values)
            axes[1, 1].set_title('Sales Trend Over Time')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Total Units Sold')
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('/home/claude/eda_visualizations.png', dpi=300, bbox_inches='tight')
        print("\nEDA visualizations saved to 'eda_visualizations.png'")
        plt.close()
        
        # Correlation heatmap
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1)
        plt.title('Correlation Heatmap of Numeric Features')
        plt.tight_layout()
        plt.savefig('/home/claude/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Correlation heatmap saved to 'correlation_heatmap.png'")
        plt.close()
        
    def feature_engineering(self):
        """Create new features for better predictions"""
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)
        
        # Parse date
        self.df['date_parsed'] = pd.to_datetime(self.df['date'], format='%m/%d/%y')
        
        # Extract time-based features
        self.df['day'] = self.df['date_parsed'].dt.day
        self.df['month'] = self.df['date_parsed'].dt.month
        self.df['year'] = self.df['date_parsed'].dt.year
        self.df['quarter'] = self.df['date_parsed'].dt.quarter
        self.df['day_of_year'] = self.df['date_parsed'].dt.dayofyear
        
        # Price-based features
        self.df['discount_amount'] = self.df['mrp'] - self.df['selling_price']
        self.df['profit_margin'] = (self.df['selling_price'] - self.df['cost_price']) / self.df['selling_price']
        self.df['price_ratio'] = self.df['selling_price'] / self.df['mrp']
        
        # Inventory features
        self.df['stock_to_safety_ratio'] = self.df['current_stock'] / (self.df['safety_stock'] + 1)
        self.df['days_until_stockout'] = self.df['current_stock'] / (self.df['units_sold'] + 1)
        
        # Lag features (simplified - using rolling statistics)
        self.df = self.df.sort_values(['sku_id', 'date_parsed'])
        self.df['sales_rolling_mean_7d'] = self.df.groupby('sku_id')['units_sold'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        self.df['sales_rolling_std_7d'] = self.df.groupby('sku_id')['units_sold'].transform(
            lambda x: x.rolling(window=7, min_periods=1).std()
        )
        
        # Encode categorical variables
        label_encoders = {}
        categorical_cols = ['category', 'sub_category', 'season_tag', 'platform_traffic_source']
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                label_encoders[col] = le
        
        print(f"Created {len([c for c in self.df.columns if c not in pd.read_csv(self.sales_path).columns])} new features")
        print("\nFeature engineering completed!")
        
        return self.df
    
    def prepare_features(self):
        """Prepare features for modeling"""
        print("\n" + "="*50)
        print("PREPARING FEATURES FOR MODELING")
        print("="*50)
        
        # Define target variable
        target = 'units_sold'
        
        # Select features for modeling
        feature_cols = [
            # Price features
            'selling_price', 'mrp', 'cost_price', 'discount_pct', 
            'discount_amount', 'profit_margin', 'price_ratio',
            
            # Inventory features
            'current_stock', 'safety_stock', 'reorder_point', 'lead_time_days',
            'stock_to_safety_ratio', 'days_until_stockout',
            
            # Promotional features
            'promo_flag', 'bundle_offer_flag', 'campaign_intensity',
            
            # Time features
            'month', 'week_of_year', 'day_of_week', 'is_weekend', 
            'is_festival_month', 'is_payday_period', 'holiday_flag', 'quarter', 'day_of_year',
            
            # Market features
            'weather_index', 'traffic_index', 'competitor_price_index', 
            'competitor_stockout_flag',
            
            # Product features
            'stock_visibility_score', 'rating_score', 'review_volume', 
            'product_visibility_rank',
            
            # Lag features
            'sales_rolling_mean_7d', 'sales_rolling_std_7d',
            
            # Categorical encoded
            'category_encoded', 'sub_category_encoded', 'season_tag_encoded',
            'platform_traffic_source_encoded'
        ]
        
        # Filter only available columns
        available_features = [col for col in feature_cols if col in self.df.columns]
        
        # Remove rows with missing values in key columns
        df_clean = self.df[available_features + [target]].dropna()
        
        print(f"Features selected: {len(available_features)}")
        print(f"Samples after cleaning: {len(df_clean)}")
        
        # Split features and target
        X = df_clean[available_features]
        y = df_clean[target]
        
        # Train-test split (80-20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
        return available_features
    
    def train_models(self):
        """Train multiple machine learning models"""
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        # Define models
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        }
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for linear models, original for tree-based
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            
            # Train model
            model.fit(X_train_use, self.y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train_use)
            y_pred_test = model.predict(X_test_use)
            
            # Calculate metrics
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            test_mape = mean_absolute_percentage_error(self.y_test, y_pred_test) * 100
            
            # Store results
            self.results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'test_mape': test_mape,
                'predictions': y_pred_test
            }
            
            print(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
            print(f"  Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")
            print(f"  Test MAE: {test_mae:.2f} | Test MAPE: {test_mape:.2f}%")
    
    def evaluate_models(self):
        """Compare and visualize model performance"""
        print("\n" + "="*50)
        print("MODEL EVALUATION & COMPARISON")
        print("="*50)
        
        # Create comparison dataframe
        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Test R²': [self.results[m]['test_r2'] for m in self.results],
            'Test RMSE': [self.results[m]['test_rmse'] for m in self.results],
            'Test MAE': [self.results[m]['test_mae'] for m in self.results],
            'Test MAPE (%)': [self.results[m]['test_mape'] for m in self.results]
        })
        
        comparison = comparison.sort_values('Test R²', ascending=False)
        print("\nModel Performance Comparison:")
        print(comparison.to_string(index=False))
        
        # Save comparison
        comparison.to_csv('/home/claude/model_comparison.csv', index=False)
        print("\nModel comparison saved to 'model_comparison.csv'")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # R² Score comparison
        axes[0, 0].barh(comparison['Model'], comparison['Test R²'], color='skyblue')
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].set_title('Model Comparison - R² Score (Higher is Better)')
        axes[0, 0].set_xlim(0, 1)
        
        # RMSE comparison
        axes[0, 1].barh(comparison['Model'], comparison['Test RMSE'], color='salmon')
        axes[0, 1].set_xlabel('RMSE')
        axes[0, 1].set_title('Model Comparison - RMSE (Lower is Better)')
        
        # MAE comparison
        axes[1, 0].barh(comparison['Model'], comparison['Test MAE'], color='lightgreen')
        axes[1, 0].set_xlabel('MAE')
        axes[1, 0].set_title('Model Comparison - MAE (Lower is Better)')
        
        # MAPE comparison
        axes[1, 1].barh(comparison['Model'], comparison['Test MAPE (%)'], color='gold')
        axes[1, 1].set_xlabel('MAPE (%)')
        axes[1, 1].set_title('Model Comparison - MAPE (Lower is Better)')
        
        plt.tight_layout()
        plt.savefig('/home/claude/model_comparison.png', dpi=300, bbox_inches='tight')
        print("Model comparison chart saved to 'model_comparison.png'")
        plt.close()
        
        # Best model predictions vs actual
        best_model_name = comparison.iloc[0]['Model']
        best_predictions = self.results[best_model_name]['predictions']
        
        plt.figure(figsize=(12, 6))
        plt.scatter(self.y_test, best_predictions, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                 [self.y_test.min(), self.y_test.max()], 
                 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Units Sold')
        plt.ylabel('Predicted Units Sold')
        plt.title(f'Best Model: {best_model_name} - Predictions vs Actual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/home/claude/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        print("Predictions vs actual plot saved to 'predictions_vs_actual.png'")
        plt.close()
        
        return best_model_name
    
    def feature_importance_analysis(self, top_n=15):
        """Analyze feature importance for tree-based models"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        # Get feature importance from Random Forest and XGBoost
        rf_model = self.results['Random Forest']['model']
        xgb_model = self.results['XGBoost']['model']
        
        feature_names = self.X_train.columns
        
        # Random Forest importance
        rf_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        
        # XGBoost importance
        xgb_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].barh(rf_importance['Feature'], rf_importance['Importance'], color='forestgreen')
        axes[0].set_xlabel('Importance')
        axes[0].set_title(f'Top {top_n} Features - Random Forest')
        axes[0].invert_yaxis()
        
        axes[1].barh(xgb_importance['Feature'], xgb_importance['Importance'], color='orange')
        axes[1].set_xlabel('Importance')
        axes[1].set_title(f'Top {top_n} Features - XGBoost')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('/home/claude/feature_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance plot saved to 'feature_importance.png'")
        plt.close()
        
        print("\nTop 10 Most Important Features (Random Forest):")
        print(rf_importance.head(10).to_string(index=False))
    
    def save_best_model(self, best_model_name):
        """Save the best performing model"""
        print("\n" + "="*50)
        print("SAVING BEST MODEL")
        print("="*50)
        
        best_model = self.results[best_model_name]['model']
        
        # Save model
        joblib.dump(best_model, '/home/claude/best_sales_model.pkl')
        joblib.dump(self.scaler, '/home/claude/scaler.pkl')
        
        # Save feature names
        feature_names = self.X_train.columns.tolist()
        joblib.dump(feature_names, '/home/claude/feature_names.pkl')
        
        print(f"Best model ({best_model_name}) saved to 'best_sales_model.pkl'")
        print("Scaler saved to 'scaler.pkl'")
        print("Feature names saved to 'feature_names.pkl'")
        
        # Create model info file
        model_info = f"""
Sales Prediction Model Information
===================================

Best Model: {best_model_name}

Performance Metrics (Test Set):
- R² Score: {self.results[best_model_name]['test_r2']:.4f}
- RMSE: {self.results[best_model_name]['test_rmse']:.2f}
- MAE: {self.results[best_model_name]['test_mae']:.2f}
- MAPE: {self.results[best_model_name]['test_mape']:.2f}%

Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Number of Features: {len(feature_names)}
Training Samples: {len(self.X_train)}
Test Samples: {len(self.X_test)}

Features Used:
{', '.join(feature_names)}
"""
        
        with open('/home/claude/model_info.txt', 'w') as f:
            f.write(model_info)
        
        print("Model information saved to 'model_info.txt'")
    
    def generate_insights(self):
        """Generate business insights from the analysis"""
        print("\n" + "="*50)
        print("BUSINESS INSIGHTS")
        print("="*50)
        
        insights = []
        
        # Sales patterns
        avg_sales = self.df['units_sold'].mean()
        max_sales = self.df['units_sold'].max()
        insights.append(f"Average daily sales per SKU: {avg_sales:.2f} units")
        insights.append(f"Maximum daily sales recorded: {max_sales:.0f} units")
        
        # Promotional impact
        promo_sales = self.df[self.df['promo_flag'] == 1]['units_sold'].mean()
        non_promo_sales = self.df[self.df['promo_flag'] == 0]['units_sold'].mean()
        promo_lift = ((promo_sales - non_promo_sales) / non_promo_sales) * 100
        insights.append(f"Promotional lift: {promo_lift:.1f}% increase in sales during promotions")
        
        # Weekend effect
        weekend_sales = self.df[self.df['is_weekend'] == 1]['units_sold'].mean()
        weekday_sales = self.df[self.df['is_weekend'] == 0]['units_sold'].mean()
        weekend_diff = ((weekend_sales - weekday_sales) / weekday_sales) * 100
        insights.append(f"Weekend vs Weekday: {weekend_diff:+.1f}% difference in sales")
        
        # Category performance
        if 'category' in self.df.columns:
            top_category = self.df.groupby('category')['units_sold'].sum().idxmax()
            insights.append(f"Top performing category: {top_category}")
        
        # Print insights
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        # Save insights
        with open('/home/claude/business_insights.txt', 'w') as f:
            f.write("Business Insights from Sales Prediction Analysis\n")
            f.write("=" * 50 + "\n\n")
            for insight in insights:
                f.write(f"• {insight}\n")
        
        print("\nBusiness insights saved to 'business_insights.txt'")
    
    def run_pipeline(self):
        """Execute the complete pipeline"""
        print("="*50)
        print("SALES PREDICTION PIPELINE - STARTING")
        print("="*50)
        
        # Step 1: Load and merge data
        self.load_and_merge_data()
        
        # Step 2: EDA
        self.exploratory_data_analysis()
        
        # Step 3: Feature engineering
        self.feature_engineering()
        
        # Step 4: Prepare features
        self.prepare_features()
        
        # Step 5: Train models
        self.train_models()
        
        # Step 6: Evaluate models
        best_model_name = self.evaluate_models()
        
        # Step 7: Feature importance
        self.feature_importance_analysis()
        
        # Step 8: Save best model
        self.save_best_model(best_model_name)
        
        # Step 9: Generate insights
        self.generate_insights()
        
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\nGenerated Files:")
        print("1. eda_visualizations.png - Exploratory data analysis charts")
        print("2. correlation_heatmap.png - Feature correlation heatmap")
        print("3. model_comparison.csv - Model performance metrics")
        print("4. model_comparison.png - Model comparison charts")
        print("5. predictions_vs_actual.png - Best model predictions visualization")
        print("6. feature_importance.png - Feature importance analysis")
        print("7. best_sales_model.pkl - Trained model (ready for deployment)")
        print("8. scaler.pkl - Feature scaler")
        print("9. feature_names.pkl - Feature names for prediction")
        print("10. model_info.txt - Model details and metadata")
        print("11. business_insights.txt - Key business insights")


def main():
    """Main execution function"""
    
    # Initialize pipeline
    pipeline = SalesPredictionPipeline(
        sales_path='/mnt/user-data/uploads/sales_fact.csv',
        products_path='/mnt/user-data/uploads/products_master.csv',
        inventory_path='/mnt/user-data/uploads/inventory_snapshot.csv',
        suppliers_path='/mnt/user-data/uploads/suppliers_master.csv'
    )
    
    # Run the complete pipeline
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()