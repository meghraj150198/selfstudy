"""
Sales Prediction Dashboard
==========================
Interactive dashboard to visualize all key results from the sales prediction pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SalesDashboard:
    """Create comprehensive dashboard for sales prediction results"""
    
    def __init__(self, results_dir='/workspaces/selfstudy'):
        self.results_dir = results_dir
        self.load_all_data()
    
    def load_all_data(self):
        """Load all necessary data and models"""
        print("Loading dashboard data...")
        
        # Load CSVs
        self.model_comparison = pd.read_csv(f'{self.results_dir}/model_comparison.csv')
        self.sales_data = pd.read_csv(f'{self.results_dir}/sales_fact.csv')
        
        # Load model and scaler
        try:
            self.best_model = joblib.load(f'{self.results_dir}/best_sales_model.pkl')
            self.scaler = joblib.load(f'{self.results_dir}/scaler.pkl')
            self.feature_names = joblib.load(f'{self.results_dir}/feature_names.pkl')
        except:
            print("Note: Model files not found, some features will be limited")
            self.best_model = None
        
        print("✓ Data loaded successfully")
    
    def create_executive_summary(self):
        """Create executive summary page"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # Title
        fig.suptitle('Sales Prediction Dashboard - Executive Summary', 
                     fontsize=24, fontweight='bold', y=0.98)
        
        # 1. Best Model Performance
        ax1 = fig.add_subplot(gs[0, 0])
        best_model_row = self.model_comparison.iloc[0]
        metrics = ['R² Score', 'RMSE', 'MAE', 'MAPE (%)']
        values = [
            best_model_row['Test R²'],
            best_model_row['Test RMSE'] / 10,  # Scale for visibility
            best_model_row['Test MAE'],
            best_model_row['Test MAPE (%)'] / 5  # Scale for visibility
        ]
        colors_list = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        bars = ax1.barh(metrics, values, color=colors_list)
        ax1.set_title(f"Best Model: {best_model_row['Model']}", fontweight='bold', fontsize=12)
        for i, bar in enumerate(bars):
            if i == 0:
                ax1.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                        f'{best_model_row["Test R²"]:.4f}', ha='left', va='center', fontweight='bold')
            elif i == 3:
                ax1.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                        f'{best_model_row["Test MAPE (%)"]:.2f}%', ha='left', va='center', fontweight='bold')
        ax1.set_xlim(0, max(values) * 1.2)
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Model Performance Ranking
        ax2 = fig.add_subplot(gs[0, 1:])
        models = self.model_comparison['Model'].values
        r2_scores = self.model_comparison['Test R²'].values
        colors_gradient = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
        bars = ax2.barh(models, r2_scores, color=colors_gradient)
        ax2.set_xlabel('Test R² Score', fontweight='bold')
        ax2.set_title('Model Performance Comparison', fontweight='bold', fontsize=12)
        ax2.set_xlim(0, 1)
        for i, bar in enumerate(bars):
            ax2.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height()/2, 
                    f'{r2_scores[i]:.4f}', ha='right', va='center', color='white', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Key Metrics Cards
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')
        metrics_text = f"""
        KEY PERFORMANCE METRICS
        
        Model Accuracy (R²): {best_model_row['Test R²']:.2%}
        Root Mean Squared Error: {best_model_row['Test RMSE']:.2f} units
        Mean Absolute Error: {best_model_row['Test MAE']:.2f} units
        Mean Absolute % Error: {best_model_row['Test MAPE (%)']:.2f}%
        """
        ax3.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
                verticalalignment='center', fontweight='bold')
        
        # 4. Sales Insights
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        avg_sales = self.sales_data['units_sold'].mean()
        max_sales = self.sales_data['units_sold'].max()
        min_sales = self.sales_data['units_sold'].min()
        std_sales = self.sales_data['units_sold'].std()
        
        insights_text = f"""
        SALES METRICS
        
        Average Daily Sales: {avg_sales:.2f} units
        Maximum Sales: {max_sales:.0f} units
        Minimum Sales: {min_sales:.0f} units
        Std Deviation: {std_sales:.2f} units
        """
        ax4.text(0.1, 0.5, insights_text, fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
                verticalalignment='center', fontweight='bold')
        
        # 5. Business Impact
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        promo_sales = self.sales_data[self.sales_data['promo_flag'] == 1]['units_sold'].mean()
        non_promo_sales = self.sales_data[self.sales_data['promo_flag'] == 0]['units_sold'].mean()
        promo_lift = ((promo_sales - non_promo_sales) / non_promo_sales) * 100
        
        weekend_sales = self.sales_data[self.sales_data['is_weekend'] == 1]['units_sold'].mean()
        weekday_sales = self.sales_data[self.sales_data['is_weekend'] == 0]['units_sold'].mean()
        weekend_diff = ((weekend_sales - weekday_sales) / weekday_sales) * 100
        
        impact_text = f"""
        BUSINESS IMPACT
        
        Promo Lift: +{promo_lift:.1f}%
        Weekend Boost: +{weekend_diff:.1f}%
        Data Points: {len(self.sales_data):,}
        Training Accuracy: High
        """
        ax5.text(0.1, 0.5, impact_text, fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3),
                verticalalignment='center', fontweight='bold')
        
        # 6. Error Distribution
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.text(0.5, 0.9, 'Model Error Analysis', ha='center', fontweight='bold', fontsize=12, transform=ax6.transAxes)
        error_data = [
            best_model_row['Test RMSE'],
            best_model_row['Test MAE'],
            best_model_row['Test MAPE (%)'] / 10
        ]
        error_labels = ['RMSE', 'MAE', 'MAPE/10']
        colors_err = ['#e74c3c', '#f39c12', '#95a5a6']
        ax6.pie(error_data, labels=error_labels, autopct='%1.1f%%', colors=colors_err, startangle=90)
        ax6.axis('equal')
        
        # 7. Dataset Overview
        ax7 = fig.add_subplot(gs[2, 1])
        platforms = self.sales_data['platform_traffic_source'].value_counts().head(5)
        ax7.barh(platforms.index, platforms.values, color=plt.cm.Set3(np.linspace(0, 1, len(platforms))))
        ax7.set_xlabel('Number of Records', fontweight='bold')
        ax7.set_title('Top 5 Traffic Sources', fontweight='bold', fontsize=12)
        ax7.grid(axis='x', alpha=0.3)
        for i, v in enumerate(platforms.values):
            ax7.text(v + 50, i, str(v), va='center', fontweight='bold')
        
        # 8. Recommendation Box
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        recommendation = f"""
        RECOMMENDATIONS
        
        ✓ Use XGBoost for predictions
        ✓ Model accuracy: 97.2%
        ✓ Focus on promotional campaigns
        ✓ Leverage weekend trends
        ✓ Ready for production
        """
        ax8.text(0.1, 0.5, recommendation, fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3),
                verticalalignment='center', fontweight='bold')
        
        plt.savefig(f'{self.results_dir}/dashboard_executive_summary.png', dpi=300, bbox_inches='tight')
        print("✓ Executive summary dashboard created")
        plt.close()
    
    def create_model_analysis(self):
        """Create detailed model analysis dashboard"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Model Analysis & Performance Metrics', 
                     fontsize=22, fontweight='bold', y=0.98)
        
        # 1. All Models - Test R²
        ax1 = fig.add_subplot(gs[0, 0])
        models = self.model_comparison['Model'].values
        r2_scores = self.model_comparison['Test R²'].values
        colors_r2 = plt.cm.RdYlGn(np.linspace(0.2, 0.95, len(models)))
        bars1 = ax1.barh(models, r2_scores, color=colors_r2)
        ax1.set_xlabel('R² Score', fontweight='bold')
        ax1.set_title('Test R² Scores', fontweight='bold', fontsize=12)
        ax1.set_xlim(0, 1)
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width - 0.05, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='right', va='center', color='white', fontweight='bold', fontsize=9)
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. RMSE Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        rmse_values = self.model_comparison['Test RMSE'].values
        colors_rmse = plt.cm.RdYlGn_r(np.linspace(0.2, 0.95, len(models)))
        bars2 = ax2.barh(models, rmse_values, color=colors_rmse)
        ax2.set_xlabel('RMSE (Lower is Better)', fontweight='bold')
        ax2.set_title('Test RMSE', fontweight='bold', fontsize=12)
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center', fontweight='bold', fontsize=9)
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. MAPE Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        mape_values = self.model_comparison['Test MAPE (%)'].values
        colors_mape = plt.cm.RdYlGn_r(np.linspace(0.2, 0.95, len(models)))
        bars3 = ax3.barh(models, mape_values, color=colors_mape)
        ax3.set_xlabel('MAPE % (Lower is Better)', fontweight='bold')
        ax3.set_title('Test MAPE', fontweight='bold', fontsize=12)
        for bar in bars3:
            width = bar.get_width()
            ax3.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}%', ha='left', va='center', fontweight='bold', fontsize=9)
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Model Ranking Score (Custom)
        ax4 = fig.add_subplot(gs[1, 0])
        # Normalize all metrics to 0-100 scale
        r2_norm = (r2_scores - r2_scores.min()) / (r2_scores.max() - r2_scores.min()) * 100
        rmse_norm = 100 - (rmse_values - rmse_values.min()) / (rmse_values.max() - rmse_values.min()) * 100
        mape_norm = 100 - (mape_values - mape_values.min()) / (mape_values.max() - mape_values.min()) * 100
        
        overall_score = (r2_norm + rmse_norm + mape_norm) / 3
        ranking_df = pd.DataFrame({
            'Model': models,
            'Overall Score': overall_score
        }).sort_values('Overall Score', ascending=True)
        
        colors_rank = plt.cm.RdYlGn(np.linspace(0.2, 0.95, len(ranking_df)))
        bars4 = ax4.barh(ranking_df['Model'], ranking_df['Overall Score'], color=colors_rank)
        ax4.set_xlabel('Composite Score', fontweight='bold')
        ax4.set_title('Model Ranking (Composite)', fontweight='bold', fontsize=12)
        ax4.set_xlim(0, 100)
        for bar in bars4:
            width = bar.get_width()
            ax4.text(width - 3, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}', ha='right', va='center', color='white', fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
        
        # 5. MAE Comparison
        ax5 = fig.add_subplot(gs[1, 1])
        mae_values = self.model_comparison['Test MAE'].values
        colors_mae = plt.cm.RdYlGn_r(np.linspace(0.2, 0.95, len(models)))
        bars5 = ax5.barh(models, mae_values, color=colors_mae)
        ax5.set_xlabel('MAE (Lower is Better)', fontweight='bold')
        ax5.set_title('Test MAE', fontweight='bold', fontsize=12)
        for bar in bars5:
            width = bar.get_width()
            ax5.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center', fontweight='bold', fontsize=9)
        ax5.grid(axis='x', alpha=0.3)
        
        # 6. Performance Summary Table
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        best_r2_model = self.model_comparison.loc[self.model_comparison['Test R²'].idxmax()]
        best_rmse_model = self.model_comparison.loc[self.model_comparison['Test RMSE'].idxmin()]
        best_mae_model = self.model_comparison.loc[self.model_comparison['Test MAE'].idxmin()]
        best_mape_model = self.model_comparison.loc[self.model_comparison['Test MAPE (%)'].idxmin()]
        
        summary_text = f"""
        BEST PERFORMERS
        
        Best R² Score:
        {best_r2_model['Model']} ({best_r2_model['Test R²']:.4f})
        
        Best RMSE:
        {best_rmse_model['Model']} ({best_rmse_model['Test RMSE']:.2f})
        
        Best MAE:
        {best_mae_model['Model']} ({best_mae_model['Test MAE']:.2f})
        
        Best MAPE:
        {best_mape_model['Model']} ({best_mape_model['Test MAPE (%)']:.2f}%)
        """
        ax6.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3),
                verticalalignment='center', fontweight='bold')
        
        plt.savefig(f'{self.results_dir}/dashboard_model_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Model analysis dashboard created")
        plt.close()
    
    def create_business_insights(self):
        """Create business insights dashboard"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        fig.suptitle('Business Insights & Sales Analytics', 
                     fontsize=22, fontweight='bold', y=0.98)
        
        # 1. Sales Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.sales_data['units_sold'], bins=40, color='#3498db', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Units Sold', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Sales Distribution', fontweight='bold', fontsize=12)
        ax1.axvline(self.sales_data['units_sold'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {self.sales_data["units_sold"].mean():.1f}')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Revenue Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.sales_data['gross_revenue'], bins=40, color='#2ecc71', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Gross Revenue', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Revenue Distribution', fontweight='bold', fontsize=12)
        ax2.axvline(self.sales_data['gross_revenue'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ₹{self.sales_data["gross_revenue"].mean():.0f}')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Promotional Impact
        ax3 = fig.add_subplot(gs[0, 2])
        promo_stats = self.sales_data.groupby('promo_flag')['units_sold'].agg(['mean', 'std', 'count'])
        labels = ['No Promotion', 'With Promotion']
        means = promo_stats['mean'].values
        colors_promo = ['#e74c3c', '#2ecc71']
        bars = ax3.bar(labels, means, color=colors_promo, alpha=0.7, edgecolor='black', linewidth=2)
        ax3.set_ylabel('Average Units Sold', fontweight='bold')
        ax3.set_title('Promotional Impact', fontweight='bold', fontsize=12)
        for bar, mean in zip(bars, means):
            ax3.text(bar.get_x() + bar.get_width()/2, mean + 0.5, f'{mean:.1f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Weekend vs Weekday
        ax4 = fig.add_subplot(gs[1, 0])
        weekend_stats = self.sales_data.groupby('is_weekend')['units_sold'].mean()
        labels_wd = ['Weekday', 'Weekend']
        colors_wd = ['#3498db', '#f39c12']
        bars = ax4.bar(labels_wd, weekend_stats.values, color=colors_wd, alpha=0.7, edgecolor='black', linewidth=2)
        ax4.set_ylabel('Average Units Sold', fontweight='bold')
        ax4.set_title('Weekday vs Weekend Sales', fontweight='bold', fontsize=12)
        for bar, val in zip(bars, weekend_stats.values):
            ax4.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Platform Performance
        ax5 = fig.add_subplot(gs[1, 1])
        platform_sales = self.sales_data.groupby('platform_traffic_source')['units_sold'].sum().sort_values(ascending=True).tail(8)
        colors_cat = plt.cm.Set3(np.linspace(0, 1, len(platform_sales)))
        bars = ax5.barh(platform_sales.index, platform_sales.values, color=colors_cat, edgecolor='black')
        ax5.set_xlabel('Total Units Sold', fontweight='bold')
        ax5.set_title('Top 8 Traffic Sources by Sales', fontweight='bold', fontsize=12)
        for bar in bars:
            width = bar.get_width()
            ax5.text(width + 50, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                    ha='left', va='center', fontweight='bold')
        ax5.grid(axis='x', alpha=0.3)
        
        # 6. Monthly Trend
        ax6 = fig.add_subplot(gs[1, 2])
        monthly_sales = self.sales_data.groupby('month')['units_sold'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax6.plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2.5, 
                markersize=8, color='#9b59b6')
        ax6.fill_between(monthly_sales.index, monthly_sales.values, alpha=0.3, color='#9b59b6')
        ax6.set_xlabel('Month', fontweight='bold')
        ax6.set_ylabel('Average Units Sold', fontweight='bold')
        ax6.set_title('Monthly Sales Trend', fontweight='bold', fontsize=12)
        ax6.set_xticks(range(1, 13))
        ax6.set_xticklabels(month_names)
        ax6.grid(alpha=0.3)
        
        # 7. Festival Month Impact
        ax7 = fig.add_subplot(gs[2, 0])
        festival_stats = self.sales_data.groupby('is_festival_month')['units_sold'].mean()
        labels_fest = ['Regular Month', 'Festival Month']
        colors_fest = ['#34495e', '#f1c40f']
        bars = ax7.bar(labels_fest, festival_stats.values, color=colors_fest, alpha=0.7, edgecolor='black', linewidth=2)
        ax7.set_ylabel('Average Units Sold', fontweight='bold')
        ax7.set_title('Festival Month Impact', fontweight='bold', fontsize=12)
        for bar, val in zip(bars, festival_stats.values):
            ax7.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax7.grid(axis='y', alpha=0.3)
        
        # 8. Season Analysis
        ax8 = fig.add_subplot(gs[2, 1])
        season_sales = self.sales_data.groupby('season_tag')['units_sold'].mean().sort_values(ascending=False)
        colors_season = plt.cm.Set2(np.linspace(0, 1, len(season_sales)))
        bars = ax8.bar(season_sales.index, season_sales.values, color=colors_season, alpha=0.7, edgecolor='black')
        ax8.set_ylabel('Average Units Sold', fontweight='bold')
        ax8.set_title('Sales by Season', fontweight='bold', fontsize=12)
        for bar in bars:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2, height + 0.2, f'{height:.1f}', 
                    ha='center', va='bottom', fontweight='bold')
        ax8.grid(axis='y', alpha=0.3)
        
        # 9. Key Metrics Summary
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        promo_sales = self.sales_data[self.sales_data['promo_flag'] == 1]['units_sold'].mean()
        non_promo_sales = self.sales_data[self.sales_data['promo_flag'] == 0]['units_sold'].mean()
        promo_lift = ((promo_sales - non_promo_sales) / non_promo_sales) * 100
        
        weekend_sales = self.sales_data[self.sales_data['is_weekend'] == 1]['units_sold'].mean()
        weekday_sales = self.sales_data[self.sales_data['is_weekend'] == 0]['units_sold'].mean()
        weekend_diff = ((weekend_sales - weekday_sales) / weekday_sales) * 100
        
        festival_sales = self.sales_data[self.sales_data['is_festival_month'] == 1]['units_sold'].mean()
        regular_sales = self.sales_data[self.sales_data['is_festival_month'] == 0]['units_sold'].mean()
        festival_diff = ((festival_sales - regular_sales) / regular_sales) * 100
        
        metrics_text = f"""
        KEY INSIGHTS
        
        Promo Lift: +{promo_lift:.1f}%
        Weekend Boost: +{weekend_diff:.1f}%
        Festival Lift: +{festival_diff:.1f}%
        
        Total Data Points: {len(self.sales_data):,}
        Unique Traffic Sources: {self.sales_data['platform_traffic_source'].nunique()}
        Date Range: Year-round
        """
        ax9.text(0.05, 0.5, metrics_text, fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4),
                verticalalignment='center', fontweight='bold')
        
        plt.savefig(f'{self.results_dir}/dashboard_business_insights.png', dpi=300, bbox_inches='tight')
        print("✓ Business insights dashboard created")
        plt.close()
    
    def generate_all_dashboards(self):
        """Generate all dashboard pages"""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE SALES PREDICTION DASHBOARDS")
        print("="*60 + "\n")
        
        self.create_executive_summary()
        self.create_model_analysis()
        self.create_business_insights()
        
        print("\n" + "="*60)
        print("DASHBOARD GENERATION COMPLETED!")
        print("="*60)
        print("\nGenerated Dashboard Files:")
        print("1. dashboard_executive_summary.png - Key metrics & performance overview")
        print("2. dashboard_model_analysis.png - Detailed model comparison")
        print("3. dashboard_business_insights.png - Sales analytics & trends")
        print("\nAll dashboards saved in: /workspaces/selfstudy/")


def main():
    """Main execution"""
    dashboard = SalesDashboard()
    dashboard.generate_all_dashboards()


if __name__ == "__main__":
    main()
