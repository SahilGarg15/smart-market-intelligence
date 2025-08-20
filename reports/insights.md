# Insights — 02_data_cleaning_analysis

**What changed (data enrichment & cleaning)**
- Added `revenue_per_unit = sales / quantity`.
- Added `month` column as `YYYY-MM` string from `date`.
- Dropped rows where `quantity <= 0` or `sales <= 0`.
- Converted `date` column to datetime and ensured numeric types for `quantity` and `sales`.

**Key computed results**
- Category with highest average `revenue_per_unit`: **Electronics**
- Category with highest volatility (std dev of monthly sales): **Electronics**
- Top 5 products by total sales:
product
Smartwatch    1259788.51
Charger       1109828.50
Headphones     912604.99
Laptop         879658.27
Smartphone     373819.82

- Bottom 5 (non-zero) products by total sales:
product
Pasta    30525.14
Rice     26847.72
Bread    25631.42
Eggs     21333.80
Milk     19759.91

**4–6 Bullet Insights / Observations**
- The category **Electronics** has the highest average revenue per unit, indicating higher-priced items or premium products are concentrated there.
- **Electronics** shows the most month-to-month sales variability; this category might be seasonal or promo-driven.
- Top products contribute a significant portion of total revenue — focus on inventory and pricing strategies for these.
- Some products have very low sales (bottom-5); investigate if they are slow-moving or mis-categorized.
- Revenue per unit distribution varies notably by category (see boxplot), suggesting different pricing strategies per category.
- Consider further analysis: cohort sales, price elasticity, and promotional impact per category.

**Files produced**
- `data/processed/transactions_cleaned.csv`
- `reports/visuals/boxplot_revenue_per_unit_by_category.png`
- `reports/visuals/bar_top10_products_total_sales.png`
