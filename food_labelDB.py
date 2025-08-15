import pandas as pd
import sqlite3

# Load CSV
csv_file = "food_label.csv"
df = pd.read_csv(csv_file)

# Create SQLite DB
db_file = "food_labelDB.db"
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Drop old table if exists
cursor.execute("DROP TABLE IF EXISTS ingredient_thresholds")

# Create table with your column names
cursor.execute("""
CREATE TABLE ingredient_thresholds (
    Nutrient TEXT PRIMARY KEY,
    Min_Val REAL,
    Max_Val REAL,
    Min_Val_Risks TEXT,
    Max_Val_Risks TEXT
)
""")

# Insert rows
for _, row in df.iterrows():
    cursor.execute("""
INSERT OR IGNORE INTO ingredient_thresholds (Nutrient, Min_Val, Max_Val, Min_Val_Risks, Max_Val_Risks)
VALUES (?, ?, ?, ?, ?)
""", (
        row["Nutrient"],
        row["Min_Val"],
        row["Max_Val"],
        row["Min_Val_Risks"],
        row["Max_Val_Risks"]
    ))

conn.commit()

# Optional: print the whole DB to check
cursor.execute("SELECT * FROM ingredient_thresholds")
rows = cursor.fetchall()
for r in rows:
    print(r)

conn.close()

print(f"Database '{db_file}' created successfully with {len(df)} records.")
