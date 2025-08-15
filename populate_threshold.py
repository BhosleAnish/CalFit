import sqlite3
import pandas as pd
data = [
    ("Sugar", "g", 25.0, None, "macronutrient",
     "High sugar intake can increase the risk of diabetes and obesity.",
     "Low sugar intake is generally not a concern unless in extreme calorie deficits."),
    
    ("Salt", "g", 5.0, None, "mineral",
     "Too much salt can raise blood pressure and lead to heart disease.",
     "Low salt intake may cause electrolyte imbalance."),

    ("Cholesterol", "mg", 300.0, None, "other",
     "High cholesterol can clog arteries and lead to cardiovascular diseases.",
     "Low cholesterol is generally beneficial but extremely low levels may impact hormone production."),

    ("Protein", "g", None, 50.0, "macronutrient",
     "Excess protein may strain kidneys if sustained long-term.",
     "Low protein levels may result in fatigue or muscle loss."),

    ("Carbohydrates", "g", None, 130.0, "macronutrient",
     "Excess carbs may lead to weight gain and insulin resistance.",
     "Low carbohydrate intake may cause low energy and brain fog."),

    ("Fat", "g", 70.0, None, "macronutrient",
     "Excess fat can contribute to heart problems and weight gain.",
     "Too little fat may lead to hormone imbalances and poor nutrient absorption."),

    ("Fiber", "g", None, 25.0, "macronutrient",
     "Too much fiber can cause bloating or digestive discomfort.",
     "Low fiber intake can cause constipation and affect digestion."),

    ("Calcium", "mg", None, 1000.0, "mineral",
     "Excess calcium may lead to kidney stones or heart issues.",
     "Low calcium increases risk of osteoporosis and weak bones."),

    ("Iron", "mg", None, 18.0, "mineral",
     "High iron levels can damage organs over time.",
     "Iron deficiency can lead to anemia and fatigue."),

    ("Vitamin C", "mg", None, 90.0, "vitamin",
     "Too much vitamin C can cause kidney stones or stomach upset.",
     "Low vitamin C can cause fatigue, joint pain, or even scurvy."),
     ("Saturated Fat", "g", 20.0, None, "fat",
     "Too much saturated fat can raise bad cholesterol levels and increase heart disease risk.",
     "Very low saturated fat is generally beneficial but should not eliminate all fat intake."),

    ("Trans Fat", "g", 2.0, None, "fat",
     "Trans fats increase bad cholesterol and lower good cholesterol, raising heart disease risk.",
     "Low trans fat is ideal for heart health."),

    ("Added Sugar", "g", 25.0, None, "carbohydrate",
     "High intake of added sugars increases risk of diabetes, obesity, and fatty liver.",
     "Low added sugar intake is preferred and beneficial."),

    ("Fructose", "g", 50.0, None, "carbohydrate",
     "Excess fructose can contribute to fatty liver and insulin resistance.",
     "Very low fructose intake is generally fine unless under metabolic therapy."),

    ("High Fructose Corn Syrup", "g", 25.0, None, "carbohydrate",
     "High HFCS intake is linked to obesity and metabolic disorders.",
     "Low HFCS intake is desirable."),

    ("Preservatives (e.g., Sodium Benzoate)", "mg", 1000.0, None, "additive",
     "High preservative intake may cause allergic reactions or hyperactivity in children.",
     "Low preservative intake is generally safer."),

    ("Artificial Colors", "mg", 200.0, None, "additive",
     "Excess artificial colors may cause hyperactivity or allergic responses.",
     "Minimal artificial color use is ideal."),

    ("Artificial Sweeteners", "mg", 100.0, None, "additive",
     "Overuse may disrupt gut health or cause headaches in sensitive individuals.",
     "Low intake is fine unless managing diabetes with alternatives."),

    ("Sodium Nitrate", "mg", 3.7, None, "preservative",
     "High levels are associated with increased cancer risk.",
     "Low nitrate levels are safer."),

    ("MSG (Monosodium Glutamate)", "mg", 3000.0, None, "flavor enhancer",
     "Excess MSG may cause headaches, nausea, or allergic reactions in sensitive people.",
     "Low MSG intake is preferred."),

    ("Sodium", "mg", 2300.0, None, "mineral",
     "High sodium raises blood pressure and increases stroke risk.",
     "Low sodium can cause muscle cramps and fatigue."),

    ("Phosphates", "mg", 700.0, None, "mineral",
     "Too much phosphate can affect kidney health and bone strength.",
     "Low phosphate intake is typically not a concern."),

    ("Potassium", "mg", None, 4700.0, "mineral",
     "Excess potassium can affect heart rhythm (especially in kidney disease).",
     "Low potassium can lead to muscle weakness or cramps."),

    ("Zinc", "mg", None, 11.0, "mineral",
     "Too much zinc can suppress immune function and cause nausea.",
     "Zinc deficiency can impair immune function and healing."),

    ("Vitamin A", "mcg", 3000.0, 700.0, "vitamin",
     "Excess vitamin A can lead to liver damage and birth defects.",
     "Low vitamin A can cause night blindness and skin issues."),

    ("Vitamin D", "mcg", 100.0, 15.0, "vitamin",
     "Too much vitamin D can cause calcium buildup and kidney damage.",
     "Low vitamin D may lead to bone weakness and fatigue."),

    ("Vitamin B12", "mcg", None, 2.4, "vitamin",
     "Very high intake rarely causes harm but can mask other deficiencies.",
     "Low B12 causes fatigue, nerve issues, and anemia."),

    ("Niacin (B3)", "mg", 35.0, 14.0, "vitamin",
     "Excess niacin may cause flushing, liver damage, or ulcers.",
     "Low niacin can cause fatigue and skin issues."),

    ("Thiamine (B1)", "mg", None, 1.1, "vitamin",
     "Very high levels are generally excreted.",
     "Low thiamine may lead to nerve and heart problems."),

    ("Folic Acid", "mcg", 1000.0, 400.0, "vitamin",
     "High folic acid may hide B12 deficiency.",
     "Low folate causes anemia and is dangerous during pregnancy."),
    ("Aspartame", "mg", 50.0, None, "sweetener",
     "High aspartame intake may cause headaches or neurological symptoms in sensitive individuals.",
     "Low aspartame intake is generally safe according to FDA."),

    ("Acesulfame Potassium", "mg", 15.0, None, "sweetener",
     "Excessive intake may disrupt gut bacteria or cause insulin spikes.",
     "Low intake is considered safe by most food safety agencies."),

    ("Sucralose", "mg", 5.0, None, "sweetener",
     "High sucralose intake may affect insulin response or gut health.",
     "Low intake is generally considered safe."),

    ("Sorbitol", "mg", 50.0, None, "sugar alcohol",
     "Too much sorbitol can cause bloating and diarrhea.",
     "Low intake is unlikely to cause side effects."),

    ("Xylitol", "mg", 40.0, None, "sugar alcohol",
     "High intake may lead to digestive discomfort.",
     "Low intake is typically well tolerated."),

    ("Carrageenan", "mg", 100.0, None, "additive",
     "Large amounts may cause inflammation or digestive issues.",
     "Low intake is likely safe for most people."),

    ("Guar Gum", "mg", 500.0, None, "thickener",
     "High levels may lead to gas and bloating.",
     "Low levels are used to improve texture and are generally safe."),

    ("Xanthan Gum", "mg", 500.0, None, "thickener",
     "Excess xanthan gum may lead to laxative effects.",
     "Small amounts are safe and common in processed foods."),

    ("Lecithin", "mg", 1000.0, None, "emulsifier",
     "Very high intake may cause gastrointestinal upset.",
     "Low levels are common in processed foods and generally safe."),

    ("BHA (Butylated Hydroxyanisole)", "mg", 3.0, None, "preservative",
     "High levels are potentially carcinogenic in animal studies.",
     "Low intake is considered safe by regulatory agencies."),

    ("BHT (Butylated Hydroxytoluene)", "mg", 3.0, None, "preservative",
     "Excess BHT may cause liver or kidney damage in animals.",
     "Small amounts are approved for food use."),

    ("Propyl Gallate", "mg", 0.1, None, "preservative",
     "Large doses may disrupt hormones.",
     "Low doses in food are considered safe but controversial."),

    ("Tocopherols (Vitamin E additive)", "mg", 1000.0, 15.0, "antioxidant",
     "Excessive supplemental intake may increase bleeding risk.",
     "Low intake helps prevent oxidation of fats in food."),

    ("Sodium Caseinate", "mg", 2000.0, None, "dairy protein",
     "May cause allergic reaction in dairy-sensitive individuals.",
     "Low intake is typically safe in non-allergic individuals."),

    ("Maltodextrin", "g", 10.0, None, "carbohydrate",
     "High intake may spike blood sugar rapidly.",
     "Low intake is generally well tolerated."),

    ("Polydextrose", "g", 15.0, None, "fiber additive",
     "High intake can cause gas and bloating.",
     "Low amounts may aid digestion as fiber source."),

    ("Citric Acid", "mg", 1000.0, None, "preservative",
     "Excess may erode tooth enamel or irritate the stomach.",
     "Small amounts are commonly used and safe."),

    ("Sodium Citrate", "mg", 800.0, None, "buffering agent",
     "Too much may cause digestive upset.",
     "Low levels are used to regulate acidity in foods."),

    ("Silicon Dioxide", "mg", 1500.0, None, "anti-caking agent",
     "Generally regarded as safe in moderate amounts.",
     "Helps prevent clumping in powdered foods."),

    ("Calcium Propionate", "mg", 500.0, None, "preservative",
     "High doses may cause migraines or hyperactivity in sensitive individuals.",
     "Low doses extend shelf life in baked goods safely."),
]

conn = sqlite3.connect('food_thresholds.db')
cursor = conn.cursor()

# 🛠️ Step 1: Drop the old table (optional if you're rebuilding)
cursor.execute("DROP TABLE IF EXISTS ingredient_thresholds")

# 🛠️ Step 2: Create new table with risk message fields
cursor.execute("""
CREATE TABLE IF NOT EXISTS ingredient_thresholds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    unit TEXT NOT NULL,
    max_threshold REAL,
    min_threshold REAL,
    category TEXT,
    high_risk_message TEXT,
    low_risk_message TEXT
)
""")

# 🛠️ Step 3: Insert the data with messages
cursor.executemany("""
INSERT OR IGNORE INTO ingredient_thresholds (
    name, unit, max_threshold, min_threshold, category, high_risk_message, low_risk_message
) VALUES (?, ?, ?, ?, ?, ?, ?)
""", data)
conn.commit()
conn.close()

print("✅ Database with risk messages created and populated.")

conn = sqlite3.connect('food_thresholds.db')

df = pd.read_sql_query("SELECT * FROM ingredient_thresholds", conn)
print(df)

conn.close()