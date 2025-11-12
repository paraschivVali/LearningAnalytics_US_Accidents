# ============================================================
# PROIECT: Learning Analytics – Analiza factorilor care influențează
# severitatea accidentelor rutiere în SUA (2016–2023)
# ============================================================

import os
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType

# ============================================================
# ETAPA 1: INIȚIALIZARE SPARK
# ============================================================

os.environ["PYSPARK_PYTHON"] = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
os.environ["PYSPARK_DRIVER_PYTHON"] = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")

spark = (
    SparkSession.builder
    .appName("LearningAnalytics_US_Accidents")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print("✅ Spark pornit corect!")

# ============================================================
# ETAPA 2: ÎNCĂRCARE ȘI INSPECȚIE DATE
# ============================================================

data_path = os.path.join("data", "US_Accidents_March23.csv")
df = spark.read.csv(data_path, header=True, inferSchema=True)
print(f"Număr înregistrări: {df.count():,} | Coloane: {len(df.columns)}")

# Afișăm primele rânduri
df.show(5)

# ============================================================
# ETAPA 3: CURĂȚARE ȘI SELECTARE VARIABILE RELEVANTE
# ============================================================

# Selectăm variabilele utile pentru analiză
# Gravitatea accidentului este coloana 'Severity'
columns_of_interest = [
    "Severity", "Temperature(F)", "Humidity(%)", "Visibility(mi)",
    "Wind_Speed(mph)", "Precipitation(in)", "Sunrise_Sunset"
]

df = df.select([c for c in columns_of_interest if c in df.columns])

# Eliminăm valorile lipsă
df = df.dropna()

# Conversii la tip numeric
for col in ["Temperature(F)", "Humidity(%)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]:
    df = df.withColumn(col, F.col(col).cast(DoubleType()))

# Codificăm variabila 'Sunrise_Sunset' (Day/Night)
indexer = StringIndexer(inputCol="Sunrise_Sunset", outputCol="Sunrise_Sunset_idx")
df = indexer.fit(df).transform(df)

# ============================================================
# ETAPA 4: VECTOR ASSEMBLER – PREGĂTIRE PENTRU MODELARE
# ============================================================

assembler = VectorAssembler(
    inputCols=["Temperature(F)", "Humidity(%)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "Sunrise_Sunset_idx"],
    outputCol="features"
)
df = assembler.transform(df).select("features", F.col("Severity").alias("label"))

# Împărțim în set de antrenare și testare
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train_data.count():,} | Test: {test_data.count():,}")

# ============================================================
# ETAPA 5: MODELARE – APLICAREA CLASIFICATORILOR
# ============================================================

evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

# 🔹 Model 1: Logistic Regression
print("\n🚦 Model 1: Logistic Regression")
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=30)
lr_model = lr.fit(train_data)
lr_pred = lr_model.transform(test_data)
acc_lr = evaluator_acc.evaluate(lr_pred)
f1_lr = evaluator_f1.evaluate(lr_pred)

# 🔹 Model 2: Decision Tree
print("\n🌳 Model 2: Decision Tree Classifier")
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=10)
dt_model = dt.fit(train_data)
dt_pred = dt_model.transform(test_data)
acc_dt = evaluator_acc.evaluate(dt_pred)
f1_dt = evaluator_f1.evaluate(dt_pred)

# 🔹 Model 3: Random Forest
print("\n🌲 Model 3: Random Forest Classifier")
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50, maxDepth=10, seed=42)
rf_model = rf.fit(train_data)
rf_pred = rf_model.transform(test_data)
acc_rf = evaluator_acc.evaluate(rf_pred)
f1_rf = evaluator_f1.evaluate(rf_pred)

# ============================================================
# ETAPA 6: REZUMAT ȘI VIZUALIZARE
# ============================================================

print("\n=== Rezultate comparative ===")
print(f"{'Model':25s} | {'Accuratețe':>10s} | {'F1-Score':>10s}")
print("-" * 50)
print(f"{'Logistic Regression':25s} | {acc_lr:10.4f} | {f1_lr:10.4f}")
print(f"{'Decision Tree':25s} | {acc_dt:10.4f} | {f1_dt:10.4f}")
print(f"{'Random Forest':25s} | {acc_rf:10.4f} | {f1_rf:10.4f}")

# Salvăm graficele
os.makedirs("results", exist_ok=True)

# Grafic comparativ pentru acuratețe
plt.figure(figsize=(8,5))
plt.bar(["Logistic Regression", "Decision Tree", "Random Forest"], [acc_lr, acc_dt, acc_rf], color=["#1f77b4","#2ca02c","#ff7f0e"])
plt.title("Compararea acurateței între modele")
plt.ylabel("Accuratețe")
plt.tight_layout()
plt.savefig("results/comparatie_accuratețe.png")
plt.close()

# Grafic comparativ pentru F1-Score
plt.figure(figsize=(8,5))
plt.bar(["Logistic Regression", "Decision Tree", "Random Forest"], [f1_lr, f1_dt, f1_rf], color=["#1f77b4","#2ca02c","#ff7f0e"])
plt.title("Compararea scorului F1 între modele")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.savefig("results/comparatie_f1.png")
plt.close()

print("\n✅ Graficele comparative au fost salvate în folderul 'results/'.")

spark.stop()
