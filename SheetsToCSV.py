import pandas as pd

# ID del Sheet
sheet_id = "1nGteO7Ol-FnmPV9iM2EUQxtjua8shcr7NDMR6z6m458"

# Nombre del Sheet
sheet_name = "Proyecto4"

# Construccion de URL para CSV
csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

# Export como CSV
df = pd.read_csv(csv_url)

df.to_csv("data.csv", index=False)

print("Datos guardados en data.csv.")