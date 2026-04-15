import psycopg2
import pandas as pd
import joblib
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

DB = {
    "host": "localhost",
    "port": 5432,
    "database": "mediciones_calidad",
    "user": "postgres",
    "password": "0513"
}

# =========================
# 1) LÓGICA DIFUSA
# =========================
N  = ctrl.Antecedent(np.linspace(0, 15, 401), 'N_agua_NO3N')
MO = ctrl.Antecedent(np.linspace(0, 6, 401), 'MO_suelo')
DA = ctrl.Antecedent(np.linspace(0.9, 1.9, 401), 'DA_suelo')
CE = ctrl.Antecedent(np.linspace(0, 2000, 401), 'CE_agua')
Impacto = ctrl.Consequent(np.linspace(0, 100, 401), 'Impacto')

MO['baja']  = fuzz.trapmf(MO.universe, [0, 0, 1.10, 1.60])
MO['media'] = fuzz.trimf(MO.universe, [1.50, 2.25, 3.00])
MO['alta']  = fuzz.trapmf(MO.universe, [2.90, 3.40, 6.00, 6.00])

N['bajo']   = fuzz.trapmf(N.universe, [0, 0, 7.0, 10.0])
N['medio']  = fuzz.trimf(N.universe, [8.0, 10.5, 13.0])
N['alto']   = fuzz.trapmf(N.universe, [11.0, 13.0, 15.0, 15.0])

DA['baja']  = fuzz.trapmf(DA.universe, [0.9, 0.9, 1.10, 1.25])
DA['media'] = fuzz.trimf(DA.universe, [1.20, 1.35, 1.50])
DA['alta']  = fuzz.trapmf(DA.universe, [1.55, 1.65, 1.9, 1.9])

CE['baja']  = fuzz.trapmf(CE.universe, [0, 0, 600, 850])
CE['media'] = fuzz.trimf(CE.universe, [800, 1150, 1500])
CE['alta']  = fuzz.trapmf(CE.universe, [1400, 1600, 2000, 2000])

Impacto['bajo']  = fuzz.trapmf(Impacto.universe, [0, 0, 25, 40])
Impacto['medio'] = fuzz.trimf(Impacto.universe, [30, 50, 70])
Impacto['alto']  = fuzz.trapmf(Impacto.universe, [60, 75, 100, 100])

rules = [
    ctrl.Rule(N['alto']  & MO['baja']  & DA['baja'],  Impacto['alto']),
    ctrl.Rule(N['alto']  & MO['baja']  & DA['media'], Impacto['alto']),
    ctrl.Rule(N['alto']  & MO['baja']  & DA['alta'],  Impacto['alto']),
    ctrl.Rule(N['alto']  & MO['media'] & DA['baja'],  Impacto['alto']),
    ctrl.Rule(N['alto']  & MO['media'] & DA['media'], Impacto['alto']),
    ctrl.Rule(N['alto']  & MO['media'] & DA['alta'],  Impacto['medio']),
    ctrl.Rule(N['alto']  & MO['alta']  & DA['baja'],  Impacto['medio']),
    ctrl.Rule(N['alto']  & MO['alta']  & DA['media'], Impacto['medio']),
    ctrl.Rule(N['alto']  & MO['alta']  & DA['alta'],  Impacto['medio']),

    ctrl.Rule(N['medio'] & MO['baja']  & DA['baja'],  Impacto['alto']),
    ctrl.Rule(N['medio'] & MO['baja']  & DA['media'], Impacto['medio']),
    ctrl.Rule(N['medio'] & MO['baja']  & DA['alta'],  Impacto['medio']),
    ctrl.Rule(N['medio'] & MO['media'] & DA['baja'],  Impacto['medio']),
    ctrl.Rule(N['medio'] & MO['media'] & DA['media'], Impacto['medio']),
    ctrl.Rule(N['medio'] & MO['media'] & DA['alta'],  Impacto['bajo']),
    ctrl.Rule(N['medio'] & MO['alta']  & DA['baja'],  Impacto['medio']),
    ctrl.Rule(N['medio'] & MO['alta']  & DA['media'], Impacto['bajo']),
    ctrl.Rule(N['medio'] & MO['alta']  & DA['alta'],  Impacto['bajo']),

    ctrl.Rule(N['bajo']  & MO['baja']  & DA['baja'],  Impacto['medio']),
    ctrl.Rule(N['bajo']  & MO['baja']  & DA['media'], Impacto['medio']),
    ctrl.Rule(N['bajo']  & MO['baja']  & DA['alta'],  Impacto['bajo']),
    ctrl.Rule(N['bajo']  & MO['media'] & DA['baja'],  Impacto['bajo']),
    ctrl.Rule(N['bajo']  & MO['media'] & DA['media'], Impacto['bajo']),
    ctrl.Rule(N['bajo']  & MO['media'] & DA['alta'],  Impacto['bajo']),
    ctrl.Rule(N['bajo']  & MO['alta']  & DA['alta'],  Impacto['bajo']),

    ctrl.Rule(CE['alta'] & N['alto'],                Impacto['alto']),
    ctrl.Rule(CE['alta'] & N['medio'] & MO['baja'],  Impacto['alto']),
    ctrl.Rule(CE['alta'] & N['medio'] & MO['media'], Impacto['medio']),
    ctrl.Rule(CE['alta'] & N['bajo']  & MO['baja'],  Impacto['medio']),

    ctrl.Rule(N['bajo'] | N['medio'] | N['alto'], Impacto['medio'])
]

fuzzy_system = ctrl.ControlSystem(rules)

def fuzzy_impact(n, mo, da, ce):
    sim = ctrl.ControlSystemSimulation(fuzzy_system)
    sim.input['N_agua_NO3N'] = float(n)
    sim.input['MO_suelo'] = float(mo)
    sim.input['DA_suelo'] = float(da)
    sim.input['CE_agua'] = float(ce)
    sim.compute()
    return float(sim.output['Impacto'])

# =========================
# 2) BD
# =========================
def leer_ultimos_parametros():
    with psycopg2.connect(**DB) as conn:
        sql = """
        SELECT DISTINCT ON ("Parametro")
               "Parametro", "Valor", "createdAt"
        FROM medicions
        ORDER BY "Parametro", "createdAt" DESC
        """
        df = pd.read_sql(sql, conn)

    datos = {}
    for _, row in df.iterrows():
        datos[str(row["Parametro"]).strip().lower()] = float(row["Valor"])
    return datos

# =========================
# 3) CAUDAL PARSHALL 3 in
# =========================
def calcular_caudal(nivel_cm):
    ha = max(0.0, 0.61 - (nivel_cm / 100.0))
    q_m3s = 5.56 * (ha ** 1.55) if ha > 0 else 0.0
    q_ls = q_m3s * 1000
    return ha, q_m3s, q_ls

# =========================
# 4) ML NITRATOS
# =========================
poly = joblib.load("poly_nitratos.pkl")
scaler = joblib.load("scaler_nitratos.pkl")
modelo = joblib.load("modelo_xgb_nitratos.pkl")

def predecir_nitratos(temp, ce_bd, ph):
    ce_us = ce_bd * 1000.0   # tu modelo fue entrenado con uS/cm

    X = pd.DataFrame([{
        "Temperatura °C": temp,
        "Conductividad Eléctrica uS/cm": ce_us,
        "pH unidades de pH": ph
    }])

    X_poly = poly.transform(X)
    X_scaled = scaler.transform(X_poly)
    pred = float(modelo.predict(X_scaled)[0])

    return max(pred, 0.0), ce_us

# =========================
# 5) UTILIDADES
# =========================
def clasificar_impacto(valor):
    if valor < 40:
        return "Bajo"
    if valor < 70:
        return "Medio"
    return "Alto"

def pedir_datos_suelo():
    mo = float(input("MO del suelo (%): "))
    da = float(input("DA del suelo (g/cm3): "))
    return mo, da

# =========================
# 6) MAIN
# =========================
def main():
    datos = leer_ultimos_parametros()

    temperatura = datos.get("temperatura")
    conductividad = datos.get("conductividad")
    ph = datos.get("ph")
    nivel = datos.get("nivel")

    faltan = [k for k, v in {
        "temperatura": temperatura,
        "conductividad": conductividad,
        "ph": ph,
        "nivel": nivel
    }.items() if v is None]

    if faltan:
        raise ValueError(f"Faltan datos en la BD: {faltan}")

    ha, q_m3s, q_ls = calcular_caudal(nivel)
    nitratos_pred, ce_us = predecir_nitratos(temperatura, conductividad, ph)

    mo, da = pedir_datos_suelo()
    impacto = fuzzy_impact(
        n=nitratos_pred,
        mo=mo,
        da=da,
        ce=ce_us
    )

    print("\n===== RESULTADOS =====")
    print(f"Temperatura: {temperatura:.2f} °C")
    print(f"pH: {ph:.2f}")
    print(f"Conductividad usada en ML/difuso: {ce_us:.2f} uS/cm")
    print(f"Nivel sensor: {nivel:.2f} cm")
    print(f"Ha: {ha:.4f} m")
    print(f"Caudal: {q_m3s:.6f} m3/s")
    print(f"Caudal: {q_ls:.2f} L/s")
    print(f"Nitratos predichos: {nitratos_pred:.4f} mg N-NO3-/L")
    print(f"Impacto ambiental: {impacto:.2f}/100")
    print(f"Clasificación: {clasificar_impacto(impacto)}")

if __name__ == "__main__":
    main()