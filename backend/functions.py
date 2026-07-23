def MAP(systolic_bp, diastolic_bp):
    return float(diastolic_bp) + (1/3) * (float(systolic_bp) - float(diastolic_bp))

def RPP(systolic_bp, heart_rate):
    return float(systolic_bp) * float(heart_rate)

def PP(systolic_bp, diastolic_bp):
    return float(systolic_bp) - float(diastolic_bp)

def UnhealthyLifeScore(smoking_status, alcohol_consumption, physical_activity):
    score = 0
    if float(smoking_status) == 1:
        score += 1
    if float(alcohol_consumption) == 1:
        score += 1
    if float(physical_activity) == 0:
        score += 1
    elif float(physical_activity) == 1:
        score += 0.5
    elif float(physical_activity) == 2:
        score += 0
    return score

def AtherogenicIndexCoefficient(cholesterol_mg_dl, systolic_bp):
    return float(cholesterol_mg_dl) * float(systolic_bp)

def SmokingHypertensionInteraction(smoking_status, systolic_bp):
    if float(smoking_status) == 1:
        return float(systolic_bp) * 1.2
    return float(systolic_bp)

def CardiacAdiposityProxy(bmi, heart_rate):
    return float(bmi) * float(heart_rate)

def CardiovascularStressIndex(Map, heart_rate):
    return float(Map) * float(heart_rate)
def get_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value 
        except ValueError:
            print("Invalid input. Please enter a valid number.")