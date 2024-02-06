import pandas as pd
import numpy as np


def generate_synthetic_data(num_samples=1000):
    np.random.seed(0)
    df = pd.DataFrame({
        'age': np.random.randint(18, 70, size=num_samples),
        'geographic_location': np.random.choice(['Urban', 'Suburban', 'Rural'], size=num_samples),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], size=num_samples),
        'number_of_dependents': np.random.randint(0, 5, size=num_samples),
        'occupation': np.random.choice(['Employed', 'Self-Employed', 'Unemployed', 'Student', 'Retired'], size=num_samples),
        'employment_type': np.random.choice(['Full-Time', 'Part-Time', 'Contract', 'Intern'], size=num_samples),
        'monthly_income': np.random.normal(5000, 2000, size=num_samples).clip(min=0),  # Ingresos con una distribución normal, mínimos de 0
        'job_stability': np.random.randint(0, 30, size=num_samples),  # Años en el empleo actual
        'total_debt': np.random.normal(20000, 15000, size=num_samples).clip(min=0),  # Deuda total con una distribución normal, mínimos de 0
        'credit_score': np.random.randint(300, 850, size=num_samples),  # Puntuación de crédito
        'savings_investments': np.random.normal(10000, 5000, size=num_samples).clip(min=0),  # Ahorros e inversiones
        'monthly_fixed_expenses': np.random.normal(1500, 500, size=num_samples).clip(min=0),  # Gastos fijos mensuales
        'monthly_variable_expenses': np.random.normal(800, 300, size=num_samples).clip(min=0),  # Gastos variables mensuales
        'saving_goals_short_long_term': np.random.normal(20000, 10000, size=num_samples).clip(min=0),  # Objetivos de ahorro a corto/largo plazo
        'investment_preferences': np.random.choice(['Stocks', 'Bonds', 'Real Estate', 'Cryptocurrency', 'None'], size=num_samples),
        'risk_tolerance': np.random.choice(['Low', 'Medium', 'High'], size=num_samples),
        'financial_priorities': np.random.choice(['Save for House', 'Retirement', 'Education', 'Travel', 'Healthcare'], size=num_samples),
        'use_of_financial_management_tools': np.random.choice([True, False], size=num_samples),
        'payment_history': np.random.choice(['Good', 'Average', 'Poor'], size=num_samples),
        'experiences_with_financial_advising': np.random.choice([True, False], size=num_samples),
    })

    df['label'] = np.random.choice(['Increase Savings', 'Reduce Expenses', 'Invest in Stocks'], size=num_samples)

    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv('../data/synthetic_financial_data.csv', index=False)
