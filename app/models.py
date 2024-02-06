from pydantic import BaseModel
from typing import List, Optional

class Debt(BaseModel):
    type: str
    total_amount: float
    interest_rate: float
    term: int
    monthly_payment: float

class UserInputSchema(BaseModel):
    age: int
    geographic_location: str
    marital_status: str
    number_of_dependents: int
    occupation: str
    employment_type: str
    monthly_income: float
    job_stability: int
    debts: List[Debt]
    credit_score: Optional[int]
    savings_investments: float
    monthly_fixed_expenses: float
    monthly_variable_expenses: float
    saving_goals_short_long_term: List[float]
    investment_preferences: str
    risk_tolerance: str
    financial_priorities: List[str]
    use_of_financial_management_tools: bool
    payment_history: str
    experiences_with_financial_advising: bool
