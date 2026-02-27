import numpy as np
import pandas as pd

def generate_grouped_data(n_samples=4000, n_customers=1500):
    np.random.seed(42)

    customer_pool = [f"CUST_{i:05d}" for i in range(n_customers)]
    customer_ids = np.random.choice(customer_pool, n_samples)

    income = np.random.normal(3000, 900, n_samples)
    age = np.random.randint(21, 70, n_samples)
    existing_debt = np.random.uniform(0, 8000, n_samples)
    utilization = np.random.uniform(0, 1, n_samples)
    month = np.random.choice(range(1, 13), n_samples)

    # Create a bit realistic risk logic
    logit = (
        -3
        + 0.0004 * existing_debt
        + 2.5 * utilization
        - 0.0003 * income
        + 0.015 * (50 - age)
    )

    pd_true = 1 / (1 + np.exp(-logit))
    target = np.random.binomial(1, pd_true)

    data = pd.DataFrame({
        "month": month,
        "customer_id": customer_ids,
        "income": income,
        "age": age,
        "existing_debt": existing_debt,
        "utilization_rate": utilization,
        "target": target
    })

    return data