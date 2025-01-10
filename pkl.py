import pickle

# Load the trained logistic regression model
with open("scale.pkl", "rb") as file:
    model = pickle.load(file)

# Check the attributes of the loaded model
print("Attributes of the model:")
print(dir(model))

# Access the coefficients and intercept
if hasattr(model, "coef_"):
    print("Model Coefficients:")
    print(model.coef_)
else:
    print("The model does not have coefficients (coef_).")

if hasattr(model, "intercept_"):
    print("Model Intercept:")
    print(model.intercept_)
else:
    print("The model does not have an intercept (intercept_).")
