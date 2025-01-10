# 1. import libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# 2. Set up the app title and description
st.title("✈️ Airline Passenger Satisfaction Prediction")
st.markdown(
    """
    Welcome to the **Airline Passenger Satisfaction Prediction** app! 
    This app predicts whether a passenger is satisfied or dissatisfied based on their ratings for various airline services.
    Please rate the following services on a scale of **1 to 5** (1 = Poor, 5 = Excellent).
    """
)

# Add a divider for better visual separation
st.divider()

# 3. Inputs with rating restrictions (1-5)
st.subheader("Rate Your Experience")
col1, col2 = st.columns(2)  # Split inputs into two columns for better layout

with col1:
    Seat_comfort = st.number_input(label="Seat comfort", value=1.0, min_value=1.0, max_value=5.0, step=1.0)
    Inflight_wifi_service = st.number_input(label="Inflight wifi service", value=2.0, min_value=1.0, max_value=5.0, step=1.0)
    Inflight_entertainment = st.number_input(label="Inflight entertainment", value=4.0, min_value=1.0, max_value=5.0, step=1.0)
    Online_support = st.number_input(label="Online support", value=3.0, min_value=1.0, max_value=5.0, step=1.0)
    Ease_of_Online_booking = st.number_input(label="Ease of Online booking", value=2.0, min_value=1.0, max_value=5.0, step=1.0)

with col2:
    On_board_service = st.number_input(label="On-board service", value=1.0, min_value=1.0, max_value=5.0, step=1.0)
    Leg_room_service = st.number_input(label="Leg room service", value=3.0, min_value=1.0, max_value=5.0, step=1.0)
    Baggage_handling = st.number_input(label="Baggage handling", value=1.0, min_value=1.0, max_value=5.0, step=1.0)
    Checkin_service = st.number_input(label="Checkin service", value=1.0, min_value=1.0, max_value=5.0, step=1.0)
    Cleanliness = st.number_input(label="Cleanliness", value=1.0, min_value=1.0, max_value=5.0, step=1.0)
    Online_boarding = st.number_input(label="Online boarding", value=1.0, min_value=1.0, max_value=5.0, step=1.0)

# 4. Combine input into an array of X
X_num = np.array(
    [
        [
            Seat_comfort,
            Inflight_wifi_service,
            Inflight_entertainment,
            Online_support,
            Ease_of_Online_booking,
            On_board_service,
            Leg_room_service,
            Baggage_handling,
            Checkin_service,
            Cleanliness,
            Online_boarding,
        ]
    ],
    dtype=np.float64,  # Use float64 for decimal values
)

# 5. Import model
# 5.1 Import scaler
with open(file="scale.pkl", mode="rb") as scale_file:
    scale = pickle.load(file=scale_file)

# 5.2 Label encoder
with open(file="encode.pkl", mode="rb") as encode_file:
    encode = pickle.load(file=encode_file)

# 5.3 Logistic Regression model
with open(file="lg.pkl", mode="rb") as lg_file:
    lg_model = pickle.load(file=lg_file)

# 6. Preprocessing data
X = scale.transform(X_num)  # Use transform instead of fit_transform for pre-fitted scaler

# 7. Make prediction
prediction = lg_model.predict(X)
prediction_decoded = encode.inverse_transform(prediction)

# 8. Combine input features and prediction into a DataFrame
data = np.concatenate([X, prediction_decoded.reshape(-1, 1)], axis=1)  # Use np.concatenate
df = pd.DataFrame(
    data=data,
    columns=[
        "Seat comfort",
        "Inflight wifi service",
        "Inflight entertainment",
        "Online support",
        "Ease of Online booking",
        "On-board service",
        "Leg room service",
        "Baggage handling",
        "Checkin service",
        "Cleanliness",
        "Online boarding",
        "Satisfaction",
    ],
)

# 9. Display the results
st.divider()
st.subheader("Prediction Results")

# Show the satisfaction prediction
if prediction_decoded[0] == "satisfied":
    st.success("🎉 The passenger is **satisfied** with the airline services!")
else:
    st.error("😞 The passenger is **dissatisfied** with the airline services.")

# Show the detailed DataFrame
st.write("### Detailed Ratings and Prediction")
st.dataframe(df)
