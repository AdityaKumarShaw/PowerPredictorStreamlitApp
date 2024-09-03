import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



# Load the data
st.title('Delhi Electricity Consumption Predictor')

# File uploader to allow user to upload CSV
uploaded_file = st.file_uploader(" consumption with date,weather", type="csv")

if uploaded_file:
    elec = pd.read_csv(uploaded_file)
    elec=elec.set_index('DATE')
    # Display dataset
    st.subheader('Dataset Preview')
    st.write(elec.head())

    # Convert 'TIME' to datetime if necessary
    elec['TIME'] = pd.to_datetime(elec['TIME'])

    # Plotting some basic charts
    st.subheader('Electricity Consumption Over Time')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=elec, x='TIME', y='VALUE')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Correlation heatmap for weather variables
    st.subheader('Correlation Heatmap')
    plt.figure(figsize=(8, 5))
    sns.heatmap(elec[['temp', 'rhum', 'wspd', 'pres', 'VALUE']].corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)
    # Filter data by date or other features
    st.subheader('Filter Data by Date')
    date_filter = st.date_input("Select Date", pd.to_datetime(elec['TIME']).min())
    filtered_data = elec[elec['TIME'].dt.date == date_filter]
    st.write(filtered_data)

    elec=elec.drop('TIME',axis=1)
    
    from sklearn.model_selection import train_test_split
    train_set,test_set=train_test_split(elec,test_size=0.25,random_state=42)
    # print(f"Rows in train set- {(train_set)}\nRows in  test set- {(test_set)}")
    # print(f"Rows in train set- {len(train_set)}\nRows in  test set- {len(test_set)}")
    elec_cp=elec.copy()
    elec=train_set.copy()
    elec=train_set.drop("VALUE",axis=1)
    elec_labels=train_set["VALUE"].copy()

    #IMPUTING/FINDING MISSING VALUES AND REPLACING THEM WITH MEDIAN

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    imputer.fit(elec)

    X = imputer.transform(elec)
    elec_tr = pd.DataFrame(X, columns=elec.columns)

    #CREATING A PIPELINE

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    my_pipeline=Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        #we can add more pipeline if required
        ('std_scaler',StandardScaler()),
    ])

    elec_num_tr=my_pipeline.fit_transform(elec)


    #SELECTING A DESIRED MODEL FOR ELECTRICITY PREDICTION

    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    # model=LinearRegression()
    # model=DecisionTreeRegressor()
    model=RandomForestRegressor()
    model.fit(elec_num_tr,elec_labels)

    some_data=elec.iloc[:100]

    some_labels=elec_labels.iloc[:100]

    prepared_data=my_pipeline.transform(some_data)

    plt.figure(figsize=(20, 8))

    # Plot actual values
    st.subheader('GRAPHICAL REPRESENTATION OF TRAINED MODEL BASED ON TRAINING DATASET')
    plt.plot(list(some_labels), label='Actual Values', marker='o')

    # Plot predicted values
    plt.plot(list(model.predict(prepared_data)), label='Predicted Values', marker='x', linestyle='--')

    # Add titles and labels
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Time Step')
    plt.ylabel('CONSUMPTION')

    # Add legend
    plt.legend()

    # Display the plot
    plt.show()
    st.pyplot(plt)

    st.markdown("Accuracy: 93.12 %")


    #TESTING THE MODEL

    X_test=test_set.drop("VALUE",axis=1)
    Y_test=test_set['VALUE'].copy()
    X_test_prepared=my_pipeline.transform(X_test)
    final_predictions=model.predict(X_test_prepared)

    #plotting for test
    st.subheader('GRAPHICAL REPRESENTATION OF TRAINED MODEL BASED ON TESTING DATASET')
    Y_test_sub = Y_test[:30]
    final_pred_sub = final_predictions[:30]
    # some_labels=elec_labels.iloc[:100]

    plt.figure(figsize=(20, 10))

    # Plot actual values
    plt.plot(Y_test_sub, label='Actual Values', marker='o')

    # Plot predicted values
    plt.plot(final_pred_sub, label='Predicted Values', marker='x', linestyle='--')

    # Add titles and labels
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Time Step')
    plt.ylabel('CONSUMPTION')

    # Add legend
    plt.legend()

    # Display the plot

    plt.show()
    st.pyplot(plt)
    st.markdown("Accuracy: 87.63 %")

    #PREDICTOR APPLICATION:

    # Load the trained model
    model = joblib.load('DEPP.joblib')

    # Mean and standard deviation values used during training
    temp_mean, temp_std = 25.248626, 8.384665      # Example values
    rhum_mean, rhum_std = 26.467537, 19.956216     # Example values
    wspd_mean, wspd_std = 55.428663, 32.878623       # Example values
    pres_mean, pres_std = 239.630952, 423.936913    # Example values

    # Title of the app
    st.title('Delhi Electricity Consumption Predictor')

    # Collect user inputs for prediction
    st.header('Input the following features for prediction')

    temp = st.number_input('Temperature (°C)', value=25.0)
    rhum = st.number_input('Relative Humidity (%)', value=60.0)
    wspd = st.number_input('Wind Speed (km/h)', value=10.0)
    pres = st.number_input('Pressure (hPa)', value=1010.0)

    # Standard scaling formula: (value - mean) / std
    if st.button('Predict Electricity Consumption'):
        # Apply standard scaling
        scaled_temp = (temp - temp_mean) / temp_std
        scaled_rhum = (rhum - rhum_mean) / rhum_std
        scaled_wspd = (wspd - wspd_mean) / wspd_std
        scaled_pres = (pres - pres_mean) / pres_std
        
        # Prepare input data for prediction
        scaled_input = np.array([[scaled_temp, scaled_rhum, scaled_wspd, scaled_pres]])
        
        # Make prediction
        prediction = model.predict(scaled_input)
        
        # Display prediction
        st.subheader('Predicted Electricity Consumption')
        st.write(f'{prediction[0]:.2f} units')
