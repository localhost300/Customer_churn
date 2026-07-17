import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Customer Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)



# ==================================================
# THEME CONTROL
# ==================================================

st.sidebar.title("⚙️ Dashboard Settings")


theme = st.sidebar.radio(
    "Select Theme",
    [
        "Light Mode",
        "Dark Mode"
    ]
)


if theme == "Dark Mode":

    st.markdown(
        """
        <style>

        .stApp{
            background-color:#0E1117;
            color:white;
        }

        div[data-testid="metric-container"]{
            background-color:#1B1F27;
            padding:20px;
            border-radius:15px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


else:

    st.markdown(
        """
        <style>

        div[data-testid="metric-container"]{
            background-color:#F5F7FB;
            padding:20px;
            border-radius:15px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )



# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load(
    "models/churn_model.pkl"
)


scaler = joblib.load(
    "models/scaler.pkl"
)


features = joblib.load(
    "models/features.pkl"
)



data = pd.read_csv(
    "data/customer_churn_cleaned.csv"
)




# ==================================================
# HEADER
# ==================================================

st.title(
    "📊 Customer Churn Intelligence Dashboard"
)


st.caption(
    "Machine Learning powered customer retention analytics"
)



# ==================================================
# KPI SECTION
# ==================================================

total_customers = len(data)


churned_customers = int(
    data["Churn Value"].sum()
)


retained_customers = (
    total_customers - churned_customers
)


churn_rate = (
    churned_customers /
    total_customers
) * 100


retention_rate = (
    100 - churn_rate
)



c1,c2,c3,c4 = st.columns(4)


c1.metric(
    "👥 Total Customers",
    f"{total_customers:,}"
)


c2.metric(
    "🚨 Churned Customers",
    f"{churned_customers:,}"
)


c3.metric(
    "📉 Churn Rate",
    f"{churn_rate:.2f}%"
)


c4.metric(
    "✅ Retention Rate",
    f"{retention_rate:.2f}%"
)




# ==================================================
# NAVIGATION
# ==================================================

page = st.sidebar.selectbox(
    "Navigate",
    [
        "Overview",
        "Churn Analysis",
        "Model Insights",
        "High Risk Customers",
        "Prediction Tool"
    ]
)




# ==================================================
# OVERVIEW
# ==================================================

if page == "Overview":


    st.header(
        "Executive Overview"
    )


    overview = pd.DataFrame(
        {
            "Status":
            [
                "Stayed",
                "Churned"
            ],

            "Customers":
            [
                retained_customers,
                churned_customers
            ]
        }
    )



    fig = px.pie(
        overview,
        values="Customers",
        names="Status",
        hole=0.5,
        title="Customer Retention Distribution"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )





# ==================================================
# CHURN ANALYSIS
# ==================================================

elif page == "Churn Analysis":


    st.header(
        "Customer Churn Analysis"
    )


    col1,col2 = st.columns(2)



    # Contract

    contract = pd.crosstab(
        data["Contract"],
        data["Churn Value"],
        normalize="index"
    ).reset_index()



    contract["Churn Rate"] = (
        contract[1] * 100
    )



    fig = px.bar(
        contract,
        x="Contract",
        y="Churn Rate",
        text="Churn Rate",
        title="Churn Rate by Contract"
    )


    fig.update_traces(
        texttemplate="%{text:.1f}%"
    )


    col1.plotly_chart(
        fig,
        use_container_width=True
    )



    # Internet


    internet = pd.crosstab(
        data["Internet Type"],
        data["Churn Value"],
        normalize="index"
    ).reset_index()



    internet["Churn Rate"] = (
        internet[1] * 100
    )



    fig = px.bar(
        internet,
        x="Internet Type",
        y="Churn Rate",
        text="Churn Rate",
        title="Churn Rate by Internet Type"
    )


    fig.update_traces(
        texttemplate="%{text:.1f}%"
    )


    col2.plotly_chart(
        fig,
        use_container_width=True
    )



    st.subheader(
        "Satisfaction Score vs Churn"
    )


    fig,ax = plt.subplots(
        figsize=(5,3)
    )


    sns.boxplot(
        data=data,
        x="Churn Value",
        y="Satisfaction Score",
        width=0.35,
        ax=ax
    )


    ax.set_xticklabels(
        [
            "Stayed",
            "Churned"
        ]
    )


    ax.set_title(
        "Customer Satisfaction Distribution"
    )


    plt.tight_layout()


    st.pyplot(fig)




# ==================================================
# MODEL INSIGHTS
# ==================================================

elif page == "Model Insights":


    st.header(
        "Important Churn Drivers"
    )


    importance = pd.DataFrame(
        {
            "Feature":features,
            "Coefficient":model.coef_[0]
        }
    )


    importance["Impact"] = np.where(
        importance["Coefficient"] > 0,
        "Higher Churn Risk",
        "Lower Churn Risk"
    )


    importance = importance.sort_values(
        "Coefficient"
    )


    fig = px.bar(
        importance.tail(15),
        x="Coefficient",
        y="Feature",
        color="Impact",
        orientation="h",
        title="Top Factors Affecting Churn"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )





# ==================================================
# HIGH RISK CUSTOMERS
# ==================================================

elif page == "High Risk Customers":


    st.header(
        "Customers With Highest Churn Risk"
    )


    X = data.drop(
        "Churn Value",
        axis=1
    )



    encoded = pd.get_dummies(
        X,
        drop_first=True
    )


    encoded = encoded.reindex(
        columns=features,
        fill_value=0
    )


    probabilities = (
        model.predict_proba(
            scaler.transform(encoded)
        )[:,1]
        *100
    )



    risk = data.copy()


    risk["Churn Probability"] = (
        probabilities.round(2)
    )


    risk = risk.sort_values(
        "Churn Probability",
        ascending=False
    )



    search = st.text_input(
        "Search Customer ID"
    )


    if search:

        risk = risk[
            risk["Customer ID"]
            .str.contains(search)
        ]



    st.dataframe(

        risk[
            [
                "Customer ID",
                "Tenure in Months",
                "Monthly Charge",
                "Satisfaction Score",
                "Churn Probability"
            ]
        ],

        use_container_width=True

    )



    csv = risk.to_csv(
        index=False
    )


    st.download_button(
        "⬇️ Download Risk Report",
        csv,
        "churn_risk_report.csv"
    )





# ==================================================
# PREDICTION TOOL
# ==================================================

elif page == "Prediction Tool":


    st.header(
        "Individual Customer Prediction"
    )


    tenure = st.slider(
        "Tenure in Months",
        0,
        100,
        12
    )


    monthly_charge = st.number_input(
        "Monthly Charge",
        value=70.0
    )


    satisfaction = st.slider(
        "Satisfaction Score",
        1,
        5,
        3
    )


    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )


    internet = st.selectbox(
        "Internet Type",
        [
            "DSL",
            "Fiber Optic",
            "No Internet"
        ]
    )


    security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No"
        ]
    )


    backup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No"
        ]
    )


    protection = st.selectbox(
        "Device Protection Plan",
        [
            "Yes",
            "No"
        ]
    )


    support = st.selectbox(
        "Premium Tech Support",
        [
            "Yes",
            "No"
        ]
    )


    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No"
        ]
    )


    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No"
        ]
    )


    streaming_music = st.selectbox(
        "Streaming Music",
        [
            "Yes",
            "No"
        ]
    )



    if st.button(
        "Predict Customer Risk"
    ):


        customer = pd.DataFrame(
            0,
            columns=features,
            index=[0]
        )



        customer["Tenure in Months"] = tenure

        customer["Monthly Charge"] = monthly_charge

        customer["Satisfaction Score"] = satisfaction



        mappings = {

            "Contract_One Year":
            contract=="One year",

            "Contract_Two Year":
            contract=="Two year",

            "Internet Type_DSL":
            internet=="DSL",

            "Internet Type_Fiber Optic":
            internet=="Fiber Optic",

            "Online Security_Yes":
            security=="Yes",

            "Online Backup_Yes":
            backup=="Yes",

            "Device Protection Plan_Yes":
            protection=="Yes",

            "Premium Tech Support_Yes":
            support=="Yes",

            "Streaming TV_Yes":
            streaming_tv=="Yes",

            "Streaming Movies_Yes":
            streaming_movies=="Yes",

            "Streaming Music_Yes":
            streaming_music=="Yes"

        }



        for column,value in mappings.items():

            if column in customer.columns:

                customer[column] = value



        customer = customer.astype(int)



        probability = (

            model.predict_proba(
                scaler.transform(customer)
            )[0][1]

            *100

        )



        st.metric(
            "Churn Probability",
            f"{probability:.2f}%"
        )



        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability,
                title={
                    "text":"Risk Score"
                },
                gauge={
                    "axis":{
                        "range":[0,100]
                    }
                }
            )
        )


        st.plotly_chart(
            gauge,
            use_container_width=True
        )



        if probability >=70:

            st.error(
                "🚨 High Risk Customer - Retention action recommended"
            )

        elif probability >=30:

            st.warning(
                "⚠️ Medium Risk Customer - Monitor closely"
            )

        else:

            st.success(
                "✅ Low Risk Customer"
            )