import pandas as pd 
import plotly.express as px
import streamlit as st 
import io

df_pre = pd.read_csv("Loan_Default_Preprocessed.csv")
df = pd.read_csv("LoanDataset - LoansDatasest.csv")
df_semi = pd.read_csv("loan_data_semi.csv")
#-------------------------------------------------------------------------------------------------------------------------------------------------
def Data_Description():
    st.title("Loan Default Preprocessing")
    st.header("About Dataset:")
    st.write("""Loan Default Prediction Dataset\n
    This dataset contains information about customer loans,\n
    including customer demographics, loan details, and default status.\n
    The dataset can be used for various data analysis and machine learning tasks\n
    such as predicting loan default risk.\n
    The dataset consists of the following columns:
    
    """)
    with st.expander("Read more"):
        st.write("""
    customer_id: Unique identifier for each customer\n
    customer_age: Age of the customer\n
    customer_income: Annual income of the customer\n
    home_ownership: Home ownership status (e.g., RENT, OWN, MORTGAGE)\n
    employment_duration: Duration of employment in months\n
    loan_intent: Purpose of the loan (e.g., PERSONAL, EDUCATION, MEDICAL, VENTURE)\n
    loan_grade: Grade assigned to the loan\n
    loan_amnt: Loan amount requested\n
    loan_int_rate: Interest rate of the loan\n
    term_years: Loan term in years\n
    historical_default: Indicates if the customer has a history of default (Y/N)\n
    cred_hist_length: Length of the customer's credit history in years\n
    Current_loan_status: Current status of the loan (DEFAULT, NO DEFAULT)\n
        """)
#-------------------------------------------------------------------------------------------------------------------------------------------------
def Descriptive_analysis():
    st.title("Descriptive Analysis")
    def data_header():
        st.write("Data Header Before")
        st.write(df.head())
        st.write("Data Header After")
        st.write(df_pre.head())
    def data_info():
        st.subheader("Data info Before & After")
        buffer = io.StringIO()
        df.info(buf=buffer)
        buffer_2 = io.StringIO()
        df_pre.info(buf=buffer)
        s = buffer.getvalue()
        s_2 = buffer_2.getvalue()
        st.text(s)
        st.text(s_2)
    def describe():
        # Insert containers separated into tabs:
        tab1, tab2 = st.tabs(["Numerical", "Categorical"])
        with tab1:
            st.subheader("Numerical Before")
            st.write(df.describe())
            st.subheader("Numerical After")
            st.write(df_pre.describe())
        with tab2:
            st.subheader("Categorical Before")
            st.write(df.describe(include='object'))
            st.subheader("Categorical After")
            st.write(df_pre.describe(include='object'))
    def Nullandshape():
        N1,N2 = st.columns(2)
        with N1:
            st.subheader("Nulls Before")
            st.text(df.isnull().sum())
        with N2:
            st.subheader("Nulls After")
            st.text(df_pre.isnull().sum())
        S1,S2 = st.columns(2)
        with S1:
            st.subheader("Shape Before")
            st.write(df.shape)
        with S2:
            st.subheader("Shape After")
            st.write(df_pre.shape)
            
    selectors ={
        "Data.Header": data_header,
        "Data.Info" : data_info,
        "Data.Describe" : describe,
        "Data Nulls & Shape":Nullandshape
    }
    User_Choice2=st.sidebar.selectbox("Please select the Stats you want to see before and after", selectors.keys())
    selectors[User_Choice2]()
#-------------------------------------------------------------------------------------------------------------------------------------------------
def Charts():
    st.title("Charts")
    def box_plots():
        st.header("Box Plot")
        x= st.sidebar.radio("Choose column to plot", df_pre.select_dtypes(include= "number").columns)
        st.subheader(f"{x} Before")
        fig = px.box(data_frame=df_semi, x=x)
        st.plotly_chart(fig)
        st.subheader(f"{x} After")
        fig_2 = px.box(data_frame=df_pre, x=x)
        st.plotly_chart(fig_2)
    def histogram():
        st.header("Histogram")
        x= st.sidebar.radio("Choose column to plot", df_pre.select_dtypes(include= "number").columns)
        y=st.sidebar.checkbox("Click if you want to show numbers")
        st.subheader(f"{x} Before")
        fig_3 = px.histogram(data_frame=df_semi, x=x, text_auto=y)
        st.plotly_chart(fig_3)
        st.subheader(f"{x} After")
        fig_4 = px.histogram(data_frame=df_pre, x=x, text_auto=y)
        st.plotly_chart(fig_4)
    def piecharts():
        st.header("Pie Chart")
        x= st.sidebar.radio("Choose column to plot", df_pre.select_dtypes(include= "object").columns)
        st.subheader(f"{x} Before")
        fig_5 = px.pie(data_frame=df_semi, names=x)
        st.plotly_chart(fig_5)
        st.subheader(f"{x} After")
        fig_6 = px.pie(data_frame=df_pre, names=x)
        st.plotly_chart(fig_6)   
    def heatmap():
        st.header("Corr between numerical columns before")
        fig_7 = px.imshow(df_semi.corr(numeric_only=True),text_auto=True, height=600)
        st.plotly_chart(fig_7)
        st.header("Corr between numerical columns after")
        fig_8 = px.imshow(df_pre.corr(numeric_only=True),text_auto=True, height=600)
        st.plotly_chart(fig_8)
    def treemap():
        st.header("Corr between Categorical columns")
        grouped_2 = df_semi.groupby(['home_ownership', 'loan_intent', 'loan_grade', 'historical_default']).size().reset_index(name='count')
        fig_9 = px.treemap(grouped_2, path=['home_ownership', 'loan_intent', 'loan_grade', 'historical_default'], values='count',
                 title='Treemap of categorical columns Before')
        st.plotly_chart(fig_9)
        grouped = df_pre.groupby(['home_ownership', 'loan_intent', 'loan_grade', 'historical_default']).size().reset_index(name='count')
        fig_10 = px.treemap(grouped, path=['home_ownership', 'loan_intent', 'loan_grade', 'historical_default'], values='count',
                 title='Treemap of categorical columns After')
        st.plotly_chart(fig_10)
        
    selectors_2 ={
        "box_plots": box_plots,
        "Histograms" : histogram,
        "Pie Charts" : piecharts,
        "Heatmaps" :heatmap,
        "Treemap" : treemap
    }
    User_Choice3=st.sidebar.radio("Choose your Chart", selectors_2.keys())
    selectors_2[User_Choice3]()
#-------------------------------------------------------------------------------------------------------------------------------------------------
def insights_recommendations():
    st.header("Insights&Recommendations")
    st.subheader("Please Choose a Tab")
    # Function to generate plot for each tab
    def group_stack(x):
        df_p= df_pre.groupby(["Current_loan_status",x])[x].count()/df_pre[x].value_counts()*100
        df_p = df_p.unstack()
        fig_x = px.bar(df_p,y=df_p.columns, barmode="group", text_auto=True).update_layout(yaxis_title="Percentage%")
        return fig_x    
    def avg_histogram(x):
        fig = px.histogram(data_frame=df_pre, x="Current_loan_status", y=x , histfunc="avg" , text_auto=True)
        return fig
    
    t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11 = st.tabs(['home_ownership', 'loan_intent', 'loan_grade', 'historical_default','customer_age',
                                       "employment_duration","term_years","cred_hist_length",'customer_income','loan_amnt','loan_int_rate'])
    with t1:
        t1.write("My recommendation based on the numbers: Is to avoid giving loans to Renters")
        st.plotly_chart(group_stack('home_ownership'))
    with t2:
        t2.write("My recommendation based on the numbers: Is to avoid giving loans for Debt Consolidation intent")
        st.plotly_chart(group_stack('loan_intent'))    
    with t3:
        t3.write("My recommendation based on the numbers: Is to avoid giving loans to loan grades D")
        st.plotly_chart(group_stack('loan_grade'))        
    with t4:
        t4.write("My recommendation based on the numbers: Is that is safe to give loans agian to people that had Historical Defaults")
        st.plotly_chart(group_stack('historical_default'))
    with t5:
        t5.write("My recommendation based on the numbers: the younger the customer the Risker")
        st.plotly_chart(group_stack('customer_age'))
    with t6:
        t6.write("The Shorter the Employment Duration the Risker ")
        st.plotly_chart(group_stack('employment_duration'))
    with t7:
        t7.write("The longer the term period the Risker ")
        st.plotly_chart(group_stack('term_years'))
    with t8:
        t8.write("It doesn't have a significant effect just a slight downturn in default rates as the years are longer")
        st.plotly_chart(group_stack('cred_hist_length'))
    with t9:
        t9.write("The customer income should be 66k\year or higher to be in the safe side")
        st.plotly_chart(avg_histogram('customer_income'))
    with t10:
        t10.write("The best loans that performed are the ones under 9k")
        st.plotly_chart(avg_histogram('loan_amnt'))
    with t11:
        t11.write("The Safest to keep the interest rates below 10.47%")
        st.plotly_chart(avg_histogram('loan_int_rate'))

#-------------------------------------------------------------------------------------------------------------------------------------------------
Func_to_names = {
    
    
    "About Dataset" : Data_Description ,
    "Descriptive analysis" : Descriptive_analysis,
    "Charts" : Charts,
    "Insights&Recommendations": insights_recommendations
    
    
}


User_Choice = st.sidebar.selectbox("Select Your Page" ,Func_to_names.keys() )


Func_to_names[User_Choice]()
