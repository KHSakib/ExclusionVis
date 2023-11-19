import base64
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout='wide')
st.title("ExclusionVis: Visual Interactive System for Exclusion Identification")


def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="dataset.csv">Download CSV File</a>'
    return href

def visualize_pattern_generation(filtered_data, top5_patterns_df, algorithm_choice, user_min_confidence):
    st.markdown(f'**Total Number of Reasons: {filtered_data.shape[0]}**')
    st.dataframe(filtered_data[["Exclusion", "reason", "confidence", "report_type"]], height=450)
    st.markdown(filedownload(filtered_data), unsafe_allow_html=True)

def create_table():
    data = {
        'Algorithm': ['Binary Relevance', 'Binary Relevance', 'Binary Relevance', 'Binary Relevance', 'Binary Relevance', 'Classifier Chain', 'Classifier Chain', 'Classifier Chain', 'Classifier Chain', 'Classifier Chain', 'Label Powerset', 'Label Powerset', 'Label Powerset', 'Label Powerset', 'Label Powerset', 'Ensemble Learning', 'Ensemble Learning', 'Ensemble Learning', 'Ensemble Learning', 'Ensemble Learning'],
        'Classifier': ['MultinomialNB', 'SVC', 'Logistic regression', 'Random Forest', 'Decision Tree', 'MultinomialNB', 'SVC', 'Logistic regression', 'Random Forest', 'Decision Tree', 'MultinomialNB', 'SVC', 'Logistic regression', 'Random Forest', 'Decision Tree', 'MultinomialNB', 'SVC', 'Logistic regression', 'Random Forest', 'Decision Tree'],
        'Precision': ['0.26', '0.62', '0.80', '0.93', '0.63', '0.26', '0.94', '0.80', '0.93', '0.61', '0.35', '0.89', '0.76', '0.88', '0.57', '0.26', '0.93', '0.80', '0.92', '0.59'],
        'Recall': ['0.11', '0.46', '0.41', '0.32', '0.56', '0.11', '0.11', '0.41', '0.32', '0.56', '0.45', '0.06', '0.41', '0.40', '0.48', '0.10', '0.10', '0.43', '0.34', '0.54'],
        'F-Score': ['0.15', '0.53', '0.54', '0.47', '0.60', '0.15', '0.20', '0.54', '0.48', '0.59', '0.08', '0.11', '0.54', '0.56', '0.52', '0.15', '0.19', '0.56', '0.49', '0.56'],
        'Hamming loss': ['0.002', '0.002', '0.001', '0.001', '0.001', '0.001', '0.001', '0.001', '0.001', '0.001', '0.001', '0.001', '0.001', '0.001', '0.001', '0.001', '0.001', '0.001', '0.001', '0.001']
    }

    df = pd.DataFrame(data)
    return df

#Generate_age_gender_chart 

def generate_age_gender_chart(trauma_reasons, user_min_age, user_gender, user_min_confidence):
    # Filter the trauma_reasons dataset based on user inputs
    if user_gender == "Both":
        filtered_age_gender_data = trauma_reasons[(trauma_reasons['age'] >= user_min_age) &
                                                  (trauma_reasons['confidence'] >= user_min_confidence)]
    else:
        filtered_age_gender_data = trauma_reasons[(trauma_reasons['age'] >= user_min_age) &
                                                  (trauma_reasons['gender'] == user_gender) &
                                                  (trauma_reasons['confidence'] >= user_min_confidence)]

    # Define age bins and labels
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    age_labels = ['1-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']

    # Assign age group labels to the dataset
    filtered_age_gender_data['age_group'] = pd.cut(filtered_age_gender_data['age'], bins=age_bins, labels=age_labels, right=False)

    # Calculate counts for each combination of age group and gender
    counts = filtered_age_gender_data.groupby(['age_group', 'gender']).size().reset_index(name='count')

    # Create an interactive bar chart using Plotly Express with barmode='group'
    age_gender_chart = px.bar(counts, x='age_group', y='count', color='gender',
                              title=f"Reasons Count by Age Group & Gender",
                              labels={'age_group': 'Age Group', 'count': 'Reason Count', 'gender': 'Gender'},
                              category_orders={'age_group': age_labels, 'gender': ['Male', 'Female']},
                              barmode='group',  # Separate bars for male and female
                              height=530)

    # Display the actual count on each bar
    age_gender_chart.update_traces(texttemplate='%{y}', textposition='outside')

    # Display the chart in the Streamlit app
    st.write(age_gender_chart, use_container_width=True)

def generate_top_occupation_gender_chart(trauma_reasons, user_gender, user_min_confidence, top_n=5):
    # Remove text after "-" in the "occupation" column
    trauma_reasons['occupation'] = trauma_reasons['occupation'].str.split('-').str[0].str.strip()

    # Count reasons by occupation and gender
    counts_occupation_gender = trauma_reasons.groupby(['occupation', 'gender'])['reason'].count().reset_index(name='count')

    # Filter by user-selected gender
    if user_gender != "Both":
        counts_occupation_gender = counts_occupation_gender[counts_occupation_gender['gender'] == user_gender]

    # Select top N occupations for the specified gender or for both genders
    top_occupations = counts_occupation_gender.groupby('occupation')['count'].sum().nlargest(top_n).index
    counts_occupation_top = counts_occupation_gender[counts_occupation_gender['occupation'].isin(top_occupations[:top_n])]

    # Create an interactive bar chart using Plotly Express with barmode='group'
    occupation_gender_chart = px.bar(counts_occupation_top, x='occupation', y='count', color='gender',
                                      title=f"Top {top_n} Occupations by Reasons Count",
                                      labels={'occupation': 'Occupation', 'count': 'Reason Count', 'gender': 'Gender'},
                                      category_orders={'gender': ['Male', 'Female']},
                                      barmode='group',  # Separate bars for male and female
                                      height=530)  # Increase the height

    # Display the actual count on each bar
    occupation_gender_chart.update_traces(texttemplate='%{y}', textposition='outside', hovertext=counts_occupation_top['occupation'])

    # Set x-axis title to be more concise
    occupation_gender_chart.update_layout(xaxis_title="Occupation")

    # Rotate x-axis labels
    occupation_gender_chart.update_layout(xaxis_tickangle=-35)

    # Display the chart in the Streamlit app
    st.write(occupation_gender_chart, use_container_width=True)


def run_dashboard():
    algorithm_choices = {
        "Logistic Regression": "Logistic Regression Configuration",
        "B": "B",
        "C": "C"
    }

    algorithm_choice = st.sidebar.selectbox("Select algorithm type:", list(algorithm_choices.keys()))

    st.sidebar.header(algorithm_choices[algorithm_choice])

    # Add sliders for minimum age and gender selection to the Streamlit sidebar
    with st.sidebar.form("user_form"):
        user_min_confidence = st.slider("Min Confidence", min_value=0.00, max_value=1.00, value=0.5, step=0.01)
        user_min_age = st.slider("Min Age", min_value=0, max_value=70, value=30, step=1)
        user_gender = st.selectbox("Select Gender", ["Male", "Female", "Both"])
        algo_filter = st.selectbox('Select Algo', ('Binary Relevance', 'Classifier Chain', 'Label Powerset', 'Ensemble Learning'))
        filter_option = st.selectbox('Select filter option', ('Precision', 'Recall'))
        generate_pattern = st.form_submit_button("Generate Pattern")

    # If the "Generate Pattern" button is clicked
    if generate_pattern:
        trauma_reasons = pd.read_csv('trauma_reasons.csv')
        trauma_disclosures = pd.read_csv('trauma_disclosures.csv')

        # Merge datasets based on 'enquiry id' and 'POLICY_NUMBER'
        merged_data = pd.merge(trauma_reasons, trauma_disclosures, on=['enquiry id', 'POLICY_NUMBER'], how='left')

        # Extract customer age range, occupation, gender
        trauma_reasons['customer_age_range'] = merged_data['customer age range']
        trauma_reasons['occupation'] = merged_data['occupation']
        trauma_reasons['age'] = merged_data['customer age']
        trauma_reasons['gender'] = merged_data['gender']

        # Save the updated trauma_reasons dataset
        trauma_reasons.to_csv("merged_trauma_reasons.csv", index=False)

        # Filter data based on user input
        filtered_data = trauma_reasons[trauma_reasons['confidence'] >= user_min_confidence]

        col1, col2 = st.columns(2)

        with col1:
        # Visualize pattern generation
            visualize_pattern_generation(filtered_data, filtered_data.head(5), algorithm_choice, user_min_confidence)
        with col2:
        # Visualize pattern generation
            st.markdown('**Result Table**')
            df = create_table()

            
            if filter_option == 'Precision':
                st.write("Table according to Precesion:")
                st.table(df[['Algorithm', 'Classifier', 'F-Score', 'Hamming loss'] + ['Precision']][(df['Algorithm'] == algo_filter)])
            elif filter_option == 'Recall':
                st.write("Table according to Recall:")
                st.table(df[['Algorithm', 'Classifier', 'F-Score', 'Hamming loss'] + ['Recall']][(df['Algorithm'] == algo_filter)])
            else:
                st.write("Invalid option selected")
        

        # Organize charts in two columns
        col1, col2 = st.columns(2)

        # Generate and display the Age & Gender Chart in the first column
        with col1:
            generate_age_gender_chart(trauma_reasons, user_min_age, user_gender, user_min_confidence)

        # Generate and display the Top Occupations & Gender Chart in the second column
        with col2:
            generate_top_occupation_gender_chart(trauma_reasons, user_gender, user_min_confidence, top_n=5)

    

if __name__ == "__main__":
    run_dashboard()
