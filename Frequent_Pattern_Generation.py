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
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'**Total Number of Reasons: {filtered_data.shape[0]}**')
        st.dataframe(filtered_data[["Exclusion", "reason", "confidence", "report_type"]], height=450)
        st.markdown(filedownload(filtered_data), unsafe_allow_html=True)

    with col2:
        try:
            # Define confidence ranges
            confidence_ranges = [(i / 10, (i + 1) / 10) for i in range(int(user_min_confidence * 10), 10)]

            # Create 'confidence_range' column
            filtered_data['confidence_range'] = pd.cut(filtered_data['confidence'], bins=[x[0] for x in confidence_ranges] + [1.01])

            # Convert 'confidence_range' to strings
            filtered_data['confidence_range'] = filtered_data['confidence_range'].astype(str)

            # Calculate the percentage of reasons in each confidence range
            percentage_by_confidence_range = (filtered_data.groupby('confidence_range').size() / len(filtered_data)) * 100
            percentage_by_confidence_range = percentage_by_confidence_range.reset_index(name='percentage')

            fig = px.bar(percentage_by_confidence_range, x='confidence_range', y='percentage',
                        text='percentage',
                        title=f"Percentage of Reasons by Confidence Range (Threshold: {user_min_confidence})",
                        labels={'confidence_range': 'Confidence Range', 'percentage': 'Percentage of Reasons'},
                        height=530)

            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

            st.write(fig, use_container_width=True)  # Use st.write instead of st.pyplot
        except Exception as e:
            st.warning(f"An error occurred: {e}")

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

        # Visualize pattern generation
        visualize_pattern_generation(filtered_data, filtered_data.head(5), algorithm_choice, user_min_confidence)

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
