import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from PIL import Image
import google.generativeai as genai
import os


# 1. Load and preprocess the data
# 1. Load and preprocess the data
df = pd.read_csv('cleaned_data.csv')
df['features'] = df['programming_languages'] + ' ' +df['programming_languages'] + ' ' + df['frameworks'] + ' ' + df['exp_level'] + ' ' + df['country']

# 2. Prepare the features and target
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['features'])
le = LabelEncoder()
y = le.fit_transform(df['job_category'])

# 3. Define the model with regularization
clf = RandomForestClassifier(
    n_estimators=100,
    min_samples_leaf=5,
    max_depth=10,
    random_state=42
)
clf.fit(X, y)

def get_recommendations(job_category,skills, experience_level, country, top_n=5):
    user_profile = f"{job_category} {' '.join(skills)} {experience_level} {country}"
    user_vector = tfidf.transform([user_profile])
    
    probabilities = clf.predict_proba(user_vector)[0]
    top_categories_indices = probabilities.argsort()[-top_n:][::-1]
    top_categories = le.inverse_transform(top_categories_indices)

    recommended_jobs = df[(df['job_category'].isin(top_categories)) & (df['country'] == country)]
    recommended_jobs['similarity'] = recommended_jobs['features'].apply(lambda x: np.dot(tfidf.transform([x]), user_vector.T).toarray()[0][0])
    recommended_jobs = recommended_jobs.sort_values('similarity', ascending=False).head(top_n)
    
    return recommended_jobs[['job_category', 'company_name', 'city', 'job_state', 'exp_level', 'programming_languages', 'posting_date', 'similarity']]


# Gemini AI setup
GOOGLE_API_KEY = "AIzaSyCEPEXd994vED407jK0d1GwnCWv1H559nc"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def gemini_chat(conversation_history, user_input):
    prompt = f"Conversation history: {conversation_history}\nUser: {user_input}"
    response = model.generate_content(prompt)
    return response.text

# Streamlit app
st.set_page_config(page_title="Job Recommendation", page_icon=":mag:", layout="wide")

# Add a header image
header_image = Image.open("job.jpg")
st.image(header_image, use_column_width=True)

st.title("Job Recommendation")
st.write("Find the perfect job that matches your skills and experience.")

if "selected_job" not in st.session_state:
    st.session_state.selected_job = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "recommendations" not in st.session_state:
    st.session_state.recommendations = pd.DataFrame()

# Create two columns
col1, col2 = st.columns(2)

with col1:
    with st.expander("User Input"):
        job_category = st.selectbox("Select your job category:",
                                    ["IT Professional", "Management and Leadership", "AI and Machine Learning",
                                     "Software Engineering", "Sales and Marketing", "Finance and Accounting",
                                     "Cybersecurity", "Data Entry", "DevOps and System Administration",
                                     "Data Science and Analytics", "Data Engineer", "Software Engineering",
                                     "Web Development", "Other"])
        skills = st.text_input("Enter your skills (comma-separated):", "python")
        experience_level = st.selectbox("Select your experience level:", ["Entry Level", "Experienced", "Manager"])
        country = st.selectbox("Select the country:",
                               ['Egypt', 'Saudi Arabia', 'United Arab Emirates'])

    if st.button("Get Recommendations"):
        st.session_state.recommendations = get_recommendations(job_category, skills.split(','), experience_level, country)

if not st.session_state.recommendations.empty:
    st.write("Recommended jobs:")

    # Make each job recommendation clickable using st.button()
    for index, job in st.session_state.recommendations.iterrows():
        job_key = f"{job['job_category']}_{job['company_name']}_{job['city']}_{index}"  # Unique key
        if st.button(f"{job['job_category']} at {job['company_name']} in {job['city']}", key=job_key):
            st.session_state.selected_job = job.to_dict()  # Store selected job in session state
with col2:
    st.subheader("Chat with our AI Job Assistant")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Display the selected job and generate response
    if st.session_state.selected_job:
        selected_job = st.session_state.selected_job
        st.write(f"You selected: {selected_job['job_category']} at {selected_job['company_name']} in {selected_job['city']}")

        # Send selected job details to chatbot
        response = gemini_chat([m["content"] for m in st.session_state.messages],
                               f"Analyze this job: {selected_job}")
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # React to user input
    if prompt := st.chat_input("What would you like to know about jobs or careers?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get AI response
        response = gemini_chat([m["content"] for m in st.session_state.messages[:-1]], prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
