import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

faq_data = [
    {"q": "What is the name of the institution?", "a": "Welcome to the admission portal for the Hyderabad Institute of Technology and Management."},
    {"q": "What is the general admission process for the college?", "a": "The admission process involves submitting an online application, clearing the required entrance examination, attending the counseling session, and finally verifying your documents for seat confirmation."},
    {"q": "What are the most popular engineering branches available?", "a": "We offer several in-demand branches, with Computer Science and Engineering (Artificial Intelligence and Machine Learning) being one of our most highly sought-after academic programs."},
    {"q": "What entrance exams are accepted for B.Tech admissions?", "a": "We accept scores from state-level engineering entrance exams (such as TS EAMCET) as well as national-level exams like JEE Main for our regular quota seats."},
    {"q": "What is the minimum percentage required in the 12th standard?", "a": "Candidates must have secured a minimum of 60% aggregate marks in their 10+2 examinations with Physics, Chemistry, and Mathematics as core subjects."},
    {"q": "How do I apply for management quota seats?", "a": "Management quota seats, also known as Category-B seats, can be applied for directly through the official college website or by visiting the admission office on campus."},
    {"q": "Where is the campus located?", "a": "The campus is located in a green, sprawling area equipped with modern infrastructure, providing a peaceful environment highly conducive to learning and technical development."},
    {"q": "What documents are required during the counseling process?", "a": "You will need your 10th and 12th mark sheets, entrance exam rank card, transfer certificate, migration certificate (if applicable), and recent passport-sized photographs."},
    {"q": "Is hostel accommodation available for students?", "a": "Yes, we provide secure, separate hostel facilities for boys and girls with amenities like Wi-Fi, laundry, and round-the-clock security."},
    {"q": "What is the fee structure for the computer science programs?", "a": "The fee structure is revised annually as per government and university regulations. Please check the 'Fee Structure' PDF available on the admissions portal for the current academic year."},
    {"q": "Are there any scholarships available for meritorious students?", "a": "Yes, scholarships are awarded based on top ranks in entrance exams and outstanding academic performance in the 12th standard. State-sponsored fee reimbursement schemes are also applicable for eligible students."},
    {"q": "How does the seat allotment process work?", "a": "Seats are allotted purely based on merit (entrance exam rank), category reservations as per state norms, and the branch preferences you select during the counseling phase."},
    {"q": "Can I change my branch after the first year?", "a": "Branch changes are permitted at the end of the first year strictly based on academic merit (CGPA) and the availability of vacant seats in the desired branch."},
    {"q": "What is the deadline for submitting the application form?", "a": "The application deadline varies depending on the specific counseling phase. It is typically announced on the official website two weeks after the entrance exam results are declared."},
    {"q": "Is there an application fee, and is it refundable?", "a": "There is a nominal application processing fee that must be paid online. This fee is strictly non-refundable under all circumstances."},
    {"q": "Does the college provide placement assistance?", "a": "Yes, we have a dedicated Training and Placement Cell that conducts regular skill-building bootcamps, resume workshops, and campus recruitment drives with top tech companies."},
    {"q": "Are there any extracurricular clubs or tech societies?", "a": "We strongly encourage holistic development. Students can join various technical chapters, coding clubs, robotics societies, and cultural organizations on campus."},
    {"q": "How is the curriculum structured for the AI and Machine Learning branch?", "a": "The curriculum combines core computer science principles with specialized modules in neural networks, data structures, natural language processing, and deep learning, supplemented by hands-on lab work."},
    {"q": "What is the duration of the B.Tech program?", "a": "The standard Bachelor of Technology (B.Tech) program spans four academic years, divided into eight semesters."},
    {"q": "Are lateral entry admissions permitted for diploma holders?", "a": "Yes, candidates holding a recognized 3-year diploma in engineering can apply for direct admission into the second year of the B.Tech program via the ECET exam."},
    {"q": "How will I be notified about my admission status?", "a": "All official updates, including seat allotment and document verification statuses, will be sent to your registered email address and mobile number via SMS."},
    {"q": "Is there a specific dress code or uniform for students?", "a": "Students are expected to wear neat, professional, and modest casual attire. A formal uniform is only required during specific lab sessions, placement drives, and official college events."},
    {"q": "Can I visit the campus before confirming my admission?", "a": "Absolutely. We welcome prospective students and their parents to schedule a campus tour through the admissions office to view our labs, library, and infrastructure."},
    {"q": "Does the college offer transportation services?", "a": "Yes, we operate a fleet of college buses covering all major routes in the city and surrounding areas to ensure safe and convenient commuting for our students."},
    {"q": "Who can I contact if I face issues during the online application?", "a": "For any technical difficulties or admission-related queries, you can reach out to our admissions helpdesk via the contact number or email ID provided at the bottom of the admissions portal."}
]

faq_questions = [item["q"] for item in faq_data]
faq_answers = [item["a"] for item in faq_data]

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return " ".join(tokens)

@st.cache_resource
def setup_bot():
    cleaned_questions = [preprocess_text(q) for q in faq_questions]
    vectorizer = TfidfVectorizer()
    faq_vectors = vectorizer.fit_transform(cleaned_questions)
    return vectorizer, faq_vectors

vectorizer, faq_vectors = setup_bot()

def get_best_answer(user_query, threshold=0.2):
    cleaned_query = preprocess_text(user_query)
    query_vector = vectorizer.transform([cleaned_query])

    similarities = cosine_similarity(query_vector, faq_vectors)[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    if best_score < threshold:
        return "Sorry, I don't have an answer for that. Please contact the admissions office for more help."
    else:
        return faq_answers[best_index]

st.title("🎓 College Admission Assistant")
st.write("Welcome! Ask me anything about the admission process, fees, or campus life.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! What would you like to know about our college admissions?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Type your question here..."):
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    bot_response = get_best_answer(user_input)

    with st.chat_message("assistant"):
        st.markdown(bot_response)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
