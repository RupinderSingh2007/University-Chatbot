from flask import Flask, request, jsonify, render_template
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

app = Flask(__name__)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

important_words = {
    "what", "who", "how", "when", "where", "why",
    "you", "your", "do", "can", "am", "is", "are"
}

last_intent = None

response_pair = {
    
    "greeting": {
        "keywords": ["hi", "hello", "hey", "hii", "heyy"],
        "response": "Hello! I’m your University of Staffordshire Exam Support Chatbot. Ask me about exam dates, timetables, rooms, ID, permitted materials, late arrival, results, resits, or exam stress."
    },

    "acknowledgement": {
        "keywords": ["ok", "okay", "alright", "fine", "got it", "understood", "cool"],
        "response": "Alright, Let me know if you need anything else about your exams."
    },

    "thanks": {
        "keywords": ["thanks", "thank", "appreciate", "cheers"],
        "response": "You’re welcome! Feel free to ask me another exam-related question."
    },

    "goodbye": {
        "keywords": ["bye", "goodbye", "see", "later"],
        "response": "Goodbye! Best of luck with your exams."
    },

    "how_are_you": {
        "keywords": ["how are you", "doing"],
        "response": "I’m working well and ready to help with exam-related questions."
    },

    "positive_feedback": {
        "keywords": ["good", "great", "helpful", "nice", "perfect", "excellent"],
        "response": "I’m glad that helped. You can ask me about exam dates, timetables, rules, results, resits, or support."
    },

    "confused": {
        "keywords": ["confused", "understand", "lost", "unclear"],
        "response": "No problem. Try asking in simple words, for example: ‘Where is my exam timetable?’ or ‘Can I bring my phone?’"
    },

    "ask_examples": {
        "keywords": ["example", "examples", "ask", "questions"],
        "response": "You can ask questions like: ‘When are exam dates?’, ‘Can I bring my phone?’, ‘What happens if I fail?’, or ‘Where is my exam room?’"
    },

    "bot_identity": {
        "keywords": ["name", "chatbot", "bot", "identity"],
        "response": "I am a University of Staffordshire exam support chatbot. I provide quick guidance about exam rules, timetables, preparation, results, and student support during the exam period."
    },

    "bot_help": {
        "keywords": ["help", "support", "assist", "questions"],
        "response": "I can help with exam timetables, exam locations, student ID, calculators, permitted materials, academic integrity, online exam issues, late arrival, results, resits, appeals, and wellbeing support."
    },

    "exam_dates": {
        "keywords": [ "season", "date", "start", "end", "may", "2026"],
        "response": "For the May 2026 exam period, the main exam timetable includes exams starting from Thursday 7 May 2026. Exact dates depend on your module, so check your personal timetable and the latest exam timetable version before attending."
    },

    "exam_timetable": {
        "keywords": ["timetable", "schedule", "beacon", "portal"],
        "response": "You can check your exam timetable through Beacon or the official timetable system using your university username and password. Timetables can change, especially rooms, so check it again close to the exam date."
    },

    "exam_room": {
        "keywords": ["exam room", "location", "venue", "sports", "hall", "catalyst"],
        "response": "Your exam room is shown on your exam timetable. Venues may include rooms such as Sports Hall, Catalyst rooms, Mellor labs, or remote Blackboard exams depending on the module. Check the room, campus, and format before travelling."
    },

    "arrival_time": {
        "keywords": ["arrive", "early", "time", "before"],
        "response": "You should arrive at least 15 minutes before your exam starts. This gives you time to find the room, show ID, store belongings if required, and settle before the exam begins."
    },

    "student_id": {
        "keywords": ["student", "id", "card", "forgot", "lost", "proof"],
        "response": "You need proof of ID to sit your exam, so bring your student ID card. If you lose it or forget it, contact the exam team or student support immediately before the exam."
    },

    "allowed_materials": {
        "keywords": ["rules","material", "bring", "permitted", "pen", "pencil", "notes", "book"],
        "response": "Bring your student ID, pens, pencils, and only the materials permitted in your exam instructions. Do not bring unauthorised notes, books, dictionaries, or stored information unless your exam clearly allows them."
    },

    "calculator": {
        "keywords": ["calculator", "allow", "allowed", "permitted", "model"],
        "response": "Calculators are only permitted if your exam instructions allow them. If calculators are permitted, make sure the model is suitable for the exam and does not contain unauthorised stored material."
    },

    "electronic_devices": {
        "keywords": ["phone", "smartwatch", "watch", "earbuds", "device", "mobile", "earpiece"],
        "response": "Phones, smartwatches, hidden earpieces, earbuds, mini cameras, and other unauthorised electronic devices are not allowed in exams. Keep them switched off and stored exactly as the invigilator instructs."
    },

    "academic_integrity": {
        "keywords": ["academic", "integrity", "cheat", "plagiarism", "misconduct", "copying", "communicating", "ai", "chatgpt"],
        "response": "Exam misconduct includes unauthorised materials, copying, concealed notes, using unauthorised devices, unauthorised AI use, or communicating with anyone other than an invigilator. These actions can lead to academic misconduct penalties."
    },

    "technical_issues": {
        "keywords": ["technical", "issue", "online", "internet", "wifi", "moodle", "blackboard", "remote"],
        "response": "For an online or Blackboard exam, if you face technical issues, take screenshots, note the time, and contact IT support and your module team immediately. Keep evidence because it may be needed if the issue affects your submission."
    },

    "late_exam": {
        "keywords": ["late", "arrive", "missed", "delay"],
        "response": "If you are late, go straight to the invigilator or exam staff. You may lose exam time, and entry depends on exam rules and how late you arrive. Aim to arrive at least 15 minutes early to avoid this."
    },

    "bathroom_break": {
        "keywords": ["toilet", "break"],
        "response": "If you need the toilet during an exam, raise your hand and ask the invigilator. Do not leave without permission. Any break will normally be supervised according to exam rules."
    },

    "food_drink": {
        "keywords": ["bring", "allow", "food", "drink", "water", "snack"],
        "response": "Water is usually the safest item to bring, but food and other drinks may be restricted unless you have approved arrangements. Follow the exam room instructions and invigilator guidance."
    },

    "exam_stress": {
        "keywords": ["stress", "anxiety", "worry", "nervous", "panic", "wellbeing"],
        "response": "For exam stress, create a revision timetable, balance study with rest and exercise, and use academic study skills support. If anxiety feels serious, contact student support or wellbeing services before the exam period."
    },

    "study_resources": {
        "keywords": ["where", "study", "revision", "resource", "prepare", "skills"],
        "response": "Use lecture notes, module materials, past-style practice questions, library resources, and Academic Study Skills support. A good plan is to revise difficult topics first, practise exam-style questions, and check anything unclear with your tutor."
    },

    "results": {
        "keywords": ["result", "grade", "mark", "receive", "released"],
        "response": "Exam results are normally released through the student portal after marking and award board processes. If a result is missing, wait for official release dates first, then contact your module team or student support."
    },

    "pass_mark": {
        "keywords": ["pass", "mark", "minimum", "grade", "40", "50"],
        "response": "The minimum pass mark is normally 40% for undergraduate modules and 50% for postgraduate modules. Your exact module outcome depends on the full assessment structure and regulations."
    },

    "resit": {
        "keywords": ["fail", "resit", "reassessment", "failed", "cap", "capped"],
        "response": "If you fail an assessment, you may need to resit it. Re-sits are normally capped at the basic pass mark: 40% for undergraduate study and 50% for postgraduate study. Late submission is not allowed for re-sits."
    },

    "appeal_review": {
        "keywords": ["appeal", "review", "paper", "remark", "complain"],
        "response": "If you want to challenge a result, you must follow the formal academic appeals process. Valid grounds usually relate to process issues or exceptional circumstances, not simply disagreeing with academic judgement."
    },

    "exceptional_circumstances": {
        "keywords": ["exceptional", "circumstances", "medical", "emergency", "illness", "sick"],
        "response": "If illness, emergency, or serious personal circumstances affect your exam, submit an Exceptional Circumstances claim as soon as possible and provide suitable evidence. Do not wait until results are released."
    },

    "exam_clash": {
        "keywords": ["clash", "same", "time", "two", "conflict"],
        "response": "If you have two exams at the same time or a serious timetable clash, contact exams@staffs.ac.uk or student support as soon as you notice it. Do not leave it until the exam day."
    },

    "contact_exam_help": {
        "keywords": ["contact", "email",  "help", "query"],
        "response": "For questions about the exam timetable or Student Inclusion Plan exam arrangements, contact exams@staffs.ac.uk. For technical issues, contact IT support. For module-specific questions, contact your module tutor."
    },

    "exam_format": {
        "keywords": ["format", "type", "written", "online", "open book", "closed book"],
        "response": "Exam format depends on your module. It may be written, online, open-book, or closed-book. Always check your module instructions or exam timetable for the exact format."
    },

    "exam_duration": {
        "keywords": ["duration", "how long", "time limit", "hours"],
        "response": "Exam duration varies by module, typically between 1 to 3 hours. Check your exam timetable or module guide for exact timing."
    },

    "exam_submission": {
        "keywords": ["submit", "submission", "upload", "blackboard", "turn in"],
        "response": "For online exams, make sure you submit your work before the deadline on Blackboard. Always confirm submission and keep a copy as evidence."
    },

    "rough_work": {
        "keywords": ["rough", "extra paper", "notes during exam"],
        "response": "You can usually use rough paper if provided, but it must be handed in if required. Do not bring your own paper unless permitted."
    },

    "invigilator_help": {
        "keywords": ["invigilator", "help", "ask during exam"],
        "response": "If you need help during an exam, raise your hand and wait for the invigilator. Do not communicate with other students."
    },

    "leaving_exam": {
        "keywords": ["leave", "exit", "finish early"],
        "response": "You may leave early only if permitted by exam rules. Once you leave, you may not be allowed to re-enter."
    },

    "cheating_consequences": {
        "keywords": ["caught cheat", "penalty", "punishment"],
        "response": "If caught cheating, you may face serious academic penalties including failing the module or further disciplinary action."
    },

    "revision_tips": {
        "keywords": ["revise", "tips", "study tips", "how to prepare"],
        "response": "Use active recall, practice past questions, create summaries, and revise regularly instead of cramming. Focus on weak topics first."
    },

    "sleep_before_exam": {
        "keywords": ["sleep", "night before exam"],
        "response": "Getting enough sleep before an exam is important. A well-rested mind performs better than last-minute cramming."
    },

    "exam_miss": {
        "keywords": ["miss exam", "absent", "did not attend"],
        "response": "If you miss an exam, you must submit an Exceptional Circumstances claim with valid evidence. Otherwise, you may receive a fail."
    },

    "multiple_choice_exam": {
        "keywords": ["mcq", "multiple choice"],
        "response": "In multiple-choice exams, read each question carefully and avoid rushing. Sometimes more than one option may seem correct, so choose the best answer."
    },
}


def normalize_input(text):
    text = text.lower()

    replacements = {
        "hlo": "hello",
        "wat": "what",
        "wht": "what",
        "whn": "when",
        "ur": "your",
        "pls": "please",
        "plz": "please",
        "cant": "cannot",
        "id": "student id",
        "uni": "university",
        "examz": "exam",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_text(text):
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)

    # Remove duplicate tokens after tokenization
    tokens = list(set(tokens))

    filtered_tokens = [
        word for word in tokens
        if word not in stop_words or word in important_words
    ]

    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return stemmed_tokens


def stem_keywords(keywords):
    return [stemmer.stem(word.lower()) for word in keywords]


def get_nlp_response(user_input):
    global last_intent
    
    user_input = normalize_input(user_input)
    cleaned_input = clean_text(user_input)
    user_tokens = process_text(user_input)

    # Basic direct responses
    if cleaned_input in ["hi", "hello", "hey", "hii", "heyy"]:
        last_intent = "greeting"
        return response_pair["greeting"]["response"]

    if any(phrase in cleaned_input for phrase in ["who are you", "what are you", "your name", "what is your name"]):
        last_intent = "bot_identity"
        return response_pair["bot_identity"]["response"]

    if any(phrase in cleaned_input for phrase in ["what do you do", "what you do", "how do you help", "what can you do"]):
        last_intent = "bot_help"
        return response_pair["bot_help"]["response"]

    if any(phrase in cleaned_input for phrase in ["how are you", "how are you doing"]):
        last_intent = "how_are_you"
        return response_pair["how_are_you"]["response"]

    if any(phrase in cleaned_input for phrase in ["thank you", "thanks", "cheers"]):
        last_intent = "thanks"
        return response_pair["thanks"]["response"]

    if any(phrase in cleaned_input for phrase in ["bye", "goodbye", "see you"]):
        last_intent = "goodbye"
        return response_pair["goodbye"]["response"]

    if any(word in cleaned_input for word in ["confused", "lost", "unclear"]):
        last_intent = "confused"
        return response_pair["confused"]["response"]

    # High-priority exam intent checks
    if any(word in cleaned_input for word in ["timetable", "schedule"]):
        last_intent = "exam_timetable"
        return response_pair["exam_timetable"]["response"]

    if any(phrase in cleaned_input for phrase in ["exam rules", "rules", "what are the exam rules"]):
        last_intent = "allowed_materials"
        return response_pair["allowed_materials"]["response"]

    if any(word in cleaned_input for word in ["room", "venue", "location"]):
        last_intent = "exam_room"
        return response_pair["exam_room"]["response"]

    if any(word in cleaned_input for word in ["phone", "phones", "smartwatch", "watch", "earbuds", "earpiece", "mobile"]):
        last_intent = "electronic_devices"
        return response_pair["electronic_devices"]["response"]

    if "calculator" in cleaned_input:
        last_intent = "calculator"
        return response_pair["calculator"]["response"]

    if any(phrase in cleaned_input for phrase in ["student id", "id card", "student card"]):
        last_intent = "student_id"
        return response_pair["student_id"]["response"]

    if any(word in cleaned_input for word in ["late", "delay", "missed"]):
        last_intent = "late_exam"
        return response_pair["late_exam"]["response"]

    # Context-aware follow-up only if no clear topic is found
    if any(phrase in cleaned_input for phrase in ["and", "also", "what about"]):
        if last_intent:
            return f"Regarding {last_intent.replace('_', ' ')}, {response_pair[last_intent]['response']}"

    # General NLP keyword scoring
    best_match = None
    highest_score = 0

    for intent, data in response_pair.items():
        keyword_stems = stem_keywords(data["keywords"])
        score = len(set(user_tokens).intersection(set(keyword_stems)))

        if score > highest_score:
            highest_score = score
            best_match = intent

    if highest_score >= 1:
        last_intent = best_match
        return response_pair[best_match]["response"]

    return "Sorry, I do not have a precise answer for that yet. Please check official university guidance or contact your module tutor."

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.get_json()
    user_message = data.get("message", "")
    bot_response = get_nlp_response(user_message)
    return jsonify({"response": bot_response})


if __name__ == "__main__":
    app.run(debug=True)