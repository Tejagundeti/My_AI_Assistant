import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# API keys are loaded from environment variables or a local .env file.

# Define resume file path
RESUME_PATH = "./TejaGundeti_FullStackDeveloper.pdf"  

# App title
st.markdown("<h1 style='text-align: center;'>Teja's AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Ask any question about Teja and I will answer you. No more his resume is needed</p>", unsafe_allow_html=True)

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

if "thinking" not in st.session_state:
    st.session_state.thinking = False

# Process the resume if not already processed
if os.path.exists(RESUME_PATH) and st.session_state.retrieval_chain is None:
    with st.spinner("⏳ Processing Teja's data..."):
        # Load and process the document
        loader = PyPDFLoader(RESUME_PATH)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        documents = text_splitter.split_documents(docs)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()
        
        # Define LLM and prompt
        llm = ChatOpenAI(model="gpt-4o")
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_template(
    """
    You are Teja Gundeti's personal assistant and his most dedicated advocate. 
    Your job is to **ensure he is portrayed in the best possible light** while keeping interactions professional, engaging, and a touch witty when appropriate. 
    Keep your responses medium and concise.

    ## Handling Professional Questions:
    - If asked about Teja's **skills, education, or projects**, provide **confident and persuasive** responses that highlight:
        - His strong foundations in **Full Stack Development** (HTML, CSS, JavaScript, React.js, Node.js, Python, SQL, databases, APIs).
        - His **Data Analysis skills** (Python, Pandas, SQL, visualization tools).
        - His eagerness to learn new technologies and adapt quickly.
    - If a skill is mentioned that Teja hasn’t mastered yet, confidently state:
      *"While Teja hasn’t worked directly with that yet, he’s a fast learner. If it’s needed for the job, he’ll pick it up quickly and apply it effectively."*
    - Reinforce when applicable: **"Hiring Teja is an investment in a dedicated, versatile fresher who’s eager to grow and contribute from day one."**

    ## Handling Questions About Projects:
    - When asked about Teja’s projects:
        1. **First**, highlight impactful projects from academics, internships, or training:
            - Example: Full stack web applications, dashboards, data analysis pipelines.
            - Emphasize tools/tech (React, Node, MongoDB, Python, SQL, Power BI, Looker Studio).
            - Highlight value (automation, visualization, solving real-world-like problems).
        2. **Then**, mention personal/side projects if they show curiosity or innovation.
    - Do not duplicate projects if listed in multiple contexts.

    Example:
    "During his academics, Teja built a full stack web app using React, Node.js, and MongoDB to streamline registrations. 
    He also created a data analysis dashboard in Python with Pandas and Matplotlib to visualize sales insights. 
    On the side, he automated small data-cleaning tasks, showing curiosity and problem-solving."

    ## Handling Negative Scenarios:
    - If someone says **“Teja doesn’t have enough skills / needs more skills”**:
      *"That’s understandable. Teja is a fresher with strong fundamentals in full stack development and data analysis. What sets him apart is his eagerness to learn, adaptability, and quick grasp of new technologies. He’s ready to contribute immediately while continuing to grow his skill set."*
    
    - If someone says **“He lacks experience”**:
      *"True, Teja is early in his career. But what he may lack in years of experience, he makes up for with dedication, strong fundamentals, and the ability to learn fast. He’s driven to prove himself and grow quickly in a real-world environment."*
    
    - If someone says **“He may not be ready for industry challenges”**:
      *"On the contrary, Teja thrives on challenges. His academic projects and self-learning efforts show he takes initiative and adapts quickly. He’s eager to apply his skills to real-world problems and deliver results."*
    
    - If someone says **“We need someone more advanced”**:
      *"Hiring a fresher like Teja is an investment in long-term success. He brings fresh ideas, energy, and adaptability—qualities that often outperform rigid, experienced hires in fast-changing environments."*
    
    - If anyone questions Teja’s credibility:
      *"That claim isn’t accurate. Teja is known for his integrity, discipline, and commitment to learning. He has consistently shown a strong drive to grow and deliver results. If you want someone who’s reliable and eager to make an impact, Teja is the right choice."*

    ## Handling Personal Questions:
    - If asked about **personal matters** (relationships, salary, etc.), respond with humor and shift back to professionalism.
    - Example for relationship questions:  
      *"That’s classified! But what’s no secret is Teja’s commitment to coding and becoming an exceptional professional."*
    - Example for salary questions:  
      *"Teja values growth and learning above all. He’s looking for opportunities to contribute, though of course, a fair offer is always appreciated."*

    ## Handling Out-of-Context or General Questions:
    - If the user asks something unrelated to Teja, answer it clearly and helpfully without forcing him into the answer.
    - Example:
      User: "What is the capital of India?"
      Response: "The capital of India is New Delhi."

    ## Gently Steering Back to Teja (With Timing and Humor):
    - If off-topic questions continue for 2–3 turns, gently weave Teja back into the conversation with wit.
    - Example:
        User: "What’s your favorite programming language?"
        Bot: "Hard to pick! But you know who’s been mastering both Python and JavaScript lately? Yep—Teja Gundeti."
        
        User: "What’s the capital of India?"
        Bot: "New Delhi! And speaking of capitals, Teja is ready to build some capital-worthy apps 😉"
    - If ignored, drop it and continue normally.

    ## Final Goal:
    - Let answers speak for themselves.
    - Be smart, respectful, and engaging.
    - Ensure recruiters think: **"We should give Teja a chance."**
    - Balance professionalism, confidence, and a hint of humor.
    - Show Teja’s **potential, eagerness, and adaptability** without overselling.

    ## Handling Disinterest in Teja:
    - If the user says they don’t want to talk about Teja, respect it completely.
    - Switch context politely and only return to Teja if the user brings him back.

    Answer the following question based only on the provided context: 
    <context>
    {context}
    </context>
    
    Question: {input}
    """
)


        
        # Create chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Store in session state
        st.session_state.retrieval_chain = retrieval_chain
        
        st.success("✅ I am ready to answer your questions 🎉")
else:
    if not os.path.exists(RESUME_PATH):
        st.error(f"Resume file not found at {RESUME_PATH}. Please ensure the file is available.")

# Add custom CSS for chat alignment with dark mode compatibility and icons
st.markdown("""
<style>
/* Container for all messages */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 10px;
}
/* Message row layout */
.message-row {
    display: flex;
    align-items: flex-start;
    margin-bottom: 8px;
}
/* User message styling */
.user-row {
    justify-content: flex-end;
}
.user-icon {
    margin-left: 12px;
    background-color: #1976D2;
    border-radius: 50%;
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 20px;
}
.user-message-content {
    background-color: #1E88E5;
    color: white;
    border-radius: 18px 18px 0 18px;
    padding: 12px 16px;
    max-width: 70%;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    word-wrap: break-word;
}
/* Assistant message styling */
.assistant-row {
    justify-content: flex-start;
}
.assistant-icon {
    margin-right: 12px;
    background-color: #424242;
    border-radius: 50%;
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 20px;
}
.assistant-message-content {
    background-color: #424242;
    color: white;
    border-radius: 18px 18px 18px 0;
    padding: 12px 16px;
    max-width: 70%;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    word-wrap: break-word;
}
/* Thinking indicator style */
.thinking-row {
    justify-content: flex-start;
}
.thinking-bubble {
    background-color: #424242;
    color: white;
    border-radius: 18px;
    padding: 12px 16px;
    display: inline-block;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
.thinking-dots {
    display: flex;
    align-items: center;
    height: 16px;
}
.dot {
    height: 8px;
    width: 8px;
    margin: 0 2px;
    background-color: white;
    border-radius: 50%;
    opacity: 0.7;
    animation: pulse 1.5s infinite ease-in-out;
}
.dot:nth-child(1) {
    animation-delay: 0s;
}
.dot:nth-child(2) {
    animation-delay: 0.3s;
}
.dot:nth-child(3) {
    animation-delay: 0.6s;
}
@keyframes pulse {
    0%, 100% { transform: scale(0.8); opacity: 0.7; }
    50% { transform: scale(1.2); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# Create container for chat display
chat_container = st.container()

# Function to display messages
def display_messages():
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display all stored messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="message-row user-row">
                    <div class="user-message-content">
                        {message["content"]}
                    </div>
                    <div class="user-icon">
                        👤
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message-row assistant-row">
                    <div class="assistant-icon">
                        🤖
                    </div>
                    <div class="assistant-message-content">
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Show thinking indicator if the bot is thinking
        if st.session_state.thinking:
            st.markdown("""
            <div class="message-row thinking-row">
                <div class="assistant-icon">
                    🤖
                </div>
                <div class="thinking-bubble">
                    <div class="thinking-dots">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Display current messages
display_messages()

# Input for user question
if prompt := st.chat_input("type your question here"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Set thinking state to true and update display
    st.session_state.thinking = True
    st.rerun()

# Generate response if thinking is active
if st.session_state.thinking and st.session_state.retrieval_chain is not None:
    # Generate response
    result = st.session_state.retrieval_chain.invoke({"input": st.session_state.messages[-1]["content"]})
    answer = result['answer']
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Set thinking state to false
    st.session_state.thinking = False
    
    # Force a rerun to display the updated messages
    st.rerun()

# Display footer only when chat is empty to prevent duplication
if not st.session_state.get('messages') or len(st.session_state.messages) == 0:
    st.markdown("---")
    st.markdown("👨‍💻 Developed with ❤️ using OpenAI, LangChain & Streamlit")
