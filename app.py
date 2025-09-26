"""
HealthBuddy Streamlit App
Beginner-friendly version with no classes!
"""

import streamlit as st
import os
from healthbuddy import ask_healthbuddy, get_all_doctors, add_new_doctor, get_api_keys_status, show_react_agent_workflow

# Page setup
st.set_page_config(
    page_title="HealthBuddy - AI Healthcare Assistant",
    page_icon="🏥",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .big-font {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main app function"""
    
    # Title
    st.markdown('<h1 class="big-font">🏥 HealthBuddy - AI Healthcare Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Your AI healthcare assistant")
    
    # Sidebar for status
    with st.sidebar:
        st.markdown("### 📊 Status")
        
        # Check API keys status
        keys_configured, keys_message = get_api_keys_status()
        
        if not keys_configured:
            st.error(f"❌ {keys_message}")
            st.info("Please update your API keys in `.streamlit/secrets.toml`")
            st.code("""
# Add your API keys to .streamlit/secrets.toml
OPENAI_API_KEY = "your_actual_openai_key"
TAVILY_API_KEY = "your_actual_tavily_key"
            """)
        else:
            st.success(f"✅ {keys_message}")
            st.info("HealthBuddy is ready to help!")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "👨‍⚕️ Doctors", "📚 Examples", "🤖 React Agent"])
    
    with tab1:
        st.markdown("### 💬 Chat with HealthBuddy")
        
        # Chat input
        user_question = st.text_area(
            "Ask HealthBuddy anything about health:",
            placeholder="e.g., What are the symptoms of diabetes?",
            height=100
        )
        
        # Simple button approach (like the working simple app)
        if st.button("🤖 Ask HealthBuddy", type="primary"):
            if user_question and user_question.strip():
                try:
                    with st.spinner("🤖 HealthBuddy is thinking..."):
                        answer = ask_healthbuddy(user_question)
                    
                    st.markdown("### 🤖 HealthBuddy's Answer:")
                    st.markdown(answer)
                    
                    # Show tool calling process
                    st.markdown("---")
                    st.markdown("### 🔧 Tool Calling Process:")
                    st.info("💡 **Check the terminal/console to see which tools were called and how the React agent worked!**")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
            else:
                st.warning("⚠️ Please enter a question first!")
        
        # Clear button
        if st.button("🔄 Clear"):
            st.rerun()
    
    with tab2:
        st.markdown("### 👨‍⚕️ Available Doctors")
        
        # Load doctors only once and cache them
        if "doctors" not in st.session_state:
            try:
                st.session_state.doctors = get_all_doctors()
            except Exception as e:
                st.error(f"Error loading doctors: {e}")
                st.session_state.doctors = []
        
        doctors = st.session_state.doctors
        
        for i, doctor in enumerate(doctors):
            with st.expander(f"Dr. {doctor['name']} - {doctor['specialization']}"):
                st.write(f"**Available:** {doctor['available_timings']}")
                st.write(f"**Location:** {doctor['location']}")
                st.write(f"**Contact:** {doctor['contact']}")
        
        # Add new doctor
        st.markdown("### ➕ Add New Doctor")
        
        with st.form("add_doctor"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Doctor Name")
                specialization = st.text_input("Specialization")
            
            with col2:
                timings = st.text_input("Available Timings")
                location = st.text_input("Location")
            
            contact = st.text_input("Contact Email")
            
            if st.form_submit_button("Add Doctor"):
                if name and specialization and timings and location and contact:
                    add_new_doctor(name, specialization, timings, location, contact)
                    st.success(f"✅ Added Dr. {name}")
                    st.rerun()
                else:
                    st.warning("Please fill in all fields")
    
    with tab3:
        st.markdown("### 📚 Example Questions")
        
        st.markdown("Try asking HealthBuddy these questions:")
        
        example_questions = [
            "What are the symptoms of diabetes?",
            "I have chest pain, can you recommend a doctor?",
            "What does research say about exercise and mental health?",
            "I'm feeling anxious, what should I do?",
            "What are the benefits of intermittent fasting?",
            "I have a fever, what type of doctor should I see?"
        ]
        
        for question in example_questions:
            if st.button(f"❓ {question}", key=f"example_{question}"):
                st.session_state.example_question = question
        
        if hasattr(st.session_state, 'example_question'):
            st.markdown(f"**Selected:** {st.session_state.example_question}")
            
            if st.button("🤖 Ask This Question"):
                if st.session_state.example_question:
                    try:
                        with st.spinner("HealthBuddy is thinking..."):
                            answer = ask_healthbuddy(st.session_state.example_question)
                        
                        st.markdown("### 🤖 HealthBuddy's Answer:")
                        st.markdown(answer)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
                else:
                    st.warning("⚠️ Please select a question first!")
    
    with tab4:
        st.markdown("### 🤖 React Agent Workflow")
        st.markdown("See how HealthBuddy's React agent works with LangGraph!")
        
        # Show the workflow diagram
        workflow = show_react_agent_workflow()
        st.markdown(workflow)
        
        st.markdown("---")
        st.markdown("### 🔍 Tool Calling in Action")
        st.markdown("Watch the terminal/console to see which tools are being called!")
        
        # Test the React agent
        st.markdown("#### Test React Agent:")
        test_question = st.text_input(
            "Ask a question to see tool calling in action:",
            placeholder="e.g., I have chest pain, recommend a doctor",
            key="react_test"
        )
        
        if st.button("🚀 Test React Agent", key="test_react"):
            if test_question and test_question.strip():
                try:
                    # Create a placeholder for real-time updates
                    status_placeholder = st.empty()
                    answer_placeholder = st.empty()
                    
                    with st.spinner("🤖 React Agent is working..."):
                        status_placeholder.info("🔧 **React Agent Process:**")
                        status_placeholder.info("1. 🤖 AI is analyzing your question...")
                        status_placeholder.info("2. 🔍 AI is deciding which tools to use...")
                        status_placeholder.info("3. ⚡ AI is calling the selected tools...")
                        status_placeholder.info("4. 📝 AI is generating the final answer...")
                        
                        answer = ask_healthbuddy(test_question)
                    
                    status_placeholder.success("✅ **React Agent completed successfully!**")
                    
                    st.markdown("### 🤖 React Agent Response:")
                    answer_placeholder.markdown(answer)
                    
                    st.markdown("---")
                    st.markdown("### 🔍 Tool Calling Log:")
                    st.info("💡 **Check the terminal/console below for detailed tool calling logs!**")
                    st.code("""
🤖 AI Response: TOOL: search_web
🔍 AI wants to use: search_web
🔍 Calling web search tool...
🔍 Web search results: 3 items
✅ HealthBuddy has an answer!
                    """, language="text")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
            else:
                st.warning("⚠️ Please enter a question first!")


if __name__ == "__main__":
    main()
