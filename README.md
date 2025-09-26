# 🏥 HealthBuddy - AI Healthcare Assistant

**No classes, just functions!** Perfect for beginners who want to understand AI agents.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your_actual_openai_key"
TAVILY_API_KEY = "your_actual_tavily_key"
```

### 3. Run HealthBuddy
```bash
streamlit run app.py
```

**That's it!** Open your browser and start chatting with HealthBuddy!

## 📁 Files

- `healthbuddy.py` - The AI agent (no classes, just functions!)
- `app.py` - Streamlit web interface
- `requirements.txt` - Dependencies
- `.streamlit/secrets.toml` - Your API keys

## 🎯 What HealthBuddy Does

- 🔍 **Web Search** - Gets current health information
- 📚 **Research Papers** - Finds scientific studies
- 👨‍⚕️ **Doctor Recommendations** - Suggests specialists
- 🤖 **AI Assistant** - Answers health questions intelligently

## 💡 How to Customize

**Add a new doctor:**
```python
from healthbuddy import add_new_doctor
add_new_doctor("Dr. Name", "Specialty", "Hours", "Location", "Email")
```

**Ask questions:**
```python
from healthbuddy import ask_healthbuddy
answer = ask_healthbuddy("What are diabetes symptoms?")
```

That's it! Simple functions, no classes needed.
