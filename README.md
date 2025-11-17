# 🎯 SymbiONE

**Jobs + Wellness, Powered by AI**

SymbiONE is an intelligent job recommendation platform that matches job seekers with opportunities while monitoring their mental wellness throughout the job search process. Built with Flask and powered by Google Gemini AI, it combines advanced job matching algorithms with empathetic wellness support.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

### 🎯 Smart Job Matching
- **Advanced Job Search**: Integrated with SerpAPI for real-time job listings from Google Jobs
- **Multi-Factor Scoring**: 
  - Skills matching (40% weight)
  - Job title relevance (30% weight)
  - Experience matching (20% weight)
  - Education matching (10% weight)
- **Resume Analysis**: PDF resume parsing with intelligent content extraction
- **Hybrid Ranking**: Combines structured matching with resume similarity for optimal results
- **Detailed Match Breakdown**: View individual scores for each matching factor

### 🧘 Mental Wellness Support
- **Wellness Check-In**: Interactive assessment with text and voice input modes
- **Stress Detection**: AI-powered analysis using sentiment analysis and keyword detection
- **Personalized Recommendations**: Context-aware wellness suggestions based on stress levels
- **Voice Wellness Assessment**: Speak your responses using Web Speech API

### 💬 AI-Powered Voice Conversation
- **Real-Time Streaming**: Live conversation with Gemini AI using Server-Sent Events (SSE)
- **Voice-Only Mode**: Fully voice-based conversation (Voice Input → AI Voice Output)
  - Automatic speech-to-text transcription
  - Real-time AI responses with streaming text
  - Text-to-speech for AI responses
  - Auto-restart listening after AI finishes speaking
- **Continuous Conversation**: Natural back-and-forth dialogue without manual intervention
- **Context-Aware**: Remembers conversation history and user profile
- **Visual Indicators**: Live connection status, streaming indicators, speaking indicators

### 🎨 Modern User Interface
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Beautiful UI**: Gradient themes with smooth animations
- **Interactive Elements**: Real-time updates, typing indicators, smooth scrolling
- **Accessibility**: Voice and text input options for all users

---

## 🛠️ Tech Stack

### Backend
- **Flask 3.0.0**: Web framework
- **Python 3.8+**: Programming language
- **Google Generative AI (Gemini)**: Conversational AI and wellness analysis
- **SerpAPI**: Job search integration
- **PyPDF2**: Resume parsing
- **TextBlob**: Sentiment analysis
- **scikit-learn**: Machine learning utilities

### Frontend
- **HTML5/CSS3**: Modern web standards
- **JavaScript (ES6+)**: Client-side interactivity
- **Web Speech API**: Voice input (Speech Recognition)
- **Web Speech Synthesis API**: Voice output (Text-to-Speech)
- **Fetch API with Streaming**: Real-time AI responses
- **Server-Sent Events (SSE)**: Live streaming from backend

### APIs & Services
- **Google Gemini API**: AI conversations and wellness analysis
- **SerpAPI**: Job search data
- **Web Speech API**: Browser-based speech recognition

---

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Google Chrome or Microsoft Edge (for voice features)
- API Keys:
  - **GEMINI_API_KEY**: Google Gemini API key
  - **SERPAPI_API_KEY**: SerpAPI key for job search
  - **SECRET_KEY**: Flask session secret key

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd IEEE-Hackathon
```

### 2. Create Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here
SECRET_KEY=your_secret_key_here
```

**Getting API Keys:**
- **Gemini API**: Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **SerpAPI**: Sign up at [SerpAPI](https://serpapi.com/) and get your API key
- **Secret Key**: Generate any random string for Flask sessions (e.g., use `python -c "import secrets; print(secrets.token_hex(32))"`)

### 5. Run the Application
```bash
python app.py
```

The application will start on `http://localhost:3000` (or the port specified in `app.py`).

---

## 📖 Usage Guide

### Job Search Flow

1. **Navigate to Job Search**: Click "Find Jobs" or go to `/register`
2. **Create Profile**: 
   - Enter your name, email, skills, education, experience
   - Upload your PDF resume
   - Select desired role and location
3. **Get Recommendations**: Click "Find My Perfect Jobs"
4. **View Results**: 
   - See job matches with detailed scores
   - Filter by match percentage
   - Sort by relevance, title, or company
   - View detailed breakdown of matching factors

### Wellness Check-In

1. **Access Wellness**: Click "Wellness" in navigation or go to `/wellness`
2. **Choose Input Mode**:
   - **Text Mode**: Type your responses
   - **Voice Mode**: Speak your responses using microphone
3. **Answer Questions**: Respond to 4 wellness questions
4. **View Analysis**: Get stress level, indicators, and personalized suggestions

### AI Conversation (Voice-Only Mode)

1. **Start Conversation**: Click "AI Chat" or go to `/conversation`
2. **Enable Voice-Only Mode**: Check "🎤 Voice-Only Mode" (enabled by default)
3. **Click Microphone**: Start speaking
4. **Natural Conversation**:
   - Speak your question/statement
   - After brief pause, message auto-sends
   - AI responds with streaming text and voice
   - After AI finishes, automatically listens for your next input
   - Continue conversation naturally!

**Voice-Only Mode Features:**
- ✅ Automatic speech-to-text
- ✅ Auto-send after speech detection
- ✅ Real-time streaming AI responses
- ✅ Text-to-speech for AI responses
- ✅ Auto-restart listening after AI speaks
- ✅ Continuous conversation loop

---

## 📁 Project Structure

```
IEEE-Hackathon/
├── app.py                    # Main Flask application
├── mental_health.py          # Wellness analysis and AI conversation module
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this)
├── README.md                 # This file
├── static/
│   └── style.css            # Global styles
└── templates/
    ├── base.html            # Base template with navigation
    ├── index.html           # Landing page
    ├── register.html        # User registration form
    ├── results_tfidf.html   # Job results display
    ├── wellness.html        # Wellness check-in page
    ├── wellness_results.html # Wellness analysis results
    └── conversation.html    # AI conversation interface
```

---

## 🔌 API Endpoints

### Job Search
- `GET /` - Homepage
- `GET /register` - Registration form
- `POST /register` - Process registration and redirect to recommendations
- `GET /recommendations` - Display job recommendations

### Wellness
- `GET /wellness` - Wellness check-in page
- `POST /wellness/analyze` - Analyze wellness responses
- `GET /wellness/results` - Display wellness results

### AI Conversation
- `GET /conversation` - Conversation interface
- `POST /conversation/send` - Send message (non-streaming)
- `POST /conversation/stream` - Stream message (SSE streaming)
- `POST /conversation/clear` - Clear conversation history

---

## 🧩 Key Components

### Job Matching Algorithm

The job matching system uses a multi-factor scoring approach:

1. **Skills Matching** (40%): Matches user skills with job requirements
2. **Title Relevance** (30%): How well job title matches desired role
3. **Experience Matching** (20%): Alignment between user experience and job level
4. **Education Matching** (10%): Educational requirements alignment
5. **Resume Similarity** (Optional 50% blend): Word overlap and tech term matching

Final score combines structured matching with resume analysis when available.

### Wellness Analysis

1. **Sentiment Analysis**: Uses TextBlob for polarity detection
2. **Keyword Detection**: Identifies stress indicators in text
3. **AI Analysis**: Gemini API provides deeper insights
4. **Categorization**: Classifies wellness as Excellent/Good/Moderate Concern/High Stress

### AI Conversation System

1. **Real-Time Streaming**: Server-Sent Events (SSE) for live response streaming
2. **Context Management**: Maintains conversation history (last 10 messages)
3. **User Profile Integration**: Uses user profile data for personalized responses
4. **Voice Pipeline**: Speech → Text → AI → Text → Speech

---

## ⚙️ Configuration

### Port Configuration

Default port is `3000`. To change:

Edit `app.py`:
```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
```

### Voice Recognition

- **Browser Support**: Works best in Chrome/Edge (full Web Speech API support)
- **Language**: Configured for English (en-US)
- **Continuous Mode**: Enabled in voice-only mode for natural conversation

### Streaming Settings

- **Streaming Mode**: Enabled by default
- **Chunk Size**: Real-time chunks from Gemini API
- **Auto-scroll**: Smooth scrolling during streaming

---

## 🔒 Security & Privacy

- **Session Management**: Flask sessions with secret key
- **Environment Variables**: Sensitive keys stored in `.env` (not in version control)
- **Input Validation**: Form validation and sanitization
- **Error Handling**: Graceful error handling with user-friendly messages

**Note**: Ensure `.env` is in `.gitignore` to protect your API keys.

---

## 🐛 Troubleshooting

### Voice Recognition Not Working
- **Issue**: Microphone button not responding
- **Solution**: Use Chrome or Edge browser. Check browser permissions for microphone access.

### API Errors
- **Issue**: "Gemini API Error" or "SerpAPI Error"
- **Solution**: Verify API keys in `.env` file are correct and have sufficient quota.

### Port Already in Use
- **Issue**: "Port 3000 already in use"
- **Solution**: Change port in `app.py` or kill the process using the port:
  ```bash
  lsof -ti:3000 | xargs kill -9
  ```

### Dependencies Not Installing
- **Issue**: scikit-learn installation fails
- **Solution**: The code works without scikit-learn (it's imported but not actively used). You can remove it from requirements.txt or install a newer version:
  ```bash
  pip install scikit-learn --upgrade
  ```

---

## 🚧 Future Enhancements

- [ ] User authentication and persistent profiles
- [ ] Job application tracking
- [ ] Email notifications for new matching jobs
- [ ] Advanced filtering (salary, company size, etc.)
- [ ] Resume builder integration
- [ ] Interview preparation features
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Integration with job boards (LinkedIn, Indeed)
- [ ] Analytics dashboard for job search progress

---

## 📝 License

This project is licensed under the MIT License.

---

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Support

For issues, questions, or suggestions, please open an issue on the repository.

---

## 🙏 Acknowledgments

- **Google Gemini AI**: For powerful conversational AI capabilities
- **SerpAPI**: For job search data integration
- **Flask Community**: For the excellent web framework
- **Web Speech API**: For browser-based speech recognition

---

## 🎯 Project Status

**Version**: 1.0.0  
**Status**: ✅ Active Development  
**Last Updated**: November 2024

---

**Made with ❤️ for job seekers**

