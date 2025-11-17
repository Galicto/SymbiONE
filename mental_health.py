import os
import google.generativeai as genai
from textblob import TextBlob
from dotenv import load_dotenv

load_dotenv(override=True)

# Configure Gemini - check for API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and GEMINI_API_KEY != 'your_gemini_api_key_here' and GEMINI_API_KEY.strip():
    genai.configure(api_key=GEMINI_API_KEY)
    print(f"✓ Gemini API configured in mental_health.py")
else:
    print("⚠️ WARNING: GEMINI_API_KEY not configured in mental_health.py")

def analyze_mental_state(user_responses):
    """
    Analyze user's mental state from their text responses
    Returns: stress_level (0-100), indicators (list), suggestions (list)
    """
    
    # Combine all responses
    full_text = " ".join(user_responses)
    
    # 1. Sentiment Analysis using TextBlob
    sentiment = TextBlob(full_text).sentiment
    polarity = sentiment.polarity  # -1 to 1
    
    # 2. Detect stress indicators in text
    stress_keywords = [
        'stressed', 'anxious', 'worried', 'overwhelmed', 'tired', 'exhausted',
        'burnout', 'depressed', 'frustrated', 'hopeless', 'difficult', 'hard',
        'struggling', 'can\'t cope', 'too much', 'pressure', 'scared'
    ]
    
    stress_count = sum(1 for keyword in stress_keywords if keyword in full_text.lower())
    
    # 3. Use Gemini for deeper analysis
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    
    prompt = f"""
    Analyze the following text from a job seeker and identify signs of stress, burnout, or declining mental health.
    
    Text: "{full_text}"
    
    Provide:
    1. A stress level score (0-100)
    2. Key indicators you noticed
    3. Brief, supportive suggestions
    
    Format your response as:
    SCORE: [number]
    INDICATORS: [comma-separated list]
    SUGGESTIONS: [comma-separated list]
    """
    
    try:
        response = model.generate_content(prompt)
        ai_analysis = response.text
        
        # Parse AI response
        lines = ai_analysis.split('\n')
        score = 50  # default
        indicators = []
        suggestions = []
        
        for line in lines:
            if line.startswith('SCORE:'):
                try:
                    score = int(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('INDICATORS:'):
                indicators = [i.strip() for i in line.split(':')[1].split(',')]
            elif line.startswith('SUGGESTIONS:'):
                suggestions = [s.strip() for s in line.split(':')[1].split(',')]
        
    except Exception as e:
        print(f"Gemini API Error: {e}")
        # Fallback to rule-based analysis
        score = min(100, int((1 - polarity) * 50 + stress_count * 10))
        indicators = ["Detected stress keywords in responses"] if stress_count > 0 else ["No major concerns detected"]
        suggestions = ["Consider taking breaks during job search", "Practice mindfulness"]
    
    # Adjust score based on sentiment
    if polarity < -0.3:
        score += 20
    elif polarity > 0.3:
        score -= 10
    
    score = max(0, min(100, score))  # Clamp between 0-100
    
    return {
        'stress_level': score,
        'sentiment_score': polarity,
        'indicators': indicators[:3],  # Top 3
        'suggestions': suggestions[:3],  # Top 3
        'wellness_category': get_wellness_category(score)
    }


def get_wellness_category(score):
    """Categorize wellness based on stress score"""
    if score < 30:
        return "Excellent"
    elif score < 50:
        return "Good"
    elif score < 70:
        return "Moderate Concern"
    else:
        return "High Stress"


def generate_wellness_questions():
    """Generate questions to assess mental health"""
    return [
        "How are you feeling about your job search today?",
        "What has been the most challenging part of looking for a new role?",
        "How would you describe your energy levels lately?",
        "Are you taking time for self-care during this process?"
    ]


def create_conversational_ai():
    """Create and configure the conversational AI model"""
    try:
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        return model
    except Exception as e:
        print(f"Error creating AI model: {e}")
        return None


def get_ai_response(user_message, conversation_history=[], user_context={}):
    """
    Get AI response for conversational interface
    
    Args:
        user_message: Current user message
        conversation_history: List of previous messages [(role, content), ...]
        user_context: User profile data (skills, experience, etc.)
    
    Returns:
        dict with 'response', 'error', and 'updated_history'
    """
    try:
        # Check API key first
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key or api_key == 'your_gemini_api_key_here' or api_key.strip() == '':
            raise Exception("GEMINI_API_KEY is not configured. Please add your Gemini API key to the .env file.")
        
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        
        # Build context-aware prompt
        context_info = ""
        if user_context:
            context_info = f"""
User Profile Context:
- Name: {user_context.get('name', 'User')}
- Skills: {user_context.get('skills', 'Not specified')}
- Experience Level: {user_context.get('experience_level', 'Not specified')}
- Desired Role: {user_context.get('desired_role', 'Not specified')}
- Location: {user_context.get('location', 'Not specified')}
"""
        
        # Detect if user message is in Hindi
        def is_hindi_text(text):
            if not text:
                return False
            # Hindi Unicode range: \u0900-\u097F
            import re
            hindi_pattern = re.compile(r'[\u0900-\u097F]')
            return bool(hindi_pattern.search(text))
        
        is_hindi = is_hindi_text(user_message)
        
        # EASTER EGG: Check if user is asking about creators/developers
        def is_creator_question(text):
            if not text:
                return False
            text_lower = text.lower()
            creator_keywords = [
                'who created you', 'who made you', 'who developed you', 'who built you',
                'who are your creators', 'who are your developers', 'who designed you',
                'created by', 'made by', 'developed by', 'built by',
                'your creator', 'your developer', 'your maker', 'your designer',
                'who are you created by', 'tell me about your creators',
                # Hindi variations
                'तुम्हें किसने बनाया', 'तुम्हें किसने बनाई', 'तुम्हारे निर्माता', 'तुम्हारे डेवलपर',
                'कौन बनाया', 'किसने बनाया', 'कौन बना', 'किसने बना'
            ]
            return any(keyword in text_lower for keyword in creator_keywords)
        
        is_creator_q = is_creator_question(user_message)
        
        # EASTER EGG: Return special response for creator questions
        if is_creator_q:
            if is_hindi:
                easter_egg_response = "मुझे Symbiosis Institute of Technology, Hyderabad के 4 छात्रों ने बनाया है। वे हैं: Raj, Smaran, Pramit, और Meraj। मैं SymbiONE हूँ, और वे मुझे नौकरी चाहने वालों की मदद करने के लिए बनाया गया है!"
            else:
                easter_egg_response = "I was created by 4 students from Symbiosis Institute of Technology, Hyderabad. Their names are: Raj, Smaran, Pramit, and Meraj. I'm SymbiONE, and they built me to help job seekers with career guidance and mental wellness support!"
            
            updated_history = conversation_history + [
                ("user", user_message),
                ("assistant", easter_egg_response)
            ]
            
            return {
                'response': easter_egg_response,
                'error': None,
                'updated_history': updated_history[-10:]
            }
        
        # Build system prompt with language detection
        if is_hindi:
            system_prompt = """आप SymbiONE हैं, नौकरी चाहने वालों की मदद करने के लिए डिज़ाइन किया गया एक सहायक और सहानुभूतिपूर्ण करियर सहायक।
आपकी भूमिका है:
1. करियर मार्गदर्शन और नौकरी खोज सुझाव प्रदान करना
2. नौकरी खोज की प्रक्रिया के दौरान मानसिक कल्याण सहायता प्रदान करना
3. उपयोगकर्ताओं को उनके कौशल और करियर पथ को समझने में मदद करना
4. प्रोत्साहन और व्यावहारिक सलाह प्रदान करना
5. बातचीत करने वाला, दोस्ताना और सहायक होना

उत्तर संक्षिप्त (2-4 वाक्य), प्राकृतिक, बातचीत करने वाला और सहायक रखें। यदि उपयोगकर्ता तनावग्रस्त या अभिभूत लगता है, 
तो उनकी भावनाओं को स्वीकार करें और कोमल सहायता प्रदान करें。

IMPORTANT: Always respond in Hindi (हिंदी) when the user writes in Hindi."""
        else:
            system_prompt = """You are SymbiONE, a helpful and empathetic career assistant designed to help job seekers. 
Your role is to:
1. Provide career guidance and job search tips
2. Offer mental wellness support during the job search process
3. Help users understand their skills and career path
4. Provide encouragement and practical advice
5. Be conversational, friendly, and supportive

Keep responses concise (2-4 sentences), natural, conversational, and helpful. If the user seems stressed or overwhelmed, 
acknowledge their feelings and offer gentle support.

IMPORTANT: Always respond in the same language as the user's input. If the user writes in Hindi, respond in Hindi."""
        
        # Build the full prompt with context and history
        if conversation_history:
            # Build conversation context from history
            history_messages = []
            for role, content in conversation_history[-6:]:  # Last 6 messages for better context
                if role == 'user':
                    history_messages.append(f"User: {content}")
                elif role == 'assistant':
                    history_messages.append(f"Assistant: {content}")
            
            history_context = "\n".join(history_messages)
            prompt = f"""{system_prompt}

{context_info}

Previous conversation:
{history_context}

User: {user_message}
Assistant:"""
        else:
            # First message, no history
            prompt = f"""{system_prompt}

{context_info}

User: {user_message}
Assistant:"""
        
        print(f"\n=== Non-streaming request to Gemini ===")
        print(f"Prompt length: {len(prompt)} chars")
        
        # Generate response
        response = model.generate_content(prompt)
        
        print(f"Response object type: {type(response)}")
        print(f"Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
        
        # Handle different response formats
        ai_response = None
        
        # Method 1: Try direct text attribute (most common)
        if hasattr(response, 'text') and response.text:
            ai_response = response.text.strip()
            print(f"Got response via .text attribute: {len(ai_response)} chars")
        # Method 2: Try parts directly
        elif hasattr(response, 'parts') and response.parts:
            for part in response.parts:
                if hasattr(part, 'text') and part.text:
                    ai_response = part.text.strip()
                    print(f"Got response via .parts: {len(ai_response)} chars")
                    break
        # Method 3: Try candidates structure (newer API versions)
        elif hasattr(response, 'candidates') and response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            ai_response = part.text.strip()
                            print(f"Got response via .candidates: {len(ai_response)} chars")
                            break
        
        # Fallback if no response found
        if not ai_response or not ai_response.strip():
            print("WARNING: No response extracted from Gemini API")
            ai_response = "I apologize, but I didn't receive a proper response from the AI. This might be due to an API configuration issue. Please check your GEMINI_API_KEY and try again."
        else:
            print(f"Successfully extracted response: {ai_response[:100]}...")
        
        # Update conversation history
        updated_history = conversation_history + [
            ("user", user_message),
            ("assistant", ai_response)
        ]
        
        # Keep only last 10 messages to avoid token limits
        updated_history = updated_history[-10:]
        
        return {
            'response': ai_response,
            'error': None,
            'updated_history': updated_history
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"AI Response Error: {e}")
        print(f"Error details: {error_details}")
        
        # Provide more helpful error messages
        if 'API key' in str(e).lower() or 'authentication' in str(e).lower():
            error_msg = "API configuration error. Please check your GEMINI_API_KEY."
        elif 'quota' in str(e).lower() or 'limit' in str(e).lower():
            error_msg = "API quota exceeded. Please check your Gemini API usage limits."
        else:
            error_msg = "I apologize, but I'm having trouble processing that right now. Could you try rephrasing?"
        
        return {
            'response': error_msg,
            'error': str(e),
            'updated_history': conversation_history
        }