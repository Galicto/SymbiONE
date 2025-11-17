from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
import json
import os
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
from serpapi import GoogleSearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables explicitly from project directory
load_dotenv(override=True)  # Override any existing env vars

# Verify critical API keys are loaded
GEMINI_API_KEY_ENV = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY_ENV or GEMINI_API_KEY_ENV == 'your_gemini_api_key_here':
    print("⚠️ WARNING: GEMINI_API_KEY not found or set to placeholder value!")
    print("Please set GEMINI_API_KEY in your .env file")
    print("Get your key from: https://makersuite.google.com/app/apikey")
else:
    print(f"✓ GEMINI_API_KEY loaded successfully (length: {len(GEMINI_API_KEY_ENV)})")

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')



def calculate_education_score(user_education, job_description):
    """Match user education level with job requirements"""
    education_mapping = {
        'phd': 4,
        'masters': 3,
        'bachelors': 2,
        'high_school': 1
    }
    
    user_level = education_mapping.get(user_education, 2)
    
    # Check what education job requires
    desc_lower = job_description.lower()
    
    required_level = 0
    if 'phd' in desc_lower or 'doctorate' in desc_lower:
        required_level = 4
    elif 'master' in desc_lower or 'msc' in desc_lower or 'mba' in desc_lower:
        required_level = 3
    elif 'bachelor' in desc_lower or 'bsc' in desc_lower or 'btech' in desc_lower:
        required_level = 2
    else:
        required_level = 1  # No specific requirement
    
    # Score based on match
    if user_level >= required_level:
        return 1.0  # Perfect match or overqualified
    elif user_level == required_level - 1:
        return 0.7  # Close match
    else:
        return 0.3  # Under-qualified


def calculate_experience_score(user_exp_level, user_years, job_description, job_title):
    """Match experience level with job requirements"""
    exp_mapping = {
        'entry': (0, 2),
        'mid': (3, 5),
        'senior': (6, 10),
        'lead': (10, 20)
    }
    
    user_range = exp_mapping.get(user_exp_level, (0, 2))
    user_years = int(user_years) if user_years else 0
    
    # Check job level indicators
    combined_text = f"{job_title} {job_description}".lower()
    
    if any(term in combined_text for term in ['senior', 'sr.', 'lead', 'principal', 'staff']):
        required_min = 5
    elif any(term in combined_text for term in ['mid-level', 'intermediate', 'experienced']):
        required_min = 3
    elif any(term in combined_text for term in ['junior', 'entry', 'fresher', 'graduate']):
        required_min = 0
    else:
        required_min = 2  # Default mid-range
    
    # Score calculation
    if user_years >= required_min:
        if user_years <= required_min + 5:
            return 1.0  # Perfect range
        else:
            return 0.8  # Overqualified but acceptable
    elif user_years >= required_min - 2:
        return 0.6  # Slightly under but close
    else:
        return 0.3  # Significantly under-qualified


def calculate_title_relevance(desired_role, job_title):
    """Calculate how well the job title matches user's desired role"""
    if not desired_role:
        return 0.5  # Neutral if no preference
    
    desired_lower = desired_role.lower().split()
    title_lower = job_title.lower()
    
    # Exact match
    if desired_role.lower() in title_lower:
        return 1.0
    
    # Partial match - count word overlaps
    matches = sum(1 for word in desired_lower if word in title_lower and len(word) > 2)
    
    if matches >= len(desired_lower):
        return 0.9
    elif matches >= len(desired_lower) / 2:
        return 0.7
    elif matches > 0:
        return 0.5
    else:
        return 0.2


def rank_jobs_structured_advanced(jobs, user):
    """Advanced structured matching with multiple factors"""
    user_skills = set(user.get('skills', '').lower().split(','))
    
    for job in jobs:
        description = ' '.join(job.get('description', [])).lower()
        title = job.get('title', '').lower()
        
        # Factor 1: Skills matching
        skill_matches = sum(1 for skill in user_skills if skill.strip() and (skill.strip() in description or skill.strip() in title))
        skill_score = min(skill_matches / max(len(user_skills), 1), 1.0)
        
        # Factor 2: Education matching
        edu_score = calculate_education_score(user.get('education', 'bachelors'), description)
        
        # Factor 3: Experience matching
        exp_score = calculate_experience_score(
            user.get('experience_level', 'entry'),
            user.get('experience', 0),
            description,
            title
        )
        
        # Factor 4: Job title relevance
        title_score = calculate_title_relevance(user.get('desired_role', ''), job.get('title', ''))
        
        # Store individual scores for transparency
        job['skill_score'] = skill_score
        job['edu_score'] = edu_score
        job['exp_score'] = exp_score
        job['title_score'] = title_score
        
        # Weighted combined score
        job['structured_score'] = (
            skill_score * 0.40 +      # 40% weight on skills
            title_score * 0.30 +       # 30% weight on title match
            exp_score * 0.20 +         # 20% weight on experience
            edu_score * 0.10           # 10% weight on education
        )
    
    return jobs


def rank_jobs_resume(jobs, user):
    """Hybrid resume matching - unchanged from before"""
    resume_text = user.get('resume_text', '').strip().lower()
    
    if not resume_text or len(resume_text) < 50:
        print("No valid resume text.")
        for job in jobs:
            job['resume_score'] = 0.0
        return jobs
    
    print(f"\n=== Resume Matching ===")
    print(f"Resume: {len(resume_text)} chars")
    
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 
                 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would'}
    
    resume_words = set(word for word in resume_text.split() if word not in stopwords and len(word) > 2)
    
    tech_terms = ['python', 'java', 'javascript', 'typescript', 'c++', 'sql', 'react', 
                  'angular', 'vue', 'django', 'flask', 'node', 'aws', 'azure', 'docker', 
                  'kubernetes', 'machine learning', 'deep learning', 'ai', 'ml', 'nlp', 
                  'tensorflow', 'pytorch', 'pandas', 'numpy']
    
    for job in jobs:
        description = ' '.join(job.get('description', [])).lower()
        title = job.get('title', '').lower()
        full_job_text = f"{title} {description}"
        
        job_words = set(word for word in full_job_text.split() if word not in stopwords and len(word) > 2)
        overlap = resume_words.intersection(job_words)
        overlap_score = min(len(overlap) / max(len(resume_words) * 0.15, 1), 1.0)
        
        tech_matches = sum(1 for term in tech_terms if term in resume_text and term in full_job_text)
        tech_score = min(tech_matches / 5.0, 1.0)
        
        combined_score = (overlap_score * 0.4) + (tech_score * 0.6)
        job['resume_score'] = min(combined_score, 1.0)
    
    print("=== End Resume Matching ===\n")
    return jobs


@app.route('/recommendations')
def get_recommendations():
    user = session.get('user', {})
    if not user:
        return redirect(url_for('register'))
    
    search_query = user.get('desired_role', '') or user.get('skills', 'software engineer')
    location = user.get('location', 'India')
    
    print(f"\n{'='*60}")
    print(f"SEARCH: '{search_query}' in '{location}'")
    print(f"{'='*60}")
    
    jobs = fetch_jobs(search_query, location)
    
    if not jobs:
        return render_template('results_tfidf.html', jobs=[], user=user)

    print(f"Found {len(jobs)} jobs\n")
    
    # Apply both ranking algorithms
    jobs = rank_jobs_structured_advanced(jobs, user)
    jobs = rank_jobs_resume(jobs, user)
    
    # Calculate final blended score
    has_resume = len(user.get('resume_text', '').strip()) > 50
    
    print(f"=== FINAL SCORING ===")
    
    for job in jobs:
        struct_score = job.get('structured_score', 0.0)
        resume_score = job.get('resume_score', 0.0)
        
        if has_resume and resume_score > 0:
            # Blend: 50% structured, 50% resume
            final = (struct_score * 0.5) + (resume_score * 0.5)
        else:
            final = struct_score
        
        job['final_score'] = final
        
        # Create detailed match reason
        job['match_reason'] = (
            f"Overall: {final*100:.1f}% "
            f"(Skills: {job.get('skill_score', 0)*100:.0f}%, "
            f"Title: {job.get('title_score', 0)*100:.0f}%, "
            f"Resume: {resume_score*100:.0f}%)"
        )
    
    # Sort by final score
    jobs.sort(key=lambda x: x['final_score'], reverse=True)
    
    print(f"\n{'='*60}")
    print("TOP 5 RECOMMENDATIONS:")
    print(f"{'='*60}")
    for i, job in enumerate(jobs[:5], 1):
        print(f"{i}. {job['title'][:45]:45} | {job['final_score']*100:5.1f}%")
        print(f"   Skills: {job.get('skill_score', 0)*100:.0f}%, Title: {job.get('title_score', 0)*100:.0f}%, "
              f"Exp: {job.get('exp_score', 0)*100:.0f}%, Resume: {job.get('resume_score', 0)*100:.0f}%")
    print(f"{'='*60}\n")
    
    # Generate resume improvement suggestions
    print("Generating resume improvement suggestions...")
    resume_improvements = generate_resume_improvements(user, jobs)
    
    # Store in session for download
    session['last_recommendations'] = jobs
    session['last_resume_improvements'] = resume_improvements
    
    return render_template('results_tfidf.html', jobs=jobs, user=user, resume_improvements=resume_improvements)

# --- (The home(), register(), and fetch_jobs() functions remain the same) ---
@app.route('/')
def home():
    return render_template('index.html')

def fetch_jobs(skills, location):
    if not SERPAPI_API_KEY:
        print("SerpApi API key not found!")
        return []
    params = {
        "engine": "google_jobs", "q": skills, "location": location, "api_key": SERPAPI_API_KEY
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        return results.get("jobs_results", [])
    except Exception as e:
        print(f"An error occurred with SerpApi: {e}")
        return []

# --- NEW rank_jobs function using TF-IDF ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        resume_file = request.files.get('resume')
        resume_text = ""
        if resume_file and resume_file.filename.endswith('.pdf'):
            try:
                pdf_reader = PyPDF2.PdfReader(resume_file.stream)
                text_parts = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
                resume_text = "\n".join(text_parts)
                # print(resume_text)
            except Exception as e:
                print(f"Error reading PDF: {e}")
                resume_text = ""
        
        manual_skills = request.form.get('skills', '').lower()
        
        # Store all the new fields
        session['user'] = {
            'name': request.form.get('name'),
            'email': request.form.get('email'),
            'skills': manual_skills,
            'education': request.form.get('education'),
            'experience_level': request.form.get('experience_level'),
            'desired_role': request.form.get('desired_role', ''),
            'location': request.form.get('location'),
            'experience': request.form.get('experience'),
            'resume_text': resume_text.lower()
        }
        return redirect(url_for('get_recommendations'))
    return render_template('register.html')


from mental_health import analyze_mental_state, generate_wellness_questions, get_ai_response
import google.generativeai as genai
from datetime import datetime

# Configure Gemini API for streaming
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY and GEMINI_API_KEY != 'your_gemini_api_key_here':
    genai.configure(api_key=GEMINI_API_KEY)
    print(f"✓ Gemini API configured (key length: {len(GEMINI_API_KEY)})")
else:
    print("⚠️ WARNING: GEMINI_API_KEY not properly configured!")

def generate_resume_improvements(user, jobs):
    """
    Generate AI-powered resume improvement suggestions based on user profile and job matches
    """
    try:
        if not GEMINI_API_KEY or GEMINI_API_KEY == 'your_gemini_api_key_here':
            return None
        
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        
        # Get user profile information
        user_skills = user.get('skills', 'Not specified')
        user_experience = user.get('experience_level', 'Not specified')
        desired_role = user.get('desired_role', 'Not specified')
        resume_text = user.get('resume_text', '')
        user_education = user.get('education', 'Not specified')
        
        # Analyze top job requirements
        top_jobs = jobs[:5] if jobs else []
        common_skills = []
        common_requirements = []
        
        for job in top_jobs:
            if job.get('description'):
                desc_text = ' '.join(job['description']) if isinstance(job['description'], list) else str(job['description'])
                common_requirements.append(f"{job.get('title', 'Unknown')} at {job.get('company_name', 'Unknown')}: {desc_text[:200]}")
        
        # Build prompt for Gemini
        prompt = f"""You are a professional career advisor and resume expert. Analyze the following job seeker's profile and provide specific, actionable resume improvement suggestions.

User Profile:
- Name: {user.get('name', 'User')}
- Skills: {user_skills}
- Experience Level: {user_experience}
- Desired Role: {desired_role}
- Education: {user_education}
- Resume Content: {resume_text[:500] if resume_text else 'No resume uploaded'}

Top Job Requirements (from matching jobs):
{chr(10).join(common_requirements[:3])}

Please provide:
1. **Specific Skills to Add**: List 3-5 skills that appear frequently in job postings but are missing from the resume
2. **Resume Structure Improvements**: 2-3 suggestions for better formatting, sections, or organization
3. **Content Enhancements**: 2-3 specific ways to strengthen achievements, quantify results, or highlight relevant experience
4. **Keyword Optimization**: Important keywords to include based on the target roles
5. **Overall Tips**: 2-3 general resume improvement tips

Format your response in clear sections with bullet points. Be specific and actionable.

Resume Improvement Suggestions:
"""
        
        response = model.generate_content(prompt)
        
        # Format response as HTML-friendly text with line breaks
        
        # Extract text from response
        improvements = None
        if hasattr(response, 'text') and response.text:
            improvements = response.text.strip()
        elif hasattr(response, 'parts') and response.parts:
            improvements = response.parts[0].text.strip() if response.parts else "Unable to generate suggestions."
        else:
            improvements = "Unable to generate suggestions at this time."
        
        # Convert markdown-style formatting to HTML-friendly format
        if improvements:
            # Replace markdown bold with HTML
            import re
            improvements = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', improvements)
            # Ensure line breaks are preserved
            improvements = improvements.replace('\n', '<br>')
        
        return improvements
        
    except Exception as e:
        print(f"Error generating resume improvements: {e}")
        import traceback
        traceback.print_exc()
        return None

# ... (your existing imports and code)

@app.route('/wellness')
def wellness_checkin():
    """Wellness check-in page"""
    questions = generate_wellness_questions()
    return render_template('wellness.html', questions=questions)


@app.route('/wellness/analyze', methods=['POST'])
def analyze_wellness():
    """Analyze user's wellness responses"""
    responses = []
    for i in range(4):  # 4 questions
        response = request.form.get(f'question_{i}', '')
        if response:
            responses.append(response)
    
    if not responses:
        return redirect(url_for('wellness_checkin'))
    
    # Analyze mental state
    analysis = analyze_mental_state(responses)
    
    # Store in session
    session['wellness_data'] = analysis
    
    return redirect(url_for('wellness_results'))


@app.route('/wellness/results')
def wellness_results():
    """Show wellness analysis results"""
    analysis = session.get('wellness_data', {})
    user = session.get('user', {})
    
    return render_template('wellness_results.html', analysis=analysis, user=user)


@app.route('/conversation')
def conversation():
    """Conversational AI interface"""
    # Initialize conversation history if not exists
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    
    user = session.get('user', {})
    history = session.get('conversation_history', [])
    
    return render_template('conversation.html', user=user, conversation_history=history)


@app.route('/conversation/send', methods=['POST'])
def send_message():
    """Handle conversational AI message (non-streaming fallback)"""
    user_message = request.json.get('message', '').strip()
    
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400
    
    # Get conversation history and user context
    conversation_history = session.get('conversation_history', [])
    user_context = session.get('user', {})
    
    # Get AI response
    result = get_ai_response(user_message, conversation_history, user_context)
    
    # Update session with new history
    session['conversation_history'] = result['updated_history']
    
    return jsonify({
        'response': result['response'],
        'error': result.get('error')
    })


@app.route('/conversation/stream', methods=['POST'])
def stream_message():
    """Handle streaming conversational AI message using SSE"""
    user_message = request.json.get('message', '').strip()
    
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400
    
    # Get conversation history and user context
    conversation_history = session.get('conversation_history', [])
    user_context = session.get('user', {})
    
    def generate():
        try:
            # Create streaming response generator
            full_response = ""
            
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
            
            # EASTER EGG: Special response for creator questions
            if is_creator_q:
                if is_hindi:
                    easter_egg_response = "मुझे Symbiosis Institute of Technology, Hyderabad के 4 छात्रों ने बनाया है। वे हैं: Raj, Smaran, Pramit, और Meraj। मैं SymbiONE हूँ, और वे मुझे नौकरी चाहने वालों की मदद करने के लिए बनाया गया है!"
                else:
                    easter_egg_response = "I was created by 4 students from Symbiosis Institute of Technology, Hyderabad. Their names are: Raj, Smaran, Pramit, and Meraj. I'm SymbiONE, and they built me to help job seekers with career guidance and mental wellness support!"
                
                # Stream the easter egg response immediately
                full_response = easter_egg_response
                yield f"data: {json.dumps({'chunk': easter_egg_response, 'done': False})}\n\n"
                
                # Update session history
                try:
                    updated_history = conversation_history + [
                        ("user", user_message),
                        ("assistant", easter_egg_response)
                    ]
                    session['conversation_history'] = updated_history[-10:]
                    session.modified = True
                except Exception as session_error:
                    print(f"Session update error: {session_error}")
                
                yield f"data: {json.dumps({'chunk': '', 'done': True, 'full_response': easter_egg_response})}\n\n"
                return
            
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
तो उनकी भावनाओं को स्वीकार करें और कोमल सहायता प्रदान करें।

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
            
            # Check API key first
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key or api_key == 'your_gemini_api_key_here' or api_key.strip() == '':
                error_msg = "GEMINI_API_KEY is not configured. Please add your Gemini API key to the .env file.\nGet your API key from: https://makersuite.google.com/app/apikey"
                raise Exception(error_msg)
            
            # Use Gemini with streaming
            print(f"\n=== Sending to Gemini ===")
            print(f"User message: {user_message[:100]}...")
            print(f"Prompt length: {len(prompt)} chars")
            print(f"API key present: {'Yes' if api_key else 'No'}")
            
            model = genai.GenerativeModel('models/gemini-2.5-flash')
            response = model.generate_content(prompt, stream=True)
            
            print(f"Response object: {type(response)}")
            
            # Stream chunks as they arrive
            chunk_count = 0
            for chunk in response:
                try:
                    # Direct text access (as per API documentation)
                    chunk_text = None
                    
                    # Method 1: Direct text attribute (standard way)
                    if hasattr(chunk, 'text') and chunk.text:
                        chunk_text = str(chunk.text).strip()
                    # Method 2: Try parts
                    elif hasattr(chunk, 'parts') and chunk.parts:
                        for part in chunk.parts:
                            if hasattr(part, 'text') and part.text:
                                chunk_text = str(part.text).strip()
                                break
                    # Method 3: Try candidates structure
                    elif hasattr(chunk, 'candidates') and chunk.candidates and len(chunk.candidates) > 0:
                        candidate = chunk.candidates[0]
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        chunk_text = str(part.text).strip()
                                        break
                    
                    # Process chunk text if found
                    if chunk_text and chunk_text != '':
                        full_response += chunk_text
                        chunk_count += 1
                        # Send each chunk as SSE
                        yield f"data: {json.dumps({'chunk': chunk_text, 'done': False})}\n\n"
                        print(f"Sent chunk {chunk_count}: {chunk_text[:50]}...")
                    else:
                        print(f"Skipped chunk - no text found. Chunk type: {type(chunk)}")
                except Exception as chunk_error:
                    import traceback
                    print(f"Error processing chunk: {chunk_error}")
                    print(traceback.format_exc())
                    continue
            
            print(f"Total chunks received: {chunk_count}")
            print(f"Full response length: {len(full_response)} chars")
            
            # Validate response
            if not full_response or not full_response.strip():
                if chunk_count == 0:
                    full_response = "I apologize, but I didn't receive a response from the AI. Please check: 1) Your GEMINI_API_KEY in .env file, 2) Your internet connection, 3) Try again in a moment."
                else:
                    full_response = "I received chunks but the response appears empty. Please try rephrasing your question."
                print(f"WARNING: Empty response after {chunk_count} chunks")
            
            # Finalize response
            final_response = full_response.strip()
            
            # Update session with new history BEFORE yielding completion
            # (Session must be modified before the generator completes)
            try:
                updated_history = conversation_history + [
                    ("user", user_message),
                    ("assistant", final_response)
                ]
                session['conversation_history'] = updated_history[-10:]
                session.modified = True
                print(f"Updated conversation history: {len(session['conversation_history'])} messages")
            except Exception as session_error:
                print(f"Session update error: {session_error}")
            
            # Send completion signal with full response
            yield f"data: {json.dumps({'chunk': '', 'done': True, 'full_response': final_response})}\n\n"
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Streaming Error: {e}")
            print(f"Error details: {error_details}")
            
            # Check for specific error types
            error_str = str(e).lower()
            if 'api key' in error_str or 'gemini_api_key' in error_str or 'authentication' in error_str:
                error_msg = "⚠️ API Key Missing: Please add your GEMINI_API_KEY to the .env file.\nGet your key from: https://makersuite.google.com/app/apikey"
            elif 'quota' in error_str or 'limit' in error_str:
                error_msg = "⚠️ API Quota Exceeded: Please check your Gemini API usage limits."
            elif 'not found' in error_str or 'missing' in error_str:
                error_msg = f"⚠️ Configuration Error: {str(e)}"
            else:
                error_msg = f"I apologize, but I'm having trouble processing that right now. Error: {str(e)[:100]}"
            
            yield f"data: {json.dumps({'chunk': error_msg, 'done': True, 'error': str(e)})}\n\n"
    
    # Create response with proper headers for SSE
    response = Response(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Connection'] = 'keep-alive'
    
    return response


@app.route('/conversation/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history"""
    session['conversation_history'] = []
    return jsonify({'status': 'cleared'})

@app.route('/recommendations/download', methods=['GET'])
def download_report():
    """Generate and download job recommendations report as HTML"""
    user = session.get('user', {})
    if not user:
        return redirect(url_for('register'))
    
    # Get jobs from session or regenerate
    jobs = session.get('last_recommendations', [])
    resume_improvements = session.get('last_resume_improvements', '')
    
    if not jobs:
        return redirect(url_for('get_recommendations'))
    
    # Generate HTML report
    report_date = datetime.now().strftime('%B %d, %Y')
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Job Recommendations Report - {user.get('name', 'User')}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #667eea;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #667eea;
            margin: 0;
        }}
        .header p {{
            color: #666;
            margin: 5px 0;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .job-card {{
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-left: 4px solid #667eea;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .job-title {{
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        .job-meta {{
            color: #666;
            margin: 10px 0;
        }}
        .match-score {{
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            display: inline-block;
            margin: 10px 0;
            font-weight: bold;
        }}
        .improvements {{
            background: white;
            padding: 20px;
            border-left: 4px solid #4caf50;
            border-radius: 5px;
            white-space: pre-wrap;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }}
        .stat-item {{
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 10px;
            min-width: 120px;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ddd;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Job Recommendations Report</h1>
        <p><strong>Generated for:</strong> {user.get('name', 'User')}</p>
        <p><strong>Date:</strong> {report_date}</p>
        <p><strong>Target Role:</strong> {user.get('desired_role', 'Not specified')}</p>
        <p><strong>Location:</strong> {user.get('location', 'Not specified')}</p>
    </div>
    
    <div class="section">
        <h2>📊 Summary Statistics</h2>
        <div class="stats">
            <div class="stat-item">
                <div class="stat-number">{len(jobs)}</div>
                <div>Jobs Found</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{jobs[0]['final_score']*100:.0f}%</div>
                <div>Top Match</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{sum(j['final_score'] for j in jobs) / len(jobs) * 100:.0f}%</div>
                <div>Avg Match</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{sum(1 for j in jobs if j['final_score'] > 0.5)}</div>
                <div>Strong Matches</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>💼 Job Recommendations</h2>
"""
    
    for i, job in enumerate(jobs[:10], 1):  # Top 10 jobs in report
        job_desc = ' '.join(job['description'][:2]) if isinstance(job.get('description'), list) else str(job.get('description', ''))[:200]
        html_content += f"""
        <div class="job-card">
            <div class="job-title">{i}. {job.get('title', 'Unknown Position')}</div>
            <div class="job-meta">
                <strong>Company:</strong> {job.get('company_name', 'Unknown')} | 
                <strong>Location:</strong> {job.get('location', 'Unknown')}
            </div>
            <div class="match-score">Match Score: {job['final_score']*100:.1f}%</div>
            <p><strong>Match Breakdown:</strong> {job.get('match_reason', 'N/A')}</p>
            <p>{job_desc}...</p>
        </div>
"""
    
    html_content += """
    </div>
"""
    
    # Add resume improvements section
    if resume_improvements:
        # Escape HTML for safety in downloaded file
        import html
        safe_improvements = html.escape(resume_improvements.replace('<br>', '\n').replace('<strong>', '**').replace('</strong>', '**'))
        html_content += f"""
    <div class="section">
        <h2>📝 Resume Improvement Suggestions</h2>
        <div class="improvements">{safe_improvements}</div>
    </div>
"""
    
    html_content += f"""
    <div class="footer">
        <p>Report generated by SymbiONE</p>
        <p>For more job opportunities, visit: <a href="http://localhost:3000/recommendations">http://localhost:3000/recommendations</a></p>
    </div>
</body>
</html>
"""
    
    # Return as downloadable file
    from flask import Response
    return Response(
        html_content,
        mimetype='text/html',
        headers={
            'Content-Disposition': f'attachment; filename=job_recommendations_{user.get("name", "user")}_{datetime.now().strftime("%Y%m%d")}.html'
        }
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)

