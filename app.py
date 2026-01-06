"""
=============================================================================
AI CHAT WEB APP - CLASSROOM DEMO
=============================================================================
This is a simplified educational demo showing how to build a multi-model
AI chat application using Flask, SQLite, and server-side rendering.

IMPORTANT: This uses PLAIN TEXT passwords for simplicity.
           Never do this in production!

Tech Stack:
- Flask: Web framework
- SQLite: Database (file-based, no server needed)
- Server-Side Rendering: No JavaScript, just HTML forms
=============================================================================
"""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
# These are all the libraries we need for our application

import os                          # For file operations and environment variables
import sqlite3                     # For SQLite database operations
from datetime import datetime      # For timestamps on chat messages
from flask import Flask, render_template, request, redirect, url_for, session, flash
# Flask components explained:
# - Flask: Main application class
# - render_template: Render HTML templates with data
# - request: Access form data from POST requests
# - redirect: Redirect users to different pages
# - url_for: Generate URLs for routes
# - session: Store user login state
# - flash: Show temporary messages to users

from dotenv import load_dotenv     # Load environment variables from .env file
import requests                    # Make HTTP requests to APIs
import openai                      # OpenAI official Python client
import google.generativeai as genai  # Google Gemini official Python client


# =============================================================================
# SECTION 2: ENVIRONMENT SETUP
# =============================================================================

# Load environment variables from .env file
# This reads the .env file and makes variables available via os.getenv()
load_dotenv()

# Create Flask application instance
app = Flask(__name__)

# Set secret key for session encryption
# In production, this should be a long random string
# Sessions store user login information in encrypted cookies
app.secret_key = os.getenv('SECRET_KEY', 'demo-secret-key-change-in-production')

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# Configure AI provider clients
# OpenAI client setup
openai.api_key = OPENAI_API_KEY

# Gemini client setup
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# =============================================================================
# SECTION 3: DATABASE SETUP
# =============================================================================

# Database file path - SQLite stores everything in a single file
DATABASE = 'chat_app.db'


def get_db_connection():
    """
    Create and return a connection to the SQLite database.

    SQLite is a file-based database - no server needed!
    Row factory lets us access columns by name like: row['username']
    """
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn


def init_db():
    """
    Initialize the database by creating tables if they don't exist.

    This function creates two tables:
    1. users - Store user accounts (username and PLAIN TEXT password)
    2. messages - Store chat history

    Called automatically when the app starts.
    """
    conn = get_db_connection()

    # Create users table
    # IMPORTANT: Passwords are stored in PLAIN TEXT for this demo
    # In production, you would use password hashing (bcrypt, etc.)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    # Create messages table
    # Stores all chat messages with relationships to users
    conn.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()  # Save changes to database
    conn.close()   # Close connection


# Initialize database when app starts
# This ensures tables exist before any requests are handled
init_db()


# =============================================================================
# SECTION 4: AI PROVIDER FUNCTIONS
# =============================================================================
# These functions handle API calls to different AI providers
# Each provider has a different API format, so we need separate functions


def call_openai_gpt(message, chat_history):
    """
    Call OpenAI GPT API (GPT-4, GPT-3.5, etc.)

    Args:
        message: The user's current message
        chat_history: List of previous messages for context

    Returns:
        AI response as a string
    """
    try:
        # Build conversation history in OpenAI format
        # OpenAI expects messages as: [{"role": "user", "content": "..."}]
        messages = []

        # Add chat history for context (last 10 messages to avoid token limits)
        for msg in chat_history[-10:]:
            messages.append({
                "role": msg['role'],  # 'user' or 'assistant'
                "content": msg['content']
            })

        # Add current user message
        messages.append({
            "role": "user",
            "content": message
        })

        # Make API call to OpenAI
        # Using GPT-4o-mini for cost-effectiveness in a demo
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # You can change this to gpt-4, gpt-3.5-turbo, etc.
            messages=messages,
            max_tokens=1000,       # Limit response length
            temperature=0.7        # Creativity level (0-2, higher = more creative)
        )

        # Extract the response text
        return response.choices[0].message.content

    except Exception as e:
        # If anything goes wrong, return an error message
        return f"Error calling OpenAI: {str(e)}"


def call_gemini(message, chat_history):
    """
    Call Google Gemini API

    Args:
        message: The user's current message
        chat_history: List of previous messages for context

    Returns:
        AI response as a string
    """
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-pro')

        # Build conversation history
        # Gemini uses a different format than OpenAI
        conversation_text = ""

        # Add previous messages for context
        for msg in chat_history[-10:]:
            role_label = "User" if msg['role'] == 'user' else "Assistant"
            conversation_text += f"{role_label}: {msg['content']}\n\n"

        # Add current message
        conversation_text += f"User: {message}\n\nAssistant:"

        # Make API call to Gemini
        response = model.generate_content(conversation_text)

        # Extract the response text
        return response.text

    except Exception as e:
        # If anything goes wrong, return an error message
        return f"Error calling Gemini: {str(e)}"


def call_deepseek(message, chat_history):
    """
    Call DeepSeek API

    DeepSeek uses an OpenAI-compatible API, so the format is similar.

    Args:
        message: The user's current message
        chat_history: List of previous messages for context

    Returns:
        AI response as a string
    """
    try:
        # DeepSeek API endpoint
        url = "https://api.deepseek.com/v1/chat/completions"

        # Build conversation history in OpenAI-compatible format
        messages = []

        # Add chat history for context
        for msg in chat_history[-10:]:
            messages.append({
                "role": msg['role'],
                "content": msg['content']
            })

        # Add current user message
        messages.append({
            "role": "user",
            "content": message
        })

        # Prepare request headers
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }

        # Prepare request body
        data = {
            "model": "deepseek-chat",  # DeepSeek's main chat model
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }

        # Make API call using requests library
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raise exception if request failed

        # Extract the response text
        result = response.json()
        return result['choices'][0]['message']['content']

    except Exception as e:
        # If anything goes wrong, return an error message
        return f"Error calling DeepSeek: {str(e)}"


def get_ai_response(message, model, chat_history):
    """
    Route the message to the appropriate AI provider based on selected model.

    This is the main function that decides which AI to call.

    Args:
        message: User's message
        model: Selected model ('openai', 'gemini', or 'deepseek')
        chat_history: Previous conversation messages

    Returns:
        AI response as a string
    """
    # Check which model was selected and call appropriate function
    if model == 'openai':
        return call_openai_gpt(message, chat_history)
    elif model == 'gemini':
        return call_gemini(message, chat_history)
    elif model == 'deepseek':
        return call_deepseek(message, chat_history)
    else:
        # Fallback if an unknown model is selected
        return "Error: Unknown model selected"


# =============================================================================
# SECTION 5: HELPER FUNCTIONS
# =============================================================================

def get_user_chat_history(user_id):
    """
    Retrieve all chat messages for a specific user from the database.

    Args:
        user_id: The ID of the logged-in user

    Returns:
        List of message dictionaries
    """
    conn = get_db_connection()

    # Query messages ordered by timestamp (oldest first)
    messages = conn.execute(
        'SELECT * FROM messages WHERE user_id = ? ORDER BY timestamp ASC',
        (user_id,)
    ).fetchall()

    conn.close()

    # Convert Row objects to dictionaries for easier use in templates
    return [dict(msg) for msg in messages]


def save_message(user_id, role, content, model):
    """
    Save a message to the database.

    Args:
        user_id: The ID of the user
        role: 'user' or 'assistant'
        content: The message text
        model: Which AI model was used
    """
    conn = get_db_connection()

    # Insert message into database
    conn.execute(
        'INSERT INTO messages (user_id, role, content, model) VALUES (?, ?, ?, ?)',
        (user_id, role, content, model)
    )

    conn.commit()  # Save changes
    conn.close()


# =============================================================================
# SECTION 6: ROUTES (URL Handlers)
# =============================================================================
# These functions handle different URLs in our application
# Routes respond to HTTP requests and return HTML pages


@app.route('/')
def index():
    """
    Home page - redirects to chat if logged in, otherwise to login page.

    This is what happens when someone visits http://localhost:5000/
    """
    # Check if user is logged in by looking for 'user_id' in session
    if 'user_id' in session:
        # User is logged in, send them to chat
        return redirect(url_for('chat'))
    else:
        # User is not logged in, send them to login page
        return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Login/Register page.

    GET request: Show the login form
    POST request: Process login/registration

    This handles both login AND registration in one function for simplicity.
    """
    # If user is already logged in, redirect to chat
    if 'user_id' in session:
        return redirect(url_for('chat'))

    # Handle form submission (POST request)
    if request.method == 'POST':
        # Get username and password from form
        # request.form is a dictionary containing form data
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        # Validate input
        if not username or not password:
            flash('Please enter both username and password', 'error')
            return redirect(url_for('login'))

        # Connect to database
        conn = get_db_connection()

        # Try to find user in database
        user = conn.execute(
            'SELECT * FROM users WHERE username = ?',
            (username,)
        ).fetchone()

        if user:
            # User exists - check password
            # IMPORTANT: This is PLAIN TEXT comparison
            # In production, you'd use password hashing
            if user['password'] == password:
                # Password correct! Log them in
                session['user_id'] = user['id']
                session['username'] = user['username']
                flash('Logged in successfully!', 'success')
                conn.close()
                return redirect(url_for('chat'))
            else:
                # Wrong password
                flash('Incorrect password', 'error')
                conn.close()
                return redirect(url_for('login'))
        else:
            # User doesn't exist - create new account
            try:
                conn.execute(
                    'INSERT INTO users (username, password) VALUES (?, ?)',
                    (username, password)
                )
                conn.commit()

                # Get the newly created user
                user = conn.execute(
                    'SELECT * FROM users WHERE username = ?',
                    (username,)
                ).fetchone()

                # Log them in automatically
                session['user_id'] = user['id']
                session['username'] = user['username']
                flash('Account created and logged in!', 'success')
                conn.close()
                return redirect(url_for('chat'))

            except sqlite3.IntegrityError:
                # This shouldn't happen, but just in case
                flash('Error creating account', 'error')
                conn.close()
                return redirect(url_for('login'))

    # If GET request, just show the login form
    return render_template('login.html')


@app.route('/logout')
def logout():
    """
    Log out the current user by clearing the session.
    """
    # Clear all session data
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """
    Main chat interface.

    GET request: Display chat history
    POST request: Send message to AI and get response
    """
    # Check if user is logged in
    if 'user_id' not in session:
        # Not logged in, redirect to login page
        return redirect(url_for('login'))

    user_id = session['user_id']

    # Handle message submission (POST request)
    if request.method == 'POST':
        # Get message and selected model from form
        message = request.form.get('message', '').strip()
        model = request.form.get('model', 'openai')

        # Validate that message is not empty
        if not message:
            flash('Please enter a message', 'error')
            return redirect(url_for('chat'))

        # Get chat history for context
        chat_history = get_user_chat_history(user_id)

        # Save user's message to database
        save_message(user_id, 'user', message, model)

        # Get AI response
        ai_response = get_ai_response(message, model, chat_history)

        # Save AI's response to database
        save_message(user_id, 'assistant', ai_response, model)

        # Redirect back to chat page (this causes page reload with new messages)
        # This is the "server-side rendering" approach - no JavaScript needed!
        return redirect(url_for('chat'))

    # If GET request, load and display chat history
    chat_history = get_user_chat_history(user_id)

    # Render the chat template with history
    return render_template('chat.html',
                         chat_history=chat_history,
                         username=session.get('username'))


@app.route('/clear-chat', methods=['POST'])
def clear_chat():
    """
    Clear all chat history for the current user.

    This deletes all messages from the database for this user.
    """
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']

    # Delete all messages for this user
    conn = get_db_connection()
    conn.execute('DELETE FROM messages WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()

    flash('Chat history cleared', 'success')
    return redirect(url_for('chat'))


# =============================================================================
# SECTION 7: RUN THE APPLICATION
# =============================================================================

if __name__ == '__main__':
    """
    Start the Flask development server.

    debug=True enables:
    - Auto-reload when code changes
    - Detailed error messages
    - Interactive debugger

    IMPORTANT: Never use debug=True in production!
    """
    print("\n" + "="*60)
    print("🚀 AI CHAT APP - CLASSROOM DEMO")
    print("="*60)
    print("📝 Make sure you've created a .env file with your API keys!")
    print("📖 Visit: http://localhost:5000")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
