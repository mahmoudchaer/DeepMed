from flask import Flask, request, jsonify
import pandas as pd
from typing import Dict, List, Any
import json
import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/medical_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

import openai
import pandas as pd
import json
import os
import sys
import logging
from typing import Dict, List, Any

class MedicalAssistant:
    def __init__(self, api_key: str):
        """Initialize the medical assistant with OpenAI API key."""
        try:
            # Set the API key directly on the openai module
            openai.api_key = api_key
            self.client = openai  # Use the openai module as the client
            logging.info("Medical Assistant: OpenAI client initialized")
            self.conversation_history = {}  # Use dictionary to store conversation history for multiple sessions
            self.system_prompt = (
                "You are an expert medical AI assistant helping healthcare professionals use a machine learning system. "
                "You specialize in understanding medical data and recommending appropriate machine learning models. "
                "Be concise, professional, and focus on providing actionable insights. "
                "When analyzing data, consider medical context and implications."
            )
        except Exception as e:
            logging.error(f"Error initializing OpenAI client for Medical Assistant: {e}")
            raise

    def _get_chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Get completion from OpenAI API."""
        try:
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3  # Lower temperature for more focused responses
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"Error communicating with AI assistant: {str(e)}")
            return f"Error communicating with AI assistant: {str(e)}"

    def analyze_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze uploaded data and provide recommendations."""
        try:
            # Create data summary with more detailed statistics
            data_summary = {
                "columns": list(data.columns),
                "sample": data.head(5).to_dict(),
                "dtypes": data.dtypes.astype(str).to_dict(),
                "missing_values": data.isnull().sum().to_dict(),
                "unique_values": {col: int(data[col].nunique()) for col in data.columns},
                "numeric_stats": data.describe().to_dict() if not data.empty else {}
            }
            
            prompt = f"""Analyze this medical dataset and provide recommendations:
            1. Identify the most likely target column for prediction
            2. Suggest the best type of model to use (classification/regression)
            3. Identify potential data quality issues
            
            Dataset summary: {json.dumps(data_summary)}
            
            Provide your response in JSON format with these keys:
            - target_column: recommended target column name
            - model_type: recommended model type (classification/regression)
            - reasoning: brief explanation of your recommendations
            - data_issues: list of potential data quality concerns
            
            IMPORTANT: Respond with ONLY the JSON object, no additional text or formatting.
            """
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self._get_chat_completion(messages)
            
            # Clean the response - remove any markdown formatting
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Parse and validate the response
            recommendations = json.loads(response)
            
            # Validate required keys
            required_keys = ["target_column", "model_type", "reasoning", "data_issues"]
            if not all(key in recommendations for key in required_keys):
                raise ValueError("Missing required keys in AI response")
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in analyze_data: {str(e)}")
            logger.error(f"Raw response: {response if 'response' in locals() else 'No response'}")
            return {
                "target_column": None,
                "model_type": "classification",
                "reasoning": "Unable to analyze data",
                "data_issues": ["Analysis failed"]
            }

    def get_model_recommendation(self, model_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze model performance and recommend the best model."""
        prompt = f"""Based on these model metrics, recommend the best model for a medical context:
        {json.dumps(model_metrics)}
        
        Consider:
        1. Overall accuracy
        2. False negative rate (critical in medical diagnosis)
        3. Model interpretability
        
        Provide your response in JSON format with:
        - recommended_model: name of recommended model
        - reasoning: explanation of choice
        - considerations: medical implications to consider
        """
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self._get_chat_completion(messages)
        try:
            return json.loads(response)
        except:
            return {
                "recommended_model": None,
                "reasoning": "Unable to parse AI response",
                "considerations": []
            }

    def get_user_guidance(self, user_query: str, context: Dict[str, Any] = None) -> str:
        """Provide contextual guidance to user questions."""
        if context:
            prompt = f"""User question: {user_query}
            Current context: {json.dumps(context)}
            
            Provide a helpful, concise response that:
            1. Directly addresses the user's question
            2. Considers the current context
            3. Gives clear next steps if applicable
            """
        else:
            prompt = user_query
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        return self._get_chat_completion(messages)

    def chat(self, message: str, session_id: str = "default") -> str:
        """Handle ongoing chat conversation with context."""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
            
        self.conversation_history[session_id].append({"role": "user", "content": message})
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history[session_id]
        ]
        
        response = self._get_chat_completion(messages)
        self.conversation_history[session_id].append({"role": "assistant", "content": response})
        
        if len(self.conversation_history[session_id]) > 10:
            self.conversation_history[session_id] = self.conversation_history[session_id][-10:]
        
        return response
        
    def clear_chat_history(self, session_id: str = "default") -> None:
        """Clear the conversation history for a specific session."""
        if session_id in self.conversation_history:
            self.conversation_history[session_id] = []


# Initialize the MedicalAssistant with API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error("OPENAI_API_KEY environment variable not set. Medical Assistant API will not function correctly.")
    medical_assistant = None
else:
    try:
        medical_assistant = MedicalAssistant(api_key)
        logging.info("Medical Assistant initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Medical Assistant: {e}")
        medical_assistant = None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "medical-assistant"}), 200

@app.route('/analyze_data', methods=['POST'])
def analyze_data():
    """
    Analyze data endpoint
    
    Expected JSON input:
    {
        "data": {...}  # Data in JSON format that can be loaded into a pandas DataFrame
    }
    
    Returns:
    {
        "recommendations": {
            "target_column": "...",
            "model_type": "...",
            "reasoning": "...",
            "data_issues": [...]
        },
        "message": "Data analyzed successfully"
    }
    """
    if not medical_assistant:
        return jsonify({"error": "Medical Assistant not initialized. Check API key."}), 503
    
    try:
        # Get request data
        request_data = request.json
        
        if not request_data or 'data' not in request_data:
            return jsonify({"error": "Invalid request. Missing 'data'."}), 400
        
        # Convert JSON to DataFrame
        try:
            data = pd.DataFrame.from_dict(request_data['data'])
        except Exception as e:
            return jsonify({"error": f"Failed to convert JSON to DataFrame: {str(e)}"}), 400
        
        # Analyze data
        recommendations = medical_assistant.analyze_data(data)
        
        return jsonify({
            "recommendations": recommendations,
            "message": "Data analyzed successfully"
        })
    
    except Exception as e:
        logging.error(f"Error in analyze_data endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/recommend_model', methods=['POST'])
def recommend_model():
    """
    Recommend model endpoint
    
    Expected JSON input:
    {
        "model_metrics": [...]  # List of model metrics
    }
    
    Returns:
    {
        "recommendation": {
            "recommended_model": "...",
            "reasoning": "...",
            "considerations": [...]
        },
        "message": "Model recommendation generated successfully"
    }
    """
    if not medical_assistant:
        return jsonify({"error": "Medical Assistant not initialized. Check API key."}), 503
    
    try:
        # Get request data
        request_data = request.json
        
        if not request_data or 'model_metrics' not in request_data:
            return jsonify({"error": "Invalid request. Missing 'model_metrics'."}), 400
        
        # Get recommendation
        recommendation = medical_assistant.get_model_recommendation(request_data['model_metrics'])
        
        return jsonify({
            "recommendation": recommendation,
            "message": "Model recommendation generated successfully"
        })
    
    except Exception as e:
        logging.error(f"Error in recommend_model endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/user_guidance', methods=['POST'])
def user_guidance():
    """
    User guidance endpoint
    
    Expected JSON input:
    {
        "query": "...",  # User query
        "context": {...}  # Optional context
    }
    
    Returns:
    {
        "guidance": "...",  # AI response
        "message": "Guidance generated successfully"
    }
    """
    if not medical_assistant:
        return jsonify({"error": "Medical Assistant not initialized. Check API key."}), 503
    
    try:
        # Get request data
        request_data = request.json
        
        if not request_data or 'query' not in request_data:
            return jsonify({"error": "Invalid request. Missing 'query'."}), 400
        
        # Get context if provided
        context = request_data.get('context')
        
        # Get guidance
        guidance = medical_assistant.get_user_guidance(request_data['query'], context)
        
        return jsonify({
            "guidance": guidance,
            "message": "Guidance generated successfully"
        })
    
    except Exception as e:
        logging.error(f"Error in user_guidance endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint
    
    Expected JSON input:
    {
        "message": "...",  # User message
        "session_id": "..."  # Optional session ID (default: "default")
    }
    
    Returns:
    {
        "response": "...",  # AI response
        "message": "Chat response generated successfully"
    }
    """
    if not medical_assistant:
        return jsonify({"error": "Medical Assistant not initialized. Check API key."}), 503
    
    try:
        # Get request data
        request_data = request.json
        
        if not request_data or 'message' not in request_data:
            return jsonify({"error": "Invalid request. Missing 'message'."}), 400
        
        # Get session ID if provided
        session_id = request_data.get('session_id', 'default')
        
        # Get chat response
        response = medical_assistant.chat(request_data['message'], session_id)
        
        return jsonify({
            "response": response,
            "message": "Chat response generated successfully"
        })
    
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """
    Clear chat history endpoint
    
    Expected JSON input:
    {
        "session_id": "..."  # Optional session ID (default: "default")
    }
    
    Returns:
    {
        "message": "Chat history cleared successfully"
    }
    """
    if not medical_assistant:
        return jsonify({"error": "Medical Assistant not initialized. Check API key."}), 503
    
    try:
        # Get request data
        request_data = request.json or {}
        
        # Get session ID if provided
        session_id = request_data.get('session_id', 'default')
        
        # Clear chat history
        medical_assistant.clear_chat_history(session_id)
        
        return jsonify({
            "message": f"Chat history cleared successfully for session {session_id}"
        })
    
    except Exception as e:
        logging.error(f"Error in clear_chat endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the app on port 5005
    port = int(os.environ.get('PORT', 5005))
    app.run(host='0.0.0.0', port=port, debug=True) 