import openai
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Any
import json
from openai import OpenAI
import logging

class MedicalAssistant:
    def __init__(self, api_key: str):
        """Initialize the medical assistant with OpenAI API key."""
        try:
            # Initialize with minimal configuration to avoid proxy issues
            self.client = OpenAI(api_key=api_key)
            logging.info("Medical Assistant: OpenAI client initialized")
            self.conversation_history = []
            self.system_prompt = """You are an expert medical AI assistant helping healthcare professionals use a machine learning system.
            You specialize in understanding medical data and recommending appropriate machine learning models.
            Be concise, professional, and focus on providing actionable insights.
            When analyzing data, consider medical context and implications."""
        except Exception as e:
            logging.error(f"Error initializing OpenAI client for Medical Assistant: {e}")
            raise
    
    def _get_chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Get completion from OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3  # Lower temperature for more focused responses
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error communicating with AI assistant: {str(e)}")
            return f"Error communicating with AI assistant: {str(e)}"

    def analyze_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze uploaded data and provide recommendations."""
        # Create data summary
        data_summary = {
            "columns": list(data.columns),
            "sample": data.head(5).to_dict(),
            "dtypes": data.dtypes.astype(str).to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "unique_values": {col: data[col].nunique() for col in data.columns}
        }
        
        prompt = f"""Analyze this medical dataset and provide recommendations:
        1. Identify the most likely target column for prediction
        2. Suggest the best type of model to use
        3. Identify potential data quality issues
        
        Dataset summary: {json.dumps(data_summary)}
        
        Provide your response in JSON format with these keys:
        - target_column: recommended target column name
        - model_type: recommended model type
        - reasoning: brief explanation of your recommendations
        - data_issues: list of potential data quality concerns
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
                "target_column": None,
                "model_type": "logistic_regression",
                "reasoning": "Unable to parse AI response",
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

    def chat(self, message: str) -> str:
        """Handle ongoing chat conversation with context."""
        self.conversation_history.append({"role": "user", "content": message})
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history
        ]
        
        response = self._get_chat_completion(messages)
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Keep conversation history manageable
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return response

def create_chat_interface():
    """Create a Streamlit chat interface for the medical assistant."""
    st.markdown("""
        <div style='background-color: #f0f9ff; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: #0077b6; margin-bottom: 10px;'>Medical AI Assistant</h3>
            <p>I can help you with:</p>
            <ul>
                <li>Analyzing your medical data</li>
                <li>Recommending appropriate models</li>
                <li>Explaining results and implications</li>
                <li>Guiding you through the process</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input using chat_input instead of text_input
    if prompt := st.chat_input("Ask me anything about your medical data analysis..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        if "medical_assistant" in st.session_state:
            with st.spinner("Thinking..."):
                response = st.session_state.medical_assistant.chat(prompt)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display assistant response immediately
                with st.chat_message("assistant"):
                    st.markdown(response)
        else:
            st.error("Medical Assistant not initialized. Please check your OpenAI API key.")

    # Add a clear chat button
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        st.experimental_rerun() 