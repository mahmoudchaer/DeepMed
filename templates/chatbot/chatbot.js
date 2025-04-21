// DeepMed Chatbot Widget
document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const chatbotWidget = document.getElementById('deepmed-chatbot-widget');
    const toggleBtn = document.getElementById('chatbot-toggle-btn');
    const closeBtn = document.getElementById('chatbot-close-btn');
    const chatWindow = document.getElementById('chatbot-window');
    const messagesContainer = document.getElementById('chatbot-messages');
    const inputField = document.getElementById('chatbot-input');
    const sendBtn = document.getElementById('chatbot-send-btn');
    
    // ChatBot state
    const chatbotState = {
        isOpen: false,
        isLoading: false,
        storageKey: 'deepmed_chatbot_history'
    };
    
    // Open/close chat window
    function toggleChatWindow() {
        chatbotState.isOpen = !chatbotState.isOpen;
        
        if (chatbotState.isOpen) {
            chatWindow.style.display = 'flex';
            setTimeout(() => {
                inputField.focus();
            }, 300);
        } else {
            chatWindow.style.display = 'none';
        }
    }
    
    // Event listeners
    toggleBtn.addEventListener('click', toggleChatWindow);
    closeBtn.addEventListener('click', toggleChatWindow);
    
    // Send message on button click
    sendBtn.addEventListener('click', sendMessage);
    
    // Send message on Enter key (but allow Shift+Enter for newlines)
    inputField.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Auto-resize textarea as user types
    inputField.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.scrollHeight > 100) {
            this.style.overflowY = 'auto';
        } else {
            this.style.overflowY = 'hidden';
        }
    });
    
    // Load chat history from sessionStorage
    function loadChatHistory() {
        const history = sessionStorage.getItem(chatbotState.storageKey);
        if (history) {
            const messages = JSON.parse(history);
            messages.forEach(message => {
                appendMessage(message.role, message.content);
            });
            scrollToBottom();
        } else {
            // Add a welcome message if no history exists
            const welcomeMessage = "Hello! I'm your DeepMed assistant. How can I help you with the platform today?";
            appendMessage('assistant', welcomeMessage);
            saveMessageToHistory('assistant', welcomeMessage);
            scrollToBottom();
        }
    }
    
    // Save message to history in sessionStorage
    function saveMessageToHistory(role, content) {
        let history = sessionStorage.getItem(chatbotState.storageKey);
        let messages = history ? JSON.parse(history) : [];
        messages.push({ role, content });
        sessionStorage.setItem(chatbotState.storageKey, JSON.stringify(messages));
    }
    
    // Get chat history for API request
    function getChatHistory() {
        const history = sessionStorage.getItem(chatbotState.storageKey);
        return history ? JSON.parse(history) : [];
    }
    
    // Append message to chat window
    function appendMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        
        if (role === 'user') {
            messageDiv.classList.add('user-message');
        } else {
            messageDiv.classList.add('assistant-message');
        }
        
        messageDiv.textContent = content;
        messagesContainer.appendChild(messageDiv);
        scrollToBottom();
    }
    
    // Show loading indicator
    function showLoadingIndicator() {
        chatbotState.isLoading = true;
        const loadingDiv = document.createElement('div');
        loadingDiv.classList.add('typing-indicator');
        loadingDiv.id = 'typing-indicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            loadingDiv.appendChild(dot);
        }
        
        messagesContainer.appendChild(loadingDiv);
        scrollToBottom();
    }
    
    // Remove loading indicator
    function removeLoadingIndicator() {
        chatbotState.isLoading = false;
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    // Scroll messages container to bottom
    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    // Send message to API and handle response
    function sendMessage() {
        const message = inputField.value.trim();
        
        if (message === '' || chatbotState.isLoading) {
            return;
        }
        
        // Add user message to UI
        appendMessage('user', message);
        
        // Save user message to history
        saveMessageToHistory('user', message);
        
        // Clear input field
        inputField.value = '';
        inputField.style.height = 'auto';
        
        // Show loading indicator
        showLoadingIndicator();
        
        // Prepare data for API request
        const requestData = {
            user_id: 'demo_user',
            message: message,
            history: getChatHistory()
        };
        
        // Always use the real endpoint
        const endpoint = '/chatbot/query';
        
        // Make API request
        fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Remove loading indicator
            removeLoadingIndicator();
            
            if (data && (data.response || data.reply)) {
                // Extract reply from either response or reply property
                const reply = data.reply || data.response;
                
                // Add assistant message to UI
                appendMessage('assistant', reply);
                
                // Save assistant message to history
                saveMessageToHistory('assistant', reply);
            } else {
                throw new Error('Invalid response from server');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            removeLoadingIndicator();
            appendMessage('assistant', 'Sorry, I encountered an error. Please try again later.');
            saveMessageToHistory('assistant', 'Sorry, I encountered an error. Please try again later.');
        });
    }
    
    // Load chat history when the component is initialized
    loadChatHistory();
}); 