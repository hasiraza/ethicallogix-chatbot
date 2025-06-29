class ChatBot {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.userInput = document.getElementById('userInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.clearChatBtn = document.getElementById('clearChat');
        this.retrainBtn = document.getElementById('retrainBtn');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.charCount = document.getElementById('charCount');
        
        this.isWaitingForResponse = false;
        this.messageHistory = [];
        
        this.init();
    }
    
    init() {
        // Set welcome message time
        this.setWelcomeTime();
        
        // Event listeners
        this.userInput.addEventListener('keypress', (e) => this.handleKeyPress(e));
        this.userInput.addEventListener('input', () => this.updateCharCount());
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.clearChatBtn.addEventListener('click', () => this.clearChat());
        this.retrainBtn.addEventListener('click', () => this.retrainModel());
        
        // Focus on input
        this.userInput.focus();
        
        // Initialize character count
        this.updateCharCount();
    }
    
    setWelcomeTime() {
        const welcomeTime = document.getElementById('welcomeTime');
        if (welcomeTime) {
            welcomeTime.textContent = this.getCurrentTime();
        }
    }
    
    getCurrentTime() {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    updateCharCount() {
        const currentLength = this.userInput.value.length;
        const maxLength = this.userInput.maxLength;
        this.charCount.textContent = `${currentLength}/${maxLength}`;
        
        // Change color based on character count
        if (currentLength > maxLength * 0.8) {
            this.charCount.style.color = '#ff6b6b';
        } else if (currentLength > maxLength * 0.6) {
            this.charCount.style.color = '#ffa726';
        } else {
            this.charCount.style.color = '#666';
        }
    }
    
    handleKeyPress(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    }
    
    async sendMessage() {
        const message = this.userInput.value.trim();
        
        if (message === '' || this.isWaitingForResponse) {
            return;
        }
        
        // Display user message
        this.displayMessage(message, 'user');
        this.userInput.value = '';
        this.updateCharCount();
        
        // Disable input and show typing indicator
        this.setInputState(false);
        this.showTypingIndicator();
        
        try {
            // Send message to backend
            const response = await fetch('/get_response', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Simulate some delay for better UX
            setTimeout(() => {
                this.hideTypingIndicator();
                this.displayMessage(data.response, 'bot');
                this.setInputState(true);
            }, 1000);
            
        } catch (error) {
            console.error('Error:', error);
            this.hideTypingIndicator();
            this.displayMessage('Sorry, there was an error processing your request. Please try again.', 'bot', true);
            this.setInputState(true);
        }
    }
    
    displayMessage(message, sender, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        
        if (sender === 'bot') {
            avatarDiv.innerHTML = '<i class="fas fa-robot"></i>';
        } else {
            avatarDiv.innerHTML = '<i class="fas fa-user"></i>';
        }
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        if (isError) {
            textDiv.style.background = '#ffebee';
            textDiv.style.color = '#c62828';
        }
        textDiv.textContent = message;
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = this.getCurrentTime();
        
        contentDiv.appendChild(textDiv);
        contentDiv.appendChild(timeDiv);
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Store message in history
        this.messageHistory.push({
            message: message,
            sender: sender,
            timestamp: new Date()
        });
    }
    
    showTypingIndicator() {
        this.typingIndicator.classList.add('show');
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        this.typingIndicator.classList.remove('show');
    }
    
    setInputState(enabled) {
        this.isWaitingForResponse = !enabled;
        this.userInput.disabled = !enabled;
        this.sendBtn.disabled = !enabled;
        
        if (enabled) {
            this.userInput.focus();
        }
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }
    
    clearChat() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            // Keep only the welcome message
            const welcomeMessage = this.chatMessages.querySelector('.message');
            this.chatMessages.innerHTML = '';
            this.chatMessages.appendChild(welcomeMessage);
            
            // Clear message history
            this.messageHistory = [];
            
            // Show confirmation
            setTimeout(() => {
                this.displayMessage('Chat history has been cleared!', 'bot');
            }, 500);
        }
    }
    
    async retrainModel() {
        if (!confirm('Are you sure you want to retrain the model? This may take a few minutes.')) {
            return;
        }
        
        this.showLoadingOverlay();
        this.setInputState(false);
        
        try {
            const response = await fetch('/retrain', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.displayMessage('Model has been successfully retrained!', 'bot');
            } else {
                throw new Error(data.error || 'Failed to retrain model');
            }
            
        } catch (error) {
            console.error('Error retraining model:', error);
            this.displayMessage(`Error retraining model: ${error.message}`, 'bot', true);
        } finally {
            this.hideLoadingOverlay();
            this.setInputState(true);
        }
    }
    
    showLoadingOverlay() {
        this.loadingOverlay.classList.add('show');
    }
    
    hideLoadingOverlay() {
        this.loadingOverlay.classList.remove('show');
    }
}

// Utility functions
function formatMessage(text) {
    // Simple text formatting (you can expand this)
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        console.log('Text copied to clipboard');
    }).catch(err => {
        console.error('Failed to copy text: ', err);
    });
}

// Initialize the chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const chatbot = new ChatBot();
    
    // Add some keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + K to clear chat
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            chatbot.clearChat();
        }
        
        // Ctrl/Cmd + R to retrain (only if not already in progress)
        if ((e.ctrlKey || e.metaKey) && e.key === 'r' && !chatbot.isWaitingForResponse) {
            e.preventDefault();
            chatbot.retrainModel();
        }
        
        // Escape to focus input
        if (e.key === 'Escape') {
            chatbot.userInput.focus();
        }
    });
    
    // Add visibility change handler to pause/resume when tab is not active
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            // Tab is hidden
            console.log('Chat tab is now hidden');
        } else {
            // Tab is visible
            console.log('Chat tab is now visible');
            chatbot.userInput.focus();
        }
    });
    
    // Service Worker for offline functionality (optional)
    if ('serviceWorker' in navigator) {
        window.addEventListener('load', () => {
            navigator.serviceWorker.register('/static/sw.js')
                .then((registration) => {
                    console.log('SW registered: ', registration);
                })
                .catch((registrationError) => {
                    console.log('SW registration failed: ', registrationError);
                });
        });
    }
});

// Export for potential module use
window.ChatBot = ChatBot;