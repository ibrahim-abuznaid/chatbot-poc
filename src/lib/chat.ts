// Create a new file to handle chat-specific functionality
import { ChatResponse, sendMessage, getConversationHistory } from './api';
import { getCurrentUser, updateUserConversation } from './auth';

interface ChatState {
  messages: Array<{
    role: string;
    content: string;
  }>;
  conversationId?: string;
}

class ChatManager {
  private state: ChatState = {
    messages: [],
    conversationId: undefined,
  };

  constructor() {
    // Initialize with any existing conversation
    const user = getCurrentUser();
    if (user?.currentConversationId) {
      this.loadConversation(user.currentConversationId);
    }
  }

  async loadConversation(conversationId: string) {
    try {
      const history = await getConversationHistory(conversationId);
      this.state = {
        messages: history,
        conversationId,
      };
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  }

  async sendMessage(
    message: string,
    qtt: string = 'basic',
    llm_model: string = 'openai'
  ): Promise<ChatResponse> {
    try {
      // Add message to local state immediately for UI feedback
      this.state.messages.push({
        role: 'user',
        content: message
      });

      const response = await sendMessage(
        message,
        qtt,
        llm_model,
        this.state.conversationId
      );

      // Update state with server response
      this.state.conversationId = response.conversation_id;
      this.state.messages = response.context.conversation_history;

      // Update user's current conversation
      const user = getCurrentUser();
      if (user) {
        updateUserConversation(user.id, response.conversation_id);
      }

      return response;
    } catch (error) {
      // Remove the message if the request failed
      this.state.messages.pop();
      console.error('Failed to send message:', error);
      throw error;
    }
  }

  getMessages() {
    return this.state.messages;
  }

  getCurrentConversationId() {
    return this.state.conversationId;
  }

  clearConversation() {
    this.state = {
      messages: [],
      conversationId: undefined,
    };
    const user = getCurrentUser();
    if (user) {
      updateUserConversation(user.id, '');
    }
  }
}

// Export a singleton instance
export const chatManager = new ChatManager(); 