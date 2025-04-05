import { toast } from "@/components/ui/use-toast";

export interface ChatResponse {
  reply: string;
  conversation_id: string;
  context: {
    original_query: string;
    refined_query: string;
    retrieved_documents: Array<{
      page_content: string;
      metadata: Record<string, any>;
    }>;
    conversation_history: Array<{
      role: string;
      content: string;
    }>;
    llm_model: string;
  };
}

export async function sendMessage(
  message: string, 
  qtt: string = 'basic',
  llm_model: string = 'openai',
  conversation_history: Message[] = []
): Promise<ChatResponse> {
  try {
    console.log('Sending request:', {
      message,
      qtt,
      llm_model,
      conversation_history
    });

    const response = await fetch('/api/send-message', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        message,
        qtt,
        llm_model,
        conversation_history
      }),
      credentials: 'same-origin'
    });

    // First check if response is ok
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Server error response:', errorText);
      try {
        const errorJson = JSON.parse(errorText);
        throw new Error(errorJson.error || 'Server returned an error');
      } catch (e) {
        throw new Error(`Server error: ${errorText}`);
      }
    }

    // Try to get response as text first
    const responseText = await response.text();
    console.log('Raw response:', responseText);

    // Try to parse as JSON
    let data;
    try {
      data = JSON.parse(responseText);
    } catch (e) {
      console.error('Failed to parse JSON response:', responseText);
      throw new Error(`Invalid JSON response from server: ${responseText.substring(0, 100)}...`);
    }

    // Validate response structure
    if (!data) {
      throw new Error('Empty response from server');
    }

    // Validate required fields
    if (!data.reply || !data.context) {
      console.error('Invalid response structure:', data);
      throw new Error('Invalid response format from server: missing required fields');
    }

    // Ensure conversation history is an array
    if (!Array.isArray(data.context.conversation_history)) {
      data.context.conversation_history = [];
    }

    return data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
}

export const getConversationHistory = async (conversation_id: string): Promise<ChatResponse['context']['conversation_history']> => {
  try {
    const response = await fetch(`http://localhost:3001/api/conversation/${conversation_id}`);
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || "Failed to fetch conversation history");
    }

    const data = await response.json();
    return data.history;
  } catch (error) {
    toast({
      title: "Error",
      description: error instanceof Error ? error.message : "Failed to fetch conversation history",
      variant: "destructive",
    });
    throw error;
  }
}; 