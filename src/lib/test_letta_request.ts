import { toast } from "@/components/ui/use-toast";

const LETTA_API_BASE = "http://localhost:8283/v1"; // Update with your Letta server URL

interface LettaAgentConfig {
  name: string;
  user_id: string;
  llm_config: {
    model: string;
    model_endpoint_type: string;
    model_endpoint: string;
  };
}

interface LettaResponse {
  id: string;
  messages: Array<{
    role: string;
    text: string;
  }>;
  usage: {
    total_tokens: number;
  };
}

// 1. Create an agent when user signs up
export const createLettaAgent = async (userId: string): Promise<string> => {
  try {
    const response = await fetch(`${LETTA_API_BASE}/agents`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        name: `user_${userId}_agent`,
        user_id: userId,
        llm_config: {
          model: "gpt-4o",
          model_endpoint_type: "openai",
          model_endpoint: "https://api.openai.com/v1"
        }
      } as LettaAgentConfig),
    });

    if (!response.ok) {
      throw new Error("Failed to create Letta agent");
    }

    const data = await response.json();
    return data.id; // Returns the agent ID
  } catch (error) {
    toast({
      title: "Error",
      description: error instanceof Error ? error.message : "Failed to create agent",
      variant: "destructive",
    });
    throw error;
  }
};

// 2. Send message to the agent
export const sendLettaMessage = async (agentId: string, message: string): Promise<LettaResponse> => {
  try {
    const response = await fetch(`${LETTA_API_BASE}/agents/${agentId}/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messages: [{
          role: "user",
          text: message
        }],
        stream_steps: false,
        stream_tokens: false
      }),
    });

    if (!response.ok) {
      throw new Error("Failed to send message to Letta agent");
    }

    return await response.json();
  } catch (error) {
    toast({
      title: "Error",
      description: error instanceof Error ? error.message : "Failed to send message",
      variant: "destructive",
    });
    throw error;
  }
};

// 3. Upload file to knowledge base
export const uploadToKnowledgeBase = async (agentId: string, file: File): Promise<void> => {
  try {
    // First create a source
    const sourceResponse = await fetch(`${LETTA_API_BASE}/sources`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        name: `source_${file.name}`,
        description: `Uploaded file: ${file.name}`
      }),
    });

    if (!sourceResponse.ok) {
      throw new Error("Failed to create source");
    }

    const sourceData = await sourceResponse.json();
    const sourceId = sourceData.id;

    // Upload file to source
    const formData = new FormData();
    formData.append("file", file);

    const uploadResponse = await fetch(`${LETTA_API_BASE}/sources/${sourceId}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!uploadResponse.ok) {
      throw new Error("Failed to upload file");
    }

    // Attach source to agent
    const attachResponse = await fetch(`${LETTA_API_BASE}/sources/${sourceId}/attach/${agentId}`, {
      method: "POST",
    });

    if (!attachResponse.ok) {
      throw new Error("Failed to attach source to agent");
    }
  } catch (error) {
    toast({
      title: "Error",
      description: error instanceof Error ? error.message : "Failed to upload file",
      variant: "destructive",
    });
    throw error;
  }
};

// 4. Get agent context
export const getAgentContext = async (agentId: string) => {
  try {
    const response = await fetch(`${LETTA_API_BASE}/agents/${agentId}/state`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error("Failed to get agent context");
    }

    return await response.json();
  } catch (error) {
    toast({
      title: "Error",
      description: error instanceof Error ? error.message : "Failed to get agent context",
      variant: "destructive",
    });
    throw error;
  }
}; 