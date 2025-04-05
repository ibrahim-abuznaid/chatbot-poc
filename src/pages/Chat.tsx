import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { ChatInput } from "@/components/ChatInput";
import { ChatMessage } from "@/components/ChatMessage";
import { getCurrentUser, logout } from "@/lib/auth";
import { sendMessage } from "@/lib/api";
import { toast } from "@/components/ui/use-toast";
import QueryTranslationSelector from "@/components/QueryTranslationSelector";
import { Select, MenuItem } from "@mui/material";

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function Chat() {
  const navigate = useNavigate();
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [aiModel, setAiModel] = useState("openai");
  const [qtt, setQtt] = useState("basic");

  const handleSend = async (content: string) => {
    if (!getCurrentUser()) {
      toast({
        title: "Error",
        description: "No user found",
        variant: "destructive",
      });
      return;
    }

    const userMessage: Message = { role: "user", content };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    try {
      console.log('Sending message:', {
        content,
        qtt,
        aiModel,
        messagesCount: messages.length
      });

      const response = await sendMessage(content, qtt, aiModel, messages);
      console.log('Server response:', response);
      
      // Create assistant message from response
      const assistantMessage: Message = {
        role: "assistant",
        content: response.reply
      };

      // Update messages with both the user's message and assistant's response
      setMessages(prev => {
        // If we have conversation history from the server, use it
        if (response.context?.conversation_history?.length > 0) {
          return response.context.conversation_history.map(msg => ({
            role: msg.role as "user" | "assistant",
            content: msg.content
          }));
        }
        // Otherwise, append the new assistant message to the existing messages
        return [...prev, assistantMessage];
      });

    } catch (error) {
      console.error("Message error details:", error);
      // Remove the failed message from the UI
      setMessages((prev) => prev.slice(0, -1));
      
      toast({
        title: "Error",
        description: error instanceof Error 
          ? `Error sending message: ${error.message}`
          : "Failed to send message",
        variant: "destructive",
        duration: 5000
      });
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  // Redirect if not logged in
  if (!getCurrentUser()) {
    navigate("/login");
    return null;
  }

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto p-4">
      {/* Header with logo and controls */}
      <div className="flex flex-col space-y-4 mb-6">
        {/* Logo and title row */}
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <img 
              src="/images/Hilton_HHR_LOGO_Hilton-Black.png" 
              alt="Hilton Logo" 
              className="h-12 mr-4 object-contain"
            />
          </div>
          <Button onClick={handleLogout} variant="outline">
            Logout
          </Button>
        </div>
        
        {/* Controls row */}
        <div className="flex items-center space-x-4 px-2">
          <Select
            value={aiModel}
            onChange={(e) => setAiModel(e.target.value)}
            sx={{ 
              minWidth: 120,
              height: 40,
              backgroundColor: 'white'
            }}
          >
            <MenuItem value="openai">OpenAI</MenuItem>
            <MenuItem value="gemini">Gemini</MenuItem>
          </Select>
          <QueryTranslationSelector qtt={qtt} setQtt={setQtt} />
        </div>
      </div>

      {/* Chat messages area */}
      <div className="flex-1 overflow-y-auto mb-4 space-y-4 border rounded-lg p-4 bg-white shadow-sm">
        {messages.map((message, index) => (
          <ChatMessage key={index} message={message} />
        ))}
      </div>

      {/* Chat input area */}
      <div className="sticky bottom-0 bg-background pt-2">
        <ChatInput onSend={handleSend} disabled={loading} />
      </div>
    </div>
  );
}