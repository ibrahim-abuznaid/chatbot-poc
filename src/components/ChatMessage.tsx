import { Message } from "@/lib/openai";
import { cn } from "@/lib/utils";
import ReactMarkdown from 'react-markdown';

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage = ({ message }: ChatMessageProps) => {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex w-full mb-4 animate-fade-in",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div
        className={cn(
          "max-w-[80%] rounded-lg px-4 py-2 shadow-sm",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-secondary text-secondary-foreground"
        )}
      >
        <ReactMarkdown>{message.content}</ReactMarkdown>
      </div>
    </div>
  );
};