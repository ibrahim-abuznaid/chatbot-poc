// import { toast } from "@/components/ui/use-toast";

// export const sendMessage = async (message: string): Promise<string> => {
//   try {
//     const response = await fetch("http://localhost:3001/api/send-message", {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({ message }),
//     });
//     if (!response.ok) {
//       const errorData = await response.json();
//       throw new Error(errorData.error || "Failed to send message");
//     }
//     const data = await response.json();
//     return data.reply;
//   } catch (error) {
//     toast({
//       title: "Error",
//       description:
//         error instanceof Error ? error.message : "Failed to send message",
//       variant: "destructive",
//     });
//     throw error;
//   }
// };
import { toast } from "@/components/ui/use-toast";

export interface Message {
  role: string;
  content: string;
}

export const sendMessage = async (message: string): Promise<string> => {
  try {
    const response = await fetch("http://localhost:3001/api/send-message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || "Failed to send message");
    }
    const data = await response.json();
    return data.reply;
  } catch (error) {
    toast({
      title: "Error",
      description:
        error instanceof Error ? error.message : "Failed to send message",
      variant: "destructive",
    });
    throw error;
  }
};