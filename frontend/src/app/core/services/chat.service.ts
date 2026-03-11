import { Injectable } from "@angular/core";
import { HttpClient } from "@angular/common/http";
import { Observable } from "rxjs";
import { AuthService } from "./auth.service";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

@Injectable({
  providedIn: "root",
})
export class ChatService {
  private apiUrl = "http://localhost:8000/api/v1/agent";

  constructor(private http: HttpClient, private authService: AuthService) {}

  async *streamChat(message: string, threadId: string): AsyncGenerator<string> {
    const response = await fetch(`${this.apiUrl}/execute`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        workflow_id: threadId,
        input_data: { message, thread_id: threadId }
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    
    // The backend returns { result: { ... }, execution_log: { ... } }
    // Extract the result content
    if (data.result) {
      const resultContent = data.result;
      
      // Handle different result formats
      if (typeof resultContent === 'string') {
        yield resultContent;
      } else if (resultContent.content) {
        yield resultContent.content;
      } else if (resultContent.message) {
        yield resultContent.message;
      } else if (resultContent.response) {
        yield resultContent.response;
      } else if (resultContent.output) {
        // Handle output field from backend
        const output = resultContent.output;
        if (output.error) {
          yield output.error;
        } else if (output.message) {
          yield output.message;
        } else if (output.content) {
          yield output.content;
        } else {
          // If output is a string, use it directly
          yield typeof output === 'string' ? output : JSON.stringify(output, null, 2);
        }
      } else {
        // If result is an object, stringify it
        yield JSON.stringify(resultContent, null, 2);
      }
    } else if (data.error) {
      throw new Error(data.error);
    } else {
      // Fallback: yield the entire response as string
      yield JSON.stringify(data, null, 2);
    }
  }

  uploadFile(
    file: File,
    threadId: string,
    description: string = ""
  ): Observable<any> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("thread_id", threadId);
    formData.append("description", description);

    return this.http.post(`${this.apiUrl}/upload`, formData, {
      withCredentials: true,
    });
  }
}