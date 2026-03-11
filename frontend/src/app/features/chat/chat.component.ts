import { Component, ElementRef, ViewChild, inject, signal, effect } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService, ChatMessage } from '../../core/services/chat.service';
import { AuthService } from '../../core/services/auth.service';
import { marked } from 'marked';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="chat-layout">
      <!-- Mobile Backdrop -->
      <div *ngIf="isSidebarOpen()" class="mobile-backdrop" (click)="closeSidebar()"></div>

      <aside class="sidebar" [class.open]="isSidebarOpen()">
        <button class="close-sidebar-btn" (click)="closeSidebar()">✕</button>
        <div class="sidebar-header">
        </div>
        
        <div class="sidebar-info">
          <div class="info-item">
            <span class="label">Scope</span>
            <span class="value">{{ scope() }}</span>
          </div>
          <div class="info-item">
            <span class="label">Role</span>
            <span class="value">{{ roles().join(', ') || 'N/A' }}</span>
          </div>
          <!-- <div class="info-item">
            <span class="label">Thread ID</span>
            <span class="value code">{{ threadId().slice(0, 8) }}...</span>
          </div> -->
        </div>

        <div class="sidebar-actions">
           <button class="btn-outline" (click)="clearHistory()">
             Clear History
           </button>
           <button class="btn-outline" (click)="logout()">
             Logout
           </button>
        </div>
      </aside>

      <main class="chat-main">
        <!-- Mobile Header -->
        <div class="mobile-header">
          <button class="menu-btn" (click)="toggleSidebar()">
            <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24" fill="currentColor">
              <path d="M120-240v-80h720v80H120Zm0-200v-80h720v80H120Zm0-200v-80h720v80H120Z"/>
            </svg>
          </button>
          <span>Chat</span>
        </div>
        <div class="messages-container" #scrollContainer>
          <div *ngIf="messages().length === 0" class="empty-state">
            <!-- Professional Empty State -->
            <h1>How can I help you today?</h1>
            <p>You can ask about orders, positions, or upload files.</p>
          </div>

          <div *ngFor="let msg of messages()" 
               class="message-wrapper" 
               [class.user]="msg.role === 'user'">
            <div class="message-bubble">
              <div class="avatar">
                <!-- User Icon -->
                <svg *ngIf="msg.role === 'user'" xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24" fill="currentColor">
                  <path d="M480-480q-66 0-113-47t-47-113q0-66 47-113t113-47q66 0 113 47t47 113q0 66-47 113t-113 47ZM160-160v-32q0-34 17.5-62.5T224-306q54-27 109-41.5T480-362q57 0 112 14.5t109 41.5q32 29 49.5 57.5T768-192v32H160Z"/>
                </svg>
                <!-- AI Icon (Sparkle/Robot) -->
                <svg *ngIf="msg.role !== 'user'" xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24" fill="currentColor">
                   <path d="M480-120 200-272v-240L40-600l440-240 440 240-160 88v240L480-120Zm0-176 195-106v-153l-55-30-140 78-140-78-55 30v153l195 106Z"/>
                </svg>
              </div>
              <div class="content" [innerHTML]="renderMarkdown(msg.content)"></div>
            </div>
          </div>

          <div *ngIf="isTyping()" class="typing-indicator">
            <span>Agent is thinking...</span>
            <div class="dots"><span>.</span><span>.</span><span>.</span></div>
          </div>
        </div>

        <div class="input-area">
          <!-- File Staging Area -->
          <div *ngIf="selectedFile()" class="file-staging">
             <div class="file-info">
               <span class="label">FILE:</span>
               <span class="name">{{ selectedFile()?.name }}</span>
               <button class="close-btn" (click)="cancelUpload()">✕</button>
             </div>
             <input 
               type="text" 
               [(ngModel)]="fileDescription" 
               placeholder="Add a description for this file..."
               class="description-input"
               (keydown.enter)="confirmUpload()"
             />
             <div class="staging-actions">
               <button class="btn-primary" (click)="confirmUpload()" [disabled]="uploadingFile() !== null">
                 {{ uploadingFile() ? 'Uploading...' : 'Upload & Send' }}
               </button>
             </div>
          </div>

          <!-- Standard Input Area (hidden if staging file?) No, let's keep it visible or hide it. 
               Hiding it avoids confusion. -->
          <div class="input-container" *ngIf="!selectedFile()">
            <button class="attach-btn" (click)="fileInput.click()" title="Upload File">
              <svg xmlns="http://www.w3.org/2000/svg" height="20" viewBox="0 -960 960 960" width="20" fill="currentColor">
                <path d="M720-330q0 104-73 177T470-80q-104 0-177-73t-73-177v-370q0-75 52.5-127.5T400-880q75 0 127.5 52.5T580-700v350q0 46-32 78t-78 32q-46 0-78-32t-32-78v-370h80v370q0 13 8.5 21.5T470-320q13 0 21.5-8.5T500-350v-350q0-42-29-71t-71-29q-42 0-71 29t-29 71v370q0 71 49.5 120.5T470-160q71 0 120.5-49.5T640-330v-390h80v390Z"/>
              </svg>
            </button>
            <input 
              #fileInput
              type="file" 
              hidden 
              (change)="onFileSelected($event)" 
            />
            
            <textarea 
              [(ngModel)]="newMessage" 
              (keydown.enter)="onEnter($event)"
              placeholder="Type your message here..."
              rows="1"
            ></textarea>
            
            <button class="send-btn" (click)="sendMessage()" [disabled]="!newMessage.trim() || isTyping()">
              Send
            </button>
          </div>
          <div *ngIf="uploadingFile()" class="upload-status">
            Uploading {{ uploadingFile() }}...
          </div>
        </div>
      </main>
    </div>
  `,
  styleUrls: ['./chat.component.scss']
})
export class ChatComponent {
  private chatService = inject(ChatService);
  private authService = inject(AuthService);

  @ViewChild('scrollContainer') private scrollContainer!: ElementRef;

  messages = signal<ChatMessage[]>([]);
  newMessage = '';
  isTyping = signal(false);
  uploadingFile = signal<string | null>(null);
  selectedFile = signal<File | null>(null);
  fileDescription = '';

  isSidebarOpen = signal(false);

  threadId = signal<string>(crypto.randomUUID());
  scope = this.authService.currentUserScope;
  roles = this.authService.currentUserRoles;

  toggleSidebar() {
    this.isSidebarOpen.update(v => !v);
  }

  closeSidebar() {
    this.isSidebarOpen.set(false);
  }

  ngOnInit() {
    this.authService.fetchUser();
  }


  constructor() {
    effect(() => {
      this.messages();
      setTimeout(() => this.scrollToBottom(), 100);
    });
  }

  renderMarkdown(content: string): string {
    return marked.parse(content) as string;
  }

  async sendMessage() {
    if (!this.newMessage.trim() || this.isTyping()) return;

    const userMsg = this.newMessage;
    this.newMessage = '';

    this.messages.update(msgs => [...msgs, { role: 'user', content: userMsg }]);
    this.isTyping.set(true);

    try {
      let assistantMsg = '';
      this.messages.update(msgs => [...msgs, { role: 'assistant', content: '' }]);

      for await (const chunk of this.chatService.streamChat(userMsg, this.threadId())) {
        assistantMsg += chunk;
        this.messages.update(msgs => {
          const newMsgs = [...msgs];
          newMsgs[newMsgs.length - 1] = { role: 'assistant', content: assistantMsg };
          return newMsgs;
        });
      }
    } catch (err) {
      console.error(err);
      this.messages.update(msgs => [...msgs, { role: 'assistant', content: 'Error: Failed to get response.' }]);
    } finally {
      this.isTyping.set(false);
    }
  }

  onEnter(event: Event) {
    if ((event as KeyboardEvent).shiftKey) return;
    event.preventDefault();
    this.sendMessage();
  }

  onFileSelected(event: any) {
    const file = event.target.files[0];
    if (!file) return;
    this.selectedFile.set(file);
    this.fileDescription = ''; 
    event.target.value = '';
  }

  cancelUpload() {
    this.selectedFile.set(null);
    this.fileDescription = '';
  }

  confirmUpload() {
    const file = this.selectedFile();
    if (!file) return;

    this.uploadingFile.set(file.name);
    const displayMsg = `📤 Uploaded file: **${file.name}**\n\n> ${this.fileDescription}`;

    this.chatService.uploadFile(file, this.threadId(), this.fileDescription).subscribe({
      next: (res) => {
        this.messages.update(msgs => [...msgs, {
          role: 'user',
          content: displayMsg
        }]);

        if (res.agent_response) {
          this.messages.update(msgs => [...msgs, {
            role: 'assistant',
            content: res.agent_response
          }]);
        }
        this.uploadingFile.set(null);
        this.selectedFile.set(null);
        this.fileDescription = '';
      },
      error: (err) => {
        alert('Upload failed: ' + err.message);
        this.uploadingFile.set(null);
      }
    });
  }

  clearHistory() {
    this.messages.set([]);
    this.threadId.set(crypto.randomUUID());
  }

  logout() {
    this.authService.logout();
  }

  private scrollToBottom() {
    if (this.scrollContainer) {
      this.scrollContainer.nativeElement.scrollTop = this.scrollContainer.nativeElement.scrollHeight;
    }
  }
}
