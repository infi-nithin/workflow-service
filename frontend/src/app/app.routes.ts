import { Routes } from '@angular/router';
import { authGuard } from './core/auth.guard';
import { DashboardComponent } from './features/dashboard/dashboard.component';
import { LoginCallbackComponent } from './features/logincallback/logincallback.component';
import { ChatComponent } from './features/chat/chat.component';

export const routes = [
  { path: '', component: DashboardComponent },
  { path: 'login-callback', component: LoginCallbackComponent },
  { path: 'chat', component: ChatComponent ,
    canActivate: [authGuard]
  }
];
