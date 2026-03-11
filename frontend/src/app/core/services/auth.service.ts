import { Injectable, computed, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { Observable, tap } from 'rxjs';
import { jwtDecode } from 'jwt-decode';

interface JwtPayload {
  scope?: string;
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private apiUrl = 'http://localhost:8081/api';
  isAuthenticated = signal(false);
  currentUserRoles = signal<string[]>([]);
  currentUserScope = signal<string>('General');

  constructor(private http: HttpClient, private router: Router) { }

  login() {
    window.location.href = `${this.apiUrl}/auth/login`;
  }

  logout() {
    this.isAuthenticated.set(false);
    window.location.href = `${this.apiUrl}/auth/logout`;
  }

  me() {
    return this.http.get<any>(`${this.apiUrl}/auth/me`, {
      withCredentials: true
    });
  }

  fetchUser() {
    this.http.get<any>(`${this.apiUrl}/auth/me`, { withCredentials: true })
      .subscribe({
        next: (res) => {
          if (res.authenticated) {
            this.isAuthenticated.set(true);
            this.currentUserRoles.set(res.roles);
            this.currentUserScope.set(res.scope);
          }
        },
        error: () => this.isAuthenticated.set(false)
      });
  }
}
