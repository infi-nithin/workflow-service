import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-dashboard',
  template: `
    <div class="dashboard-container">
      <div class="hero-card">
        <div class="icon"></div>
        <h1>DFTP-MCP</h1>
        <button (click)="login()" class="login-btn">
          Login to Dashboard
        </button>
      </div>
    </div>
  `,
  styles: [`
    .dashboard-container {
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      background-color: var(--bg-color); /* Light Background */
      font-family: var(--font-family, sans-serif);
      color: var(--text-color);
    }

    .hero-card {
      background: var(--input-bg); /* White/Light card */
      backdrop-filter: blur(10px);
      border: 1px solid var(--border-color);
      border-radius: 24px;
      padding: 3rem;
      text-align: center;
      max-width: 500px;
      width: 90%;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.05); /* Lighter shadow */
    }

    .icon {
      font-size: 4rem;
      margin-bottom: 1.5rem;
      color: var(--primary-color); /* Orange Icon */
    }

    h1 {
      font-size: 2rem;
      margin: 0 0 1rem 0;
      color: var(--title-color); /* Green Title */
      font-family: var(--font-heading); /* Canela */
      font-weight: 700;
      letter-spacing: -0.5px;
    }

    .subtitle {
      color: var(--text-color);
      font-size: 1.1rem;
      margin-bottom: 2rem;
      line-height: 1.5;
    }

    .login-btn {
      padding: 1rem 3rem;
      background: var(--primary-color); /* Orange */
      color: #FFFFFF;
      border: 1px solid var(--primary-color);
      border-radius: 50px; /* Pill shape */
      font-size: 1.1rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.3s ease;
      box-shadow: 0 4px 6px rgba(255, 117, 64, 0.3);
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    .login-btn:hover {
      background: var(--title-color); /* Green Hover */
      border-color: var(--title-color);
      box-shadow: 0 4px 6px rgba(14, 84, 71, 0.4);
      transform: translateY(-2px);
    }

    .login-btn:active {
      transform: translateY(0);
    }
  `]
})
export class DashboardComponent {
  constructor(private router: Router) { }
  login() {
    window.location.href = 'http://localhost:8081/api/auth/login?redirect_uri=http://localhost:4200/login-callback';
  }

}
