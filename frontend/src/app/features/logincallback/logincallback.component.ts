import { Component, OnInit, inject } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService } from '../../core/services/auth.service';

@Component({
  selector: 'app-login-callback',
  template: `<p>Logging in...</p>`
})
export class LoginCallbackComponent implements OnInit {
  private authService = inject(AuthService);
  constructor(private router: Router) { }

  ngOnInit() {
  this.router.navigate(['/chat']);
}

}
