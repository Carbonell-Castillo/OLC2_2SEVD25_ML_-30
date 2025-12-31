import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router'; // Para <router-outlet>
// Ajusta las rutas según donde estén definidos tus componentes Header y Sidebar
import { Header } from '../components/header/header'; // Para <app-header>
import { Sidebar } from '../components/sidebar/sidebar'; // Para <app-sidebar>

@Component({
  selector: 'app-dashboard-layout',
  standalone: true, // Si es un componente independiente
  templateUrl: './dashboard-layout.html',
  styleUrls: ['./dashboard-layout.css'],
  imports: [
    RouterOutlet, // Añadir aquí
    Sidebar, // Añadir aquí
    Header, // Añadir aquí
  ],
  
})
export class DashboardLayout {
  // ...
}