import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { CommonModule } from '@angular/common'; // Necesario para ngClass

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [RouterLink, RouterLinkActive, CommonModule],
  templateUrl: './sidebar.html',
  styleUrl: './sidebar.css'
})
export class Sidebar {
  // Lista de items de navegación para renderizado
  navItems = [
    { name: 'Carga Masiva', icon: 'fas fa-cloud-upload-alt', route: '/carga-masiva', colorClass: 'group-hover:text-accent-purple' },
    { name: 'Ajuste de Parámetros', icon: 'fas fa-sliders-h', route: '/ajuste-parametros', colorClass: 'group-hover:text-accent-cyan' },
    { name: 'Evaluación de Modelos', icon: 'fas fa-flask', route: '/evaluacion-modelos', colorClass: 'group-hover:text-accent-cyan' },
    { name: 'Predicción', icon: 'fas fa-chart-line', route: '/prediccion-riesgo', colorClass: 'group-hover:text-accent-purple' }
  ];

  // La clase 'active-link-style' se logra con RouterLinkActive y ngClass
  activeClasses = 'border-l-4 border-accent-purple bg-card-dark text-white';
  inactiveClasses = 'text-gray-400';
}