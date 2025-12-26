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
    { name: 'Evaluación de Modelos', icon: 'fas fa-flask', route: '/evaluacion-modelos', colorClass: 'group-hover:text-accent-cyan' },
    { name: 'Configuración del Modelo', icon: 'fas fa-cog', route: '/configuracion-modelo', colorClass: 'group-hover:text-accent-purple' },
    { name: 'Interpretación de Segmentos', icon: 'fas fa-pie-chart', route: '/interpretacion-segmentos', colorClass: 'group-hover:text-accent-yellow' },
    { name: 'Exportación de Reportes', icon: 'fas fa-file-export', route: '/exportacion-reportes', colorClass: 'group-hover:text-accent-green' }
  ];

  // La clase 'active-link-style' se logra con RouterLinkActive y ngClass
  activeClasses = 'border-l-4 border-accent-purple bg-card-dark text-white';
  inactiveClasses = 'text-gray-400';
}