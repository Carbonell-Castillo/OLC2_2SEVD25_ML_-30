import { Component, OnInit, ChangeDetectorRef, Inject, PLATFORM_ID } from '@angular/core';
import { isPlatformBrowser, CommonModule } from '@angular/common'; // Importar esto
import { DataService, ClusteringResults } from '../../services/data.service';

@Component({
  selector: 'app-evaluacion-modelos',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './evaluacion-modelos.html',
  styleUrl: './evaluacion-modelos.css',
})
export class EvaluacionModelos implements OnInit {
  resultados: ClusteringResults | null = null;
  pcaImageUrl: string | null = null;
  clusterNames: any = {}; // Para guardar los nombres de los clústers
  isLoading: boolean = true;

  constructor(
    private dataService: DataService,
    private cdr: ChangeDetectorRef,
    @Inject(PLATFORM_ID) private platformId: Object
  ) {}

  ngOnInit(): void {
   // Solo ejecutar si estamos en el navegador
    if (isPlatformBrowser(this.platformId)) {
      this.cargarDatos();
    } else {
      this.isLoading = false; 
    }
  }
cargarDatos(): void {
    if (typeof localStorage === 'undefined') return;
    const fileId = localStorage.getItem('lastFileId');
    if (!fileId) {
      this.isLoading = false;
      return;
    }

    // 1. Obtener métricas
    this.dataService.getResults(fileId).subscribe({
      next: (res) => {
        this.resultados = res;
        console.log('Resultados recibidos:', res);
        this.cdr.detectChanges();
      }
    });

    // 2. Obtener la info de la visualización (JSON)
    this.dataService.getVisualization(fileId).subscribe({
      next: (data: any) => {
        console.log('Datos de visualización recibidos:', data);
        
        // Guardamos los nombres de los clústers que vienen en el JSON
        this.clusterNames = data.cluster_names;
        console.log('Nombres de clústers:', this.clusterNames);
        /**
         * IMPORTANTE:
         * Si tu backend de FastAPI tiene configurado app.mount("/static", ...),
         * deberías construir la URL hacia el servidor, no usar la ruta local D:\.
         */
        const fileName = `${fileId}_pcaGrafica.png`;
        this.pcaImageUrl = `http://localhost:8000/static/${fileName}`;
        
        this.isLoading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Error al cargar visualización:', err);
        this.isLoading = false;
      }
    });
  }
}