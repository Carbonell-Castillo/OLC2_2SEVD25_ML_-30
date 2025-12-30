import { Component, OnInit, ChangeDetectorRef, Inject, PLATFORM_ID } from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { NgApexchartsModule } from "ng-apexcharts";
import { DataService, ClusteringResults } from '../../services/data.service';

@Component({
  selector: 'app-interpretacion-segmentos',
  standalone: true,
  imports: [CommonModule, NgApexchartsModule],
  templateUrl: './interpretacion-segmentos.html'
})
export class InterpretacionSegmentosComponent implements OnInit {
  public chartDonut: any;
  public chartLine: any;
  public clusterList: any[] = [];
  public isLoading: boolean = true;

  constructor(
    private dataService: DataService,
    private cdr: ChangeDetectorRef,
    @Inject(PLATFORM_ID) private platformId: Object
  ) {
    this.initEmptyCharts();
  }

  ngOnInit(): void {
    if (isPlatformBrowser(this.platformId)) {
      this.cargarDatosReales();
    }
  }

  private cargarDatosReales(): void {
    const fileId = localStorage.getItem('lastFileId');
    if (!fileId) {
      this.isLoading = false;
      return;
    }

    this.dataService.getResults(fileId).subscribe({
      next: (res: ClusteringResults) => {
        this.mapDataToCharts(res);
        this.isLoading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Error al cargar segmentos:', err);
        this.isLoading = false;
      }
    });
  }

  private mapDataToCharts(res: ClusteringResults): void {
    // 1. Mapear datos para la Tabla
    this.clusterList = res.clusters.map(c => ({
      name: `Cluster ${c.cluster_id}: ${c.main_channel}`,
      val1: c.percentage.toFixed(1),
      val2: c.size,
      status: c.percentage > 25 ? 'up' : 'down', // Lógica de negocio ficticia
      description: c.description
    }));

    // 2. Mapear datos para el Donut
    this.chartDonut = {
      ...this.chartDonut,
      series: res.clusters.map(c => c.size),
      labels: res.clusters.map(c => `Grupo ${c.cluster_id}`)
    };

    // 3. Simular Línea de Evolución (Basada en la calidad del modelo)
    // Nota: Como K-Means es estático, simulamos una tendencia basada en el score
    const baseScore = res.metrics.silhouette_score * 100;
    this.chartLine.series = [
      { 
        name: "Calidad de Segmentación", 
        data: [baseScore - 10, baseScore - 5, baseScore - 2, baseScore, baseScore + 2, baseScore + 1] 
      }
    ];
  }

  private initEmptyCharts() {
    this.chartDonut = {
      series: [],
      chart: { type: "donut", height: 300, foreColor: '#94a3b8' },
      labels: [],
      colors: ["#a855f7", "#06b6d4", "#3b82f6", "#f43f5e", "#eab308"],
      stroke: { show: false },
      dataLabels: { enabled: true },
      legend: { position: 'bottom', labels: { colors: '#94a3b8' } }
    };

    this.chartLine = {
      series: [],
      chart: { type: "area", height: 300, toolbar: { show: false }, foreColor: '#94a3b8' },
      colors: ["#a855f7"],
      dataLabels: { enabled: false },
      stroke: { curve: "smooth", width: 3 },
      xaxis: { categories: ["V1", "V2", "V3", "V4", "V5", "Actual"], labels: { style: { colors: '#94a3b8' } } },
      tooltip: { theme: "dark" },
      grid: { borderColor: "#334155" }
    };
  }
}