import { Component, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  ChartComponent,
  ApexAxisChartSeries,
  ApexChart,
  ApexXAxis,
  ApexDataLabels,
  ApexTooltip,
  ApexStroke,
  ApexNonAxisChartSeries,
  ApexTheme,
  NgApexchartsModule
} from "ng-apexcharts";

export type ChartOptions = {
  series: ApexAxisChartSeries | ApexNonAxisChartSeries;
  chart: ApexChart;
  xaxis: ApexXAxis;
  stroke: ApexStroke;
  tooltip: ApexTooltip;
  dataLabels: ApexDataLabels;
  labels: string[];
  theme: ApexTheme;
  colors: string[];
};

@Component({
  selector: 'app-interpretacion-segmentos',
  standalone: true,
  imports: [CommonModule, NgApexchartsModule],
  templateUrl: './interpretacion-segmentos.html'
})
export class InterpretacionSegmentosComponent {
  public chartDonut: Partial<ChartOptions> | any;
  public chartLine: Partial<ChartOptions> | any;

  mockData = [
    { name: 'Segmento A - Premium', val1: 85, val2: 92, status: 'up' },
    { name: 'Segmento B - Ocasional', val1: 42, val2: 58, status: 'down' },
    { name: 'Segmento C - Leales', val1: 76, val2: 88, status: 'up' },
    { name: 'Segmento D - Riesgo', val1: 15, val2: 30, status: 'down' }
  ];

  constructor() {
    this.initCharts();
  }

  initCharts() {
    // Gráfica de Donut (Distribución)
    this.chartDonut = {
      series: [44, 55, 13, 33],
      chart: { type: "donut", height: 300, foreColor: '#94a3b8' },
      labels: ["Premium", "Ocasional", "Leales", "Riesgo"],
      colors: ["#a855f7", "#06b6d4", "#3b82f6", "#f43f5e"],
      stroke: { show: false },
      dataLabels: { enabled: false },
      legend: { position: 'bottom' }
    };

    // Gráfica de Líneas (Tendencias)
    this.chartLine = {
      series: [
        { name: "Satisfacción", data: [31, 40, 28, 51, 42, 109, 100] },
        { name: "Retención", data: [11, 32, 45, 32, 34, 52, 41] }
      ],
      chart: { type: "area", height: 300, toolbar: { show: false }, foreColor: '#94a3b8' },
      colors: ["#a855f7", "#06b6d4"],
      dataLabels: { enabled: false },
      stroke: { curve: "smooth", width: 3 },
      xaxis: {
        categories: ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
      },
      tooltip: { theme: "dark" },
      grid: { borderColor: "#334155" }
    };
  }
}