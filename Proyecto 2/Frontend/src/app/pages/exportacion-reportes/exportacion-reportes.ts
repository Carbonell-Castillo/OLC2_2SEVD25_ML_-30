import { Component, Inject, PLATFORM_ID } from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { DataService } from '../../services/data.service';

@Component({
  selector: 'app-exportacion-reportes',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './exportacion-reportes.html',
  styleUrl: './exportacion-reportes.css',
})
export class ExportacionReportes {
  isDownloadingCSV: boolean = false;
  isExportingPDF: boolean = false;

  constructor(
    private dataService: DataService,
    @Inject(PLATFORM_ID) private platformId: Object
  ) {}

  // Llama a tu método downloadCSV
  descargarCSV(): void {
    if (!isPlatformBrowser(this.platformId)) return;

    const fileId = localStorage.getItem('lastFileId');
    if (!fileId) {
      alert('No hay un reporte activo. Por favor, procesa un archivo primero.');
      return;
    }

    this.isDownloadingCSV = true;

    this.dataService.downloadCSV(fileId).subscribe({
      next: (blob: Blob) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `segmentacion_${fileId}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        this.isDownloadingCSV = false;
      },
      error: (err) => {
        console.error('Error al descargar CSV:', err);
        this.isDownloadingCSV = false;
      }
    });
  }

  // Llama a tu método exportPDF
  descargarPDF(): void {
    if (!isPlatformBrowser(this.platformId)) return;

    const fileId = localStorage.getItem('lastFileId');
    if (!fileId) return;

    this.isExportingPDF = true;

    this.dataService.exportPDF(fileId).subscribe({
      next: (res: any) => {

        console.log('Respuesta de exportación PDF:', res);
        const fileName = `${fileId}_reporte.pdf`;
        const pdfUrl = `http://localhost:8000/reports/${fileName}`;
        console.log('URL del PDF generado:', pdfUrl);
        if (res.pdf_path) {
            window.open(pdfUrl, '_blank');
        } else {
            alert('Reporte PDF generado con éxito.');
        }
        this.isExportingPDF = false;
      },
      error: (err) => {
        console.error('Error al exportar PDF:', err);
        this.isExportingPDF = false;
      }
    });
  }
}