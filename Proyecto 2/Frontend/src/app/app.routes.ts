import { Routes } from '@angular/router';
import { DashboardLayout } from './components/dashboard-layout';
import { CargaMasiva } from './pages/carga-masiva/carga-masiva';
import { EvaluacionModelos } from './pages/evaluacion-modelos/evaluacion-modelos';
import { ExportacionReportes } from './pages/exportacion-reportes/exportacion-reportes';
import { InterpretacionSegmentosComponent } from './pages/interpretacion-segmentos/interpretacion-segmentos';
import { ConfiguracionModeloComponent } from './pages/configuracion-modelo/configuracion-modelo';

export const routes: Routes = [
    {
        path: '',
        component: DashboardLayout,
        children: [
            // Ruta por defecto
            { path: '', redirectTo: 'carga-masiva', pathMatch: 'full' },
            
            // Rutas del menú
            { path: 'carga-masiva', component: CargaMasiva, title: 'Carga Masiva' },
            { path: 'evaluacion-modelos', component: EvaluacionModelos, title: 'Evaluación de Modelos' },
            { path: 'exportacion-reportes', component: ExportacionReportes, title: 'Exportación de Reportes' },
            { path: 'interpretacion-segmentos', component: InterpretacionSegmentosComponent, title: 'Interpretación de Segmentos'},
            { path: 'configuracion-modelo', component: ConfiguracionModeloComponent, title: 'Configuración del Modelo' }

        ]
    },
    { path: '**', redirectTo: '' }
];