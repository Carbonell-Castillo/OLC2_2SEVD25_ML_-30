import { Routes } from '@angular/router';
import { DashboardLayout } from './components/dashboard-layout';
import { CargaMasiva } from './pages/carga-masiva/carga-masiva';
import { AjusteParametros } from './pages/ajuste-parametros/ajuste-parametros';
import { EvaluacionModelos } from './pages/evaluacion-modelos/evaluacion-modelos';
import { PrediccionRiesgo } from './pages/prediccion-riesgo/prediccion-riesgo';

export const routes: Routes = [
    {
        path: '',
        component: DashboardLayout,
        children: [
            // Ruta por defecto
            { path: '', redirectTo: 'carga-masiva', pathMatch: 'full' },
            
            // Rutas del menú
            { path: 'carga-masiva', component: CargaMasiva, title: 'Carga Masiva' },
            { path: 'ajuste-parametros', component: AjusteParametros, title: 'Ajuste de Parámetros' },
            { path: 'evaluacion-modelos', component: EvaluacionModelos, title: 'Evaluación de Modelos' },
            { path: 'prediccion-riesgo', component: PrediccionRiesgo, title: 'Predicción' }
        ]
    },
    // Opcional: Ruta para manejar 404
    { path: '**', redirectTo: '' }
];