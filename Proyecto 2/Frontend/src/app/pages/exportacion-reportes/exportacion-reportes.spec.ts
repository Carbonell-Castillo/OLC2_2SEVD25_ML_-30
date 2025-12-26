import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ExportacionReportes } from './exportacion-reportes';

describe('ExportacionReportes', () => {
  let component: ExportacionReportes;
  let fixture: ComponentFixture<ExportacionReportes>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ExportacionReportes]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ExportacionReportes);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
