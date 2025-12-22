import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EvaluacionModelos } from './evaluacion-modelos';

describe('EvaluacionModelos', () => {
  let component: EvaluacionModelos;
  let fixture: ComponentFixture<EvaluacionModelos>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EvaluacionModelos]
    })
    .compileComponents();

    fixture = TestBed.createComponent(EvaluacionModelos);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
