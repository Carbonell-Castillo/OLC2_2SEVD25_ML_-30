import { ComponentFixture, TestBed } from '@angular/core/testing';

import { InterpretacionSegmentos } from './interpretacion-segmentos';

describe('InterpretacionSegmentos', () => {
  let component: InterpretacionSegmentos;
  let fixture: ComponentFixture<InterpretacionSegmentos>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [InterpretacionSegmentos]
    })
    .compileComponents();

    fixture = TestBed.createComponent(InterpretacionSegmentos);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
