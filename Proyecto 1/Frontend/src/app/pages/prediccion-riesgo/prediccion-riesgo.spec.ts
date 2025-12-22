import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PrediccionRiesgo } from './prediccion-riesgo';

describe('PrediccionRiesgo', () => {
  let component: PrediccionRiesgo;
  let fixture: ComponentFixture<PrediccionRiesgo>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PrediccionRiesgo]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PrediccionRiesgo);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
