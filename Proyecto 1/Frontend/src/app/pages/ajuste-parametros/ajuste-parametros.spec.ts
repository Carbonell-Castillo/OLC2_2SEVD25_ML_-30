import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AjusteParametros } from './ajuste-parametros';

describe('AjusteParametros', () => {
  let component: AjusteParametros;
  let fixture: ComponentFixture<AjusteParametros>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AjusteParametros]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AjusteParametros);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
