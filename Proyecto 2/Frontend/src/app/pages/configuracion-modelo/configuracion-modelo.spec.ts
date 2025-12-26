import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ConfiguracionModelo } from './configuracion-modelo';

describe('ConfiguracionModelo', () => {
  let component: ConfiguracionModelo;
  let fixture: ComponentFixture<ConfiguracionModelo>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ConfiguracionModelo]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ConfiguracionModelo);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
