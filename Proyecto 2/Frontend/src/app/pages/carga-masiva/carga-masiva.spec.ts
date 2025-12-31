import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CargaMasiva } from './carga-masiva';

describe('CargaMasiva', () => {
  let component: CargaMasiva;
  let fixture: ComponentFixture<CargaMasiva>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CargaMasiva]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CargaMasiva);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
