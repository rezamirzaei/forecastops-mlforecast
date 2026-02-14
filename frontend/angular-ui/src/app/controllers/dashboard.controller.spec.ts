import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { DashboardControllerComponent } from './dashboard.controller';
import { ForecastApiService } from '../services/forecast-api.service';

class MockApiService {
  runPipeline() {
    return of({ rows: 100, unique_series: 2, start: '2024-01-01', end: '2024-01-10', trained_models: ['lin_reg'] });
  }

  forecast() {
    return of({
      records: [
        { unique_id: 'AAPL.US', ds: '2024-01-11', model_name: 'lin_reg', value: 123.4 },
      ],
      count: 1,
    });
  }
}

describe('DashboardControllerComponent', () => {
  let component: DashboardControllerComponent;
  let fixture: ComponentFixture<DashboardControllerComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DashboardControllerComponent],
      providers: [{ provide: ForecastApiService, useClass: MockApiService }],
    }).compileComponents();

    fixture = TestBed.createComponent(DashboardControllerComponent);
    component = fixture.componentInstance;
  });

  it('runs pipeline and stores summary', () => {
    component.runPipeline();
    expect(component.summary?.rows).toBe(100);
  });

  it('runs forecast and stores records', () => {
    component.runForecast();
    expect(component.records.length).toBe(1);
  });
});
