import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { DashboardControllerComponent } from './dashboard.controller';
import { ForecastApiService } from '../services/forecast-api.service';

class MockApiService {
  getAvailableSeries() {
    return of({
      series: ['AAPL.US', 'MSFT.US', 'GOOG.US'],
      count: 3,
    });
  }

  getCompanies() {
    return of({
      companies: [
        { ticker: 'AAPL.US', symbol: 'AAPL', name: 'Apple Inc.', sector: 'Technology' },
        { ticker: 'MSFT.US', symbol: 'MSFT', name: 'Microsoft Corporation', sector: 'Technology' },
        { ticker: 'GOOG.US', symbol: 'GOOG', name: 'Alphabet Inc.', sector: 'Communication Services' },
      ],
      sectors: ['Technology', 'Communication Services'],
      count: 3,
    });
  }

  runPipeline() {
    return of({
      rows: 100,
      unique_series: 2,
      start: '2024-01-01',
      end: '2024-01-10',
      trained_models: ['lin_reg'],
    });
  }

  getMetrics() {
    return of({
      metrics: [{ model: 'lin_reg', smape: 10.2, wape: 9.1 }],
      best_model: 'lin_reg',
      count: 1,
    });
  }

  forecast() {
    return of({
      records: [
        { unique_id: 'AAPL.US', ds: '2024-01-11', model_name: 'lin_reg', value: 123.4 },
      ],
      count: 1,
    });
  }

  getHistory() {
    return of({
      records: [
        { unique_id: 'AAPL.US', ds: '2024-01-10', value: 120.0 },
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
    fixture.detectChanges(); // triggers ngOnInit
  });

  it('loads companies on init', () => {
    expect(component.allCompanies.length).toBe(3);
    expect(component.allSectors.length).toBe(2);
  });

  it('filters companies by search query', () => {
    component.searchQuery = 'apple';
    expect(component.filteredCompanies.length).toBe(1);
    expect(component.filteredCompanies[0].symbol).toBe('AAPL');
  });

  it('filters companies by sector', () => {
    component.selectedSector = 'Technology';
    expect(component.filteredCompanies.length).toBe(2);
  });

  it('toggles company selection', () => {
    component.selectedIds = ['AAPL.US'];
    component.toggleSelection('AAPL.US');
    expect(component.selectedIds).not.toContain('AAPL.US');

    component.toggleSelection('MSFT.US');
    expect(component.selectedIds).toContain('MSFT.US');
  });

  it('runs pipeline and stores summary', () => {
    component.runPipeline();
    expect(component.summary?.rows).toBe(100);
    expect(component.metrics.length).toBe(1);
  });

  it('runs forecast and stores records', () => {
    component.summary = {
      rows: 100,
      unique_series: 2,
      start: '2024-01-01',
      end: '2024-01-10',
      trained_models: ['lin_reg'],
    };
    component.selectedIds = ['AAPL.US'];
    component.runForecast();
    expect(component.records.length).toBe(1);
    expect(component.historyRecords.length).toBe(1);
  });
});
