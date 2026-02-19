import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { DashboardControllerComponent } from './dashboard.controller';
import { ForecastApiService } from '../services/forecast-api.service';

class MockApiService {
  getAvailableSeries() {
    return of({ series: ['AAPL.US', 'MSFT.US', 'GOOG.US'], count: 3 });
  }

  getCompanies() {
    return of({
      companies: [
        { ticker: 'AAPL.US', symbol: 'AAPL', name: 'Apple Inc.', sector: 'Technology', has_data: true },
        { ticker: 'MSFT.US', symbol: 'MSFT', name: 'Microsoft', sector: 'Technology', has_data: true },
        { ticker: 'GOOG.US', symbol: 'GOOG', name: 'Alphabet', sector: 'Communication Services', has_data: true },
      ],
      sectors: ['Technology', 'Communication Services'],
      count: 3,
    });
  }

  runPipeline() {
    return of({ rows: 100, unique_series: 2, start: '2024-01-01', end: '2024-01-10', trained_models: ['lin_reg'] });
  }

  getMetrics() {
    return of({ metrics: [{ model: 'lin_reg', smape: 10.2, wape: 9.1 }], best_model: 'lin_reg', count: 1 });
  }

  forecast() {
    return of({ records: [{ unique_id: 'AAPL.US', ds: '2024-01-11', model_name: 'lin_reg', value: 123.4 }], count: 1 });
  }

  getHistory() {
    return of({ records: [{ unique_id: 'AAPL.US', ds: '2024-01-10', value: 120.0 }], count: 1 });
  }

  getBacktest() {
    return of({
      records: [{ unique_id: 'AAPL.US', ds: '2024-01-10', model_name: 'lin_reg', value: 119.7 }],
      count: 1,
    });
  }

  getSystemStatus() {
    return of({
      has_data: true,
      has_model: true,
      is_busy: false,
      current_task: null,
      data_stats: { rows: 50000, companies: 100, start_date: '2019-01-01', end_date: '2024-01-10' },
      ready_for_predictions: true,
    });
  }

  startDataUpdate() {
    return of({
      task: {
        task_id: 'abc123',
        task_type: 'data_update',
        status: 'running',
        created_at: '2024-01-10T10:00:00',
        started_at: '2024-01-10T10:00:01',
        completed_at: null,
        progress: 10,
        message: 'Downloading...',
        result: {},
        error: null,
        tickers_requested: [],
      },
      message: 'Data update started in background',
    });
  }

  startModelTraining() {
    return of({
      task: {
        task_id: 'abc124',
        task_type: 'model_training',
        status: 'running',
        created_at: '2024-01-10T10:00:00',
        started_at: '2024-01-10T10:00:01',
        completed_at: null,
        progress: 10,
        message: 'Training...',
        result: {},
        error: null,
        tickers_requested: [],
      },
      message: 'Model training started in background',
    });
  }

  startFullPipeline() {
    return of({
      task: {
        task_id: 'abc125',
        task_type: 'full_pipeline',
        status: 'running',
        created_at: '2024-01-10T10:00:00',
        started_at: '2024-01-10T10:00:01',
        completed_at: null,
        progress: 5,
        message: 'Starting...',
        result: {},
        error: null,
        tickers_requested: [],
      },
      message: 'Full pipeline started in background',
    });
  }

  getTaskStatus() {
    return of({
      task: {
        task_id: 'abc123',
        task_type: 'data_update',
        status: 'completed',
        created_at: '2024-01-10T10:00:00',
        started_at: '2024-01-10T10:00:01',
        completed_at: '2024-01-10T10:05:00',
        progress: 100,
        message: 'Done',
        result: { rows: 50000, companies: 100 },
        error: null,
        tickers_requested: [],
      },
    });
  }

  getAllTasks() {
    return of({ tasks: [], count: 0 });
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
    fixture.detectChanges();
  });

  it('loads companies and system status on init', () => {
    expect(component.allCompanies.length).toBe(3);
    expect(component.allSectors.length).toBe(2);
    expect(component.systemStatus).toBeTruthy();
    expect(component.systemStatus?.ready_for_predictions).toBe(true);
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

  it('starts full pipeline as background task', () => {
    component.runPipeline();
    expect(component.currentTask).toBeTruthy();
    expect(component.currentTask?.task_type).toBe('full_pipeline');
    expect(component.successMessage).toContain('background');
  });

  it('runs forecast and stores records', () => {
    component.selectedIds = ['AAPL.US'];
    component.runForecast();
    expect(component.records.length).toBe(1);
    expect(component.historyRecords.length).toBe(1);
    expect(component.backtestRecords.length).toBe(1);
    expect(component.backtestMessage).toBe('');
  });

  it('starts data update as background task', () => {
    component.startDataUpdate();
    expect(component.currentTask).toBeTruthy();
    expect(component.currentTask?.task_type).toBe('data_update');
  });

  it('sets training tickers from selection', () => {
    component.selectedIds = ['AAPL.US', 'MSFT.US'];
    component.setTrainingTickers();
    expect(component.trainingTickers).toEqual(['AAPL.US', 'MSFT.US']);
  });
});
