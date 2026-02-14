import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';

import { ForecastApiService } from './forecast-api.service';

describe('ForecastApiService', () => {
  let service: ForecastApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [ForecastApiService, provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(ForecastApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('calls pipeline endpoint', () => {
    service.runPipeline(true).subscribe((summary) => {
      expect(summary.rows).toBe(10);
    });

    const req = httpMock.expectOne('/api/pipeline/run?download=true');
    expect(req.request.method).toBe('POST');
    req.flush({ rows: 10, unique_series: 1, start: '', end: '', trained_models: [] });
  });

  it('calls metrics endpoint', () => {
    service.getMetrics(true).subscribe((resp) => {
      expect(resp.count).toBe(1);
    });

    const req = httpMock.expectOne('/api/pipeline/metrics?run_if_missing=true');
    expect(req.request.method).toBe('GET');
    req.flush({ metrics: [{ model: 'lin_reg', smape: 1.2, wape: 1.1 }], best_model: 'lin_reg', count: 1 });
  });
});
