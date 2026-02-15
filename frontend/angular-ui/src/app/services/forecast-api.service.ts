import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import {
  ForecastRequest,
  ForecastResponse,
  MetricsResponse,
  PipelineSummary,
  SeriesResponse,
} from '../models/forecast.models';

@Injectable({ providedIn: 'root' })
export class ForecastApiService {
  private readonly baseUrl = '/api';

  constructor(private readonly http: HttpClient) {}

  getAvailableSeries(): Observable<SeriesResponse> {
    return this.http.get<SeriesResponse>(`${this.baseUrl}/series`);
  }

  runPipeline(download = false): Observable<PipelineSummary> {
    const params = new HttpParams().set('download', String(download));
    return this.http.post<PipelineSummary>(`${this.baseUrl}/pipeline/run`, null, { params });
  }

  getMetrics(runIfMissing = true): Observable<MetricsResponse> {
    const params = new HttpParams().set('run_if_missing', String(runIfMissing));
    return this.http.get<MetricsResponse>(`${this.baseUrl}/pipeline/metrics`, { params });
  }

  forecast(request: ForecastRequest): Observable<ForecastResponse> {
    return this.http.post<ForecastResponse>(`${this.baseUrl}/forecast`, request);
  }

  health(): Observable<{ status: string }> {
    return this.http.get<{ status: string }>(`${this.baseUrl}/health`);
  }
}
