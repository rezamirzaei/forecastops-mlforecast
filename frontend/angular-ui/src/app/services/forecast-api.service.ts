import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import {
  ForecastRequest,
  ForecastResponse,
  PipelineSummary,
} from '../models/forecast.models';

@Injectable({ providedIn: 'root' })
export class ForecastApiService {
  private readonly baseUrl = '/api';

  constructor(private readonly http: HttpClient) {}

  runPipeline(download = true): Observable<PipelineSummary> {
    const params = new HttpParams().set('download', String(download));
    return this.http.post<PipelineSummary>(`${this.baseUrl}/pipeline/run`, null, { params });
  }

  forecast(request: ForecastRequest): Observable<ForecastResponse> {
    return this.http.post<ForecastResponse>(`${this.baseUrl}/forecast`, request);
  }

  health(): Observable<{ status: string }> {
    return this.http.get<{ status: string }>(`${this.baseUrl}/health`);
  }
}
