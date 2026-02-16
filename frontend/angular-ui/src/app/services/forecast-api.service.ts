import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import {
  CompaniesResponse,
  ForecastRequest,
  ForecastResponse,
  HistoryResponse,
  MetricsResponse,
  PipelineSummary,
  SeriesResponse,
  SystemStatus,
  TaskInfo,
  TaskResponse,
  TasksListResponse,
  TaskStartRequest,
  TrainingRequest,
} from '../models/forecast.models';

@Injectable({ providedIn: 'root' })
export class ForecastApiService {
  private readonly baseUrl = '/api';

  constructor(private readonly http: HttpClient) {}

  getAvailableSeries(): Observable<SeriesResponse> {
    return this.http.get<SeriesResponse>(`${this.baseUrl}/series`);
  }

  getCompanies(): Observable<CompaniesResponse> {
    return this.http.get<CompaniesResponse>(`${this.baseUrl}/companies`);
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

  getHistory(ids?: string[], lastN = 60): Observable<HistoryResponse> {
    let params = new HttpParams().set('last_n', String(lastN));
    if (ids && ids.length > 0) {
      params = params.set('ids', ids.join(','));
    }
    return this.http.get<HistoryResponse>(`${this.baseUrl}/history`, { params });
  }

  health(): Observable<{ status: string }> {
    return this.http.get<{ status: string }>(`${this.baseUrl}/health`);
  }

  // =============== Background Task API ===============

  getSystemStatus(): Observable<SystemStatus> {
    return this.http.get<SystemStatus>(`${this.baseUrl}/status`);
  }

  startDataUpdate(tickers?: string[]): Observable<TaskResponse> {
    const body: TaskStartRequest = { download: true, tickers: tickers || null };
    return this.http.post<TaskResponse>(`${this.baseUrl}/tasks/data-update`, body);
  }

  startModelTraining(tickers?: string[], download = false): Observable<TaskResponse> {
    const body: TrainingRequest = { tickers: tickers || null, download };
    return this.http.post<TaskResponse>(`${this.baseUrl}/tasks/train`, body);
  }

  startFullPipeline(download = true, tickers?: string[]): Observable<TaskResponse> {
    const body: TaskStartRequest = { download, tickers: tickers || null };
    return this.http.post<TaskResponse>(`${this.baseUrl}/tasks/full-pipeline`, body);
  }

  getTaskStatus(taskId: string): Observable<{ task: TaskInfo }> {
    return this.http.get<{ task: TaskInfo }>(`${this.baseUrl}/tasks/${taskId}`);
  }

  getAllTasks(): Observable<TasksListResponse> {
    return this.http.get<TasksListResponse>(`${this.baseUrl}/tasks`);
  }
}
