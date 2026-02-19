import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { catchError, forkJoin, interval, of, Subject, takeUntil } from 'rxjs';

import { ForecastChartComponent } from '../components/forecast-chart.component';
import { MetricsChartComponent } from '../components/metrics-chart.component';
import {
  AccuracyMetric,
  ForecastRecord,
  HistoryRecord,
  PipelineSummary,
  SP500Company,
  SystemStatus,
  TaskInfo,
} from '../models/forecast.models';
import { ForecastApiService } from '../services/forecast-api.service';

@Component({
  selector: 'app-dashboard-controller',
  standalone: true,
  imports: [CommonModule, FormsModule, MetricsChartComponent, ForecastChartComponent],
  templateUrl: '../views/dashboard.view.html',
  styleUrls: ['../views/dashboard.view.scss'],
})
export class DashboardControllerComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();

  // System status
  systemStatus: SystemStatus | null = null;
  currentTask: TaskInfo | null = null;

  // Company data
  allCompanies: SP500Company[] = [];
  allSectors: string[] = [];

  // Filters
  searchQuery = '';
  selectedSector = '';

  // Selection
  selectedIds: string[] = [];
  trainingTickers: string[] = []; // Tickers selected for training
  availableModels: string[] = [];
  selectedModel = '';

  // Forecast params
  horizon = 14;
  historyDays = 60;
  levels = '80,95';

  // State
  summary: PipelineSummary | null = null;
  metrics: AccuracyMetric[] = [];
  records: ForecastRecord[] = [];
  historyRecords: HistoryRecord[] = [];
  backtestRecords: ForecastRecord[] = [];
  showBacktest = true;
  backtestMessage = '';
  errorMessage = '';
  successMessage = '';
  isLoading = false;
  isRunningPipeline = false;
  isForecasting = false;
  isLoadingMetrics = false;

  constructor(private readonly api: ForecastApiService) {}

  ngOnInit(): void {
    this.loadSystemStatus();
    this.loadCompanies();
    // Poll for status updates every 2 seconds
    interval(2000)
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => {
        if (this.currentTask && this.currentTask.status === 'running') {
          this.pollTaskStatus();
        }
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  loadSystemStatus(): void {
    this.api.getSystemStatus().subscribe({
      next: (status) => {
        this.systemStatus = status;
        this.currentTask = status.current_task;
        if (status.ready_for_predictions && status.data_stats) {
          // Create a summary from data stats if we have a model ready
          this.summary = {
            rows: status.data_stats.rows,
            unique_series: status.data_stats.companies,
            start: status.data_stats.start_date,
            end: status.data_stats.end_date,
            trained_models: [],
          };
        }
      },
      error: () => {
        // Silent fail - status is optional
      },
    });
  }

  pollTaskStatus(): void {
    if (!this.currentTask) return;

    this.api.getTaskStatus(this.currentTask.task_id).subscribe({
      next: (response) => {
        this.currentTask = response.task;
        if (response.task.status === 'completed') {
          this.successMessage = `Task completed: ${response.task.message}`;
          this.loadCompanies(); // Refresh companies after data update
          this.loadSystemStatus();
          this.loadMetrics(true);
        } else if (response.task.status === 'failed') {
          this.errorMessage = `Task failed: ${response.task.error || response.task.message}`;
        }
      },
      error: () => {
        // Task may have been cleared
        this.currentTask = null;
      },
    });
  }

  loadCompanies(): void {
    this.isLoading = true;
    this.api.getCompanies().subscribe({
      next: (response) => {
        this.allCompanies = response.companies;
        this.allSectors = response.sectors;
        // Default selection: first few available companies
        const availableTickers = response.companies.map(c => c.ticker);
        this.selectedIds = availableTickers.slice(0, Math.min(5, availableTickers.length));
        this.isLoading = false;

        if (response.companies.length === 0) {
          this.errorMessage = 'No companies with data available. Run the pipeline first to download data.';
        }
      },
      error: () => {
        this.errorMessage = 'Failed to load companies. API may be unavailable.';
        this.isLoading = false;
      },
    });
  }

  get filteredCompanies(): SP500Company[] {
    let result = this.allCompanies;

    // Filter by sector
    if (this.selectedSector) {
      result = result.filter(c => c.sector === this.selectedSector);
    }

    // Filter by search query (ticker or name)
    if (this.searchQuery.trim()) {
      const query = this.searchQuery.trim().toLowerCase();
      result = result.filter(c =>
        c.ticker.toLowerCase().includes(query) ||
        c.name.toLowerCase().includes(query) ||
        c.symbol.toLowerCase().includes(query)
      );
    }

    return result;
  }

  isSelected(ticker: string): boolean {
    return this.selectedIds.includes(ticker);
  }

  toggleSelection(ticker: string): void {
    if (this.isSelected(ticker)) {
      this.selectedIds = this.selectedIds.filter(id => id !== ticker);
    } else {
      this.selectedIds = [...this.selectedIds, ticker];
    }
  }

  selectAll(): void {
    const filtered = this.filteredCompanies.map(c => c.ticker);
    // Add all filtered that aren't already selected
    const newIds = new Set([...this.selectedIds, ...filtered]);
    this.selectedIds = Array.from(newIds);
  }

  clearSelection(): void {
    this.selectedIds = [];
  }

  runPipeline(download = true): void {
    if (this.currentTask && this.currentTask.status === 'running') {
      this.errorMessage = 'A task is already running. Please wait for it to complete.';
      return;
    }
    this.isRunningPipeline = true;
    this.errorMessage = '';
    this.successMessage = '';

    // Use background task API for non-blocking operation
    this.api.startFullPipeline(download).subscribe({
      next: (response) => {
        this.currentTask = response.task;
        this.successMessage = response.message;
        this.isRunningPipeline = false;
      },
      error: (err) => {
        this.errorMessage = this.buildError(err);
        this.isRunningPipeline = false;
      },
    });
  }

  startDataUpdate(): void {
    if (this.currentTask && this.currentTask.status === 'running') {
      this.errorMessage = 'A task is already running. Please wait for it to complete.';
      return;
    }
    this.errorMessage = '';
    this.successMessage = '';

    this.api.startDataUpdate().subscribe({
      next: (response) => {
        this.currentTask = response.task;
        this.successMessage = response.message;
      },
      error: (err) => {
        this.errorMessage = this.buildError(err);
      },
    });
  }

  startTraining(useSelectedTickers = false): void {
    if (this.currentTask && this.currentTask.status === 'running') {
      this.errorMessage = 'A task is already running. Please wait for it to complete.';
      return;
    }
    this.errorMessage = '';
    this.successMessage = '';

    const tickers = useSelectedTickers && this.trainingTickers.length > 0
      ? this.trainingTickers
      : undefined;

    this.api.startModelTraining(tickers).subscribe({
      next: (response) => {
        this.currentTask = response.task;
        this.successMessage = response.message;
      },
      error: (err) => {
        this.errorMessage = this.buildError(err);
      },
    });
  }

  setTrainingTickers(): void {
    // Copy currently selected companies to training tickers
    this.trainingTickers = [...this.selectedIds];
    this.successMessage = `Training will use ${this.trainingTickers.length} selected companies.`;
  }

  clearTrainingTickers(): void {
    this.trainingTickers = [];
    this.successMessage = 'Training will use all available companies.';
  }

  loadMetrics(runIfMissing = true): void {
    this.isLoadingMetrics = true;
    this.api.getMetrics(runIfMissing).subscribe({
      next: (response) => {
        this.metrics = response.metrics;
        this.isLoadingMetrics = false;
      },
      error: (err) => {
        this.errorMessage = this.buildError(err);
        this.isLoadingMetrics = false;
      },
    });
  }

  runForecast(): void {
    if (this.selectedIds.length === 0) {
      this.errorMessage = 'Please select at least one company.';
      return;
    }
    this.isForecasting = true;
    this.errorMessage = '';
    this.successMessage = '';
    this.backtestMessage = '';

    forkJoin({
      forecast: this.api.forecast({
        horizon: this.horizon,
        ids: this.selectedIds,
        levels: this.parseLevels(this.levels),
      }),
      history: this.api.getHistory(this.selectedIds, this.historyDays),
      backtest: this.api.getBacktest(this.selectedIds, this.historyDays).pipe(
        catchError((err) => {
          this.backtestMessage = `Historical predictions unavailable: ${this.buildError(err)}`;
          return of({ records: [] as ForecastRecord[], count: 0 });
        }),
      ),
    }).subscribe({
      next: ({ forecast, history, backtest }) => {
        this.records = forecast.records;
        this.historyRecords = history.records;
        this.backtestRecords = backtest.records;
        if (backtest.count === 0 && !this.backtestMessage) {
          this.backtestMessage =
            'Historical predictions are not available yet. Train the model again to generate fitted values.';
        }
        this.availableModels = [...new Set(forecast.records.map(r => r.model_name))];
        if (this.availableModels.length > 0 && !this.selectedModel) {
          this.selectedModel = this.availableModels.includes('ensemble_mean')
            ? 'ensemble_mean'
            : this.availableModels[0];
        }
        this.isForecasting = false;
      },
      error: (err) => {
        this.errorMessage = this.buildError(err);
        this.isForecasting = false;
      },
    });
  }

  private parseLevels(value: string): number[] {
    return value
      .split(',')
      .map((token) => token.trim())
      .filter((token) => token.length > 0)
      .map((token) => Number(token))
      .filter((level) => Number.isFinite(level));
  }

  private buildError(error: unknown): string {
    if (typeof error === 'object' && error) {
      const withMessage = error as { message?: string; error?: { detail?: string } };
      if (withMessage.error?.detail) {
        return String(withMessage.error.detail);
      }
      if (withMessage.message) {
        return String(withMessage.message);
      }
    }
    return 'Request failed. Check backend logs and API availability.';
  }
}
