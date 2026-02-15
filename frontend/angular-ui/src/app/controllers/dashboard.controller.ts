import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { forkJoin } from 'rxjs';

import { ForecastChartComponent } from '../components/forecast-chart.component';
import { MetricsChartComponent } from '../components/metrics-chart.component';
import {
  AccuracyMetric,
  ForecastRecord,
  HistoryRecord,
  PipelineSummary,
  SP500Company,
} from '../models/forecast.models';
import { ForecastApiService } from '../services/forecast-api.service';

@Component({
  selector: 'app-dashboard-controller',
  standalone: true,
  imports: [CommonModule, FormsModule, MetricsChartComponent, ForecastChartComponent],
  templateUrl: '../views/dashboard.view.html',
  styleUrls: ['../views/dashboard.view.scss'],
})
export class DashboardControllerComponent implements OnInit {
  // Company data
  allCompanies: SP500Company[] = [];
  allSectors: string[] = [];

  // Filters
  searchQuery = '';
  selectedSector = '';

  // Selection
  selectedIds: string[] = [];
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
  errorMessage = '';
  isLoading = false;
  isRunningPipeline = false;
  isForecasting = false;
  isLoadingMetrics = false;

  constructor(private readonly api: ForecastApiService) {}

  ngOnInit(): void {
    this.loadCompanies();
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

  runPipeline(): void {
    this.isRunningPipeline = true;
    this.errorMessage = '';
    this.api.runPipeline(false).subscribe({
      next: (summary) => {
        this.summary = summary;
        this.loadMetrics(false);
        this.isRunningPipeline = false;
      },
      error: (err) => {
        this.errorMessage = this.buildError(err);
        this.isRunningPipeline = false;
      },
    });
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
    if (!this.summary) {
      this.errorMessage = 'Run the pipeline first to train and persist models.';
      return;
    }
    if (this.selectedIds.length === 0) {
      this.errorMessage = 'Please select at least one company.';
      return;
    }
    this.isForecasting = true;
    this.errorMessage = '';

    forkJoin({
      forecast: this.api.forecast({
        horizon: this.horizon,
        ids: this.selectedIds,
        levels: this.parseLevels(this.levels),
      }),
      history: this.api.getHistory(this.selectedIds, this.historyDays),
    }).subscribe({
      next: ({ forecast, history }) => {
        this.records = forecast.records;
        this.historyRecords = history.records;
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
