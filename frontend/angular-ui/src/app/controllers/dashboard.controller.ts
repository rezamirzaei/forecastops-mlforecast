import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { AccuracyMetric, ForecastRecord, PipelineSummary } from '../models/forecast.models';
import { ForecastApiService } from '../services/forecast-api.service';

@Component({
  selector: 'app-dashboard-controller',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: '../views/dashboard.view.html',
  styleUrl: '../views/dashboard.view.scss',
})
export class DashboardControllerComponent {
  readonly availableIds = ['AAPL.US', 'MSFT.US', 'GOOG.US', 'AMZN.US', 'META.US'];

  horizon = 14;
  levels = '80,95';
  selectedIds = ['AAPL.US', 'MSFT.US', 'GOOG.US'];
  summary: PipelineSummary | null = null;
  metrics: AccuracyMetric[] = [];
  records: ForecastRecord[] = [];
  errorMessage = '';
  isRunningPipeline = false;
  isForecasting = false;
  isLoadingMetrics = false;

  constructor(private readonly api: ForecastApiService) {}

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
    this.isForecasting = true;
    this.errorMessage = '';
    this.api
      .forecast({
        horizon: this.horizon,
        ids: this.selectedIds,
        levels: this.parseLevels(this.levels),
      })
      .subscribe({
        next: (response) => {
          this.records = response.records;
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
