import { Component, Input, OnChanges, SimpleChanges, ViewChild, ElementRef, AfterViewInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Chart, registerables } from 'chart.js';

import { ForecastRecord, HistoryRecord } from '../models/forecast.models';

Chart.register(...registerables);

@Component({
  selector: 'app-forecast-chart',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="chart-container" *ngIf="records.length > 0 || historyRecords.length > 0">
      <canvas #chartCanvas></canvas>
    </div>
    <p class="no-data" *ngIf="records.length === 0 && historyRecords.length === 0">
      No data to display. Run a forecast first.
    </p>
  `,
  styles: [`
    .chart-container {
      position: relative;
      height: 400px;
      width: 100%;
      margin: 1rem 0;
    }
    .no-data {
      color: #666;
      text-align: center;
      padding: 2rem;
    }
  `]
})
export class ForecastChartComponent implements AfterViewInit, OnChanges, OnDestroy {
  @Input() records: ForecastRecord[] = [];
  @Input() historyRecords: HistoryRecord[] = [];
  @Input() selectedModel: string = '';
  @ViewChild('chartCanvas') chartCanvas!: ElementRef<HTMLCanvasElement>;

  private chart: Chart | null = null;

  private readonly seriesColors: Record<string, { solid: string; dashed: string }> = {
    'AAPL.US': { solid: 'rgba(54, 162, 235, 1)', dashed: 'rgba(54, 162, 235, 0.7)' },
    'MSFT.US': { solid: 'rgba(255, 99, 132, 1)', dashed: 'rgba(255, 99, 132, 0.7)' },
    'GOOG.US': { solid: 'rgba(75, 192, 192, 1)', dashed: 'rgba(75, 192, 192, 0.7)' },
    'AMZN.US': { solid: 'rgba(255, 206, 86, 1)', dashed: 'rgba(255, 206, 86, 0.7)' },
    'META.US': { solid: 'rgba(153, 102, 255, 1)', dashed: 'rgba(153, 102, 255, 0.7)' },
  };

  private readonly defaultColor = { solid: 'rgba(100, 100, 100, 1)', dashed: 'rgba(100, 100, 100, 0.7)' };

  ngAfterViewInit(): void {
    if (this.records.length > 0 || this.historyRecords.length > 0) {
      this.createChart();
    }
  }

  ngOnChanges(changes: SimpleChanges): void {
    if ((changes['records'] || changes['historyRecords'] || changes['selectedModel']) && this.chartCanvas) {
      this.updateChart();
    }
  }

  ngOnDestroy(): void {
    this.chart?.destroy();
  }

  private getColor(seriesId: string) {
    return this.seriesColors[seriesId] || this.defaultColor;
  }

  private createChart(): void {
    if (!this.chartCanvas?.nativeElement) return;
    if (this.records.length === 0 && this.historyRecords.length === 0) return;

    // Get all unique series from both history and forecast
    const allSeriesIds = [...new Set([
      ...this.historyRecords.map(r => r.unique_id),
      ...this.records.map(r => r.unique_id),
    ])];

    // Filter forecast by selected model
    const models = [...new Set(this.records.map(r => r.model_name))];
    const targetModel = this.selectedModel || models[0] || 'ensemble_mean';
    const filteredForecast = this.records.filter(r => r.model_name === targetModel);

    // Combine all dates
    const historyDates = this.historyRecords.map(r => r.ds);
    const forecastDates = filteredForecast.map(r => r.ds);
    const allDates = [...new Set([...historyDates, ...forecastDates])].sort();

    // Find last history date for connection point
    const lastHistoryDate = historyDates.length > 0 ? historyDates.sort().slice(-1)[0] : null;

    // Build datasets
    const datasets: any[] = [];

    for (const seriesId of allSeriesIds) {
      const color = this.getColor(seriesId);

      // History data map
      const seriesHistory = this.historyRecords.filter(r => r.unique_id === seriesId);
      const historyMap = new Map(seriesHistory.map(r => [r.ds, r.value]));

      // Forecast data map
      const seriesForecast = filteredForecast.filter(r => r.unique_id === seriesId);
      const forecastMap = new Map(seriesForecast.map(r => [r.ds, r.value]));

      // Historical line (solid)
      if (seriesHistory.length > 0) {
        datasets.push({
          label: seriesId + ' (History)',
          data: allDates.map(date => historyMap.get(date) ?? null),
          borderColor: color.solid,
          backgroundColor: 'transparent',
          borderWidth: 2,
          fill: false,
          tension: 0.1,
          pointRadius: 0,
          pointHoverRadius: 4,
        });
      }

      // Forecast line (dashed) - connect from last history point
      if (seriesForecast.length > 0) {
        const forecastData = allDates.map(date => {
          // Include last history point for seamless connection
          if (date === lastHistoryDate && historyMap.has(date)) {
            return historyMap.get(date);
          }
          return forecastMap.get(date) ?? null;
        });

        datasets.push({
          label: seriesId + ' (Forecast)',
          data: forecastData,
          borderColor: color.dashed,
          backgroundColor: 'transparent',
          borderWidth: 3,
          borderDash: [6, 4],
          fill: false,
          tension: 0.1,
          pointRadius: 3,
          pointHoverRadius: 6,
        });
      }
    }

    this.chart = new Chart(this.chartCanvas.nativeElement, {
      type: 'line',
      data: {
        labels: allDates.map(d => new Date(d).toLocaleDateString()),
        datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: true, position: 'top' },
          title: {
            display: true,
            text: 'Price History & Forecast (' + targetModel + ')',
            font: { size: 14 }
          },
          tooltip: {
            callbacks: {
              label: (ctx) => ctx.parsed.y !== null ? ctx.dataset.label + ': $' + ctx.parsed.y.toFixed(2) : ''
            }
          }
        },
        scales: {
          x: { title: { display: true, text: 'Date' }, grid: { display: false } },
          y: { title: { display: true, text: 'Price ($)' }, grid: { color: 'rgba(0,0,0,0.05)' } }
        }
      }
    });
  }

  private updateChart(): void {
    this.chart?.destroy();
    this.createChart();
  }
}
