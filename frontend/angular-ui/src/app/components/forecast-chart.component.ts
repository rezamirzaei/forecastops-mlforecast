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
  private colorCache: Map<string, { solid: string; dashed: string }> = new Map();

  // Base colors that will be cycled through for unlimited companies
  private readonly baseColors = [
    { h: 210, s: 79, l: 57 },  // Blue
    { h: 354, s: 100, l: 69 }, // Red/Pink
    { h: 174, s: 50, l: 52 },  // Teal
    { h: 45, s: 100, l: 67 },  // Yellow/Gold
    { h: 262, s: 100, l: 70 }, // Purple
    { h: 120, s: 50, l: 50 },  // Green
    { h: 30, s: 100, l: 60 },  // Orange
    { h: 300, s: 60, l: 60 },  // Magenta
    { h: 190, s: 80, l: 45 },  // Cyan
    { h: 0, s: 70, l: 55 },    // Dark Red
    { h: 240, s: 60, l: 60 },  // Indigo
    { h: 60, s: 70, l: 50 },   // Olive
    { h: 330, s: 70, l: 60 },  // Rose
    { h: 150, s: 60, l: 45 },  // Sea Green
    { h: 280, s: 50, l: 55 },  // Violet
    { h: 15, s: 80, l: 55 },   // Coral
  ];

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

  private getColor(seriesId: string, index: number): { solid: string; dashed: string } {
    // Check cache first
    if (this.colorCache.has(seriesId)) {
      return this.colorCache.get(seriesId)!;
    }

    // Generate color based on index
    const baseColor = this.baseColors[index % this.baseColors.length];

    // Add slight variation for colors that repeat
    const variation = Math.floor(index / this.baseColors.length) * 15;
    const h = (baseColor.h + variation) % 360;
    const s = Math.max(30, baseColor.s - variation);
    const l = Math.min(75, baseColor.l + (variation / 2));

    const color = {
      solid: `hsla(${h}, ${s}%, ${l}%, 1)`,
      dashed: `hsla(${h}, ${s}%, ${l}%, 0.7)`,
    };

    this.colorCache.set(seriesId, color);
    return color;
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

    for (const [index, seriesId] of allSeriesIds.entries()) {
      const color = this.getColor(seriesId, index);

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
