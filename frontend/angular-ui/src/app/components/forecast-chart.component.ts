import { Component, Input, OnChanges, SimpleChanges, ViewChild, ElementRef, AfterViewInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Chart, registerables } from 'chart.js';

import { ForecastRecord } from '../models/forecast.models';

Chart.register(...registerables);

@Component({
  selector: 'app-forecast-chart',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="chart-container" *ngIf="records.length > 0">
      <canvas #chartCanvas></canvas>
    </div>
    <p class="no-data" *ngIf="records.length === 0">
      No forecast data to display. Run a forecast first.
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
  @Input() selectedModel: string = '';
  @ViewChild('chartCanvas') chartCanvas!: ElementRef<HTMLCanvasElement>;

  private chart: Chart | null = null;

  private readonly colors = [
    { bg: 'rgba(54, 162, 235, 0.2)', border: 'rgba(54, 162, 235, 1)' },
    { bg: 'rgba(255, 99, 132, 0.2)', border: 'rgba(255, 99, 132, 1)' },
    { bg: 'rgba(75, 192, 192, 0.2)', border: 'rgba(75, 192, 192, 1)' },
    { bg: 'rgba(255, 206, 86, 0.2)', border: 'rgba(255, 206, 86, 1)' },
    { bg: 'rgba(153, 102, 255, 0.2)', border: 'rgba(153, 102, 255, 1)' },
  ];

  ngAfterViewInit(): void {
    if (this.records.length > 0) {
      this.createChart();
    }
  }

  ngOnChanges(changes: SimpleChanges): void {
    if ((changes['records'] || changes['selectedModel']) && this.chartCanvas) {
      this.updateChart();
    }
  }

  ngOnDestroy(): void {
    this.chart?.destroy();
  }

  private createChart(): void {
    if (!this.chartCanvas?.nativeElement || !this.records.length) return;

    // Filter by selected model or use first model
    const models = [...new Set(this.records.map(r => r.model_name))];
    const targetModel = this.selectedModel || models[0];
    const filteredRecords = this.records.filter(r => r.model_name === targetModel);

    // Group by series
    const seriesMap = new Map<string, ForecastRecord[]>();
    for (const record of filteredRecords) {
      const existing = seriesMap.get(record.unique_id) || [];
      existing.push(record);
      seriesMap.set(record.unique_id, existing);
    }

    // Get unique sorted dates
    const allDates = [...new Set(filteredRecords.map(r => r.ds))].sort();

    // Build datasets
    const datasets = Array.from(seriesMap.entries()).map(([seriesId, records], index) => {
      const color = this.colors[index % this.colors.length];
      const dataMap = new Map(records.map(r => [r.ds, r.value]));

      return {
        label: seriesId,
        data: allDates.map(date => dataMap.get(date) ?? null),
        backgroundColor: color.bg,
        borderColor: color.border,
        borderWidth: 2,
        fill: false,
        tension: 0.1,
      };
    });

    this.chart = new Chart(this.chartCanvas.nativeElement, {
      type: 'line',
      data: {
        labels: allDates.map(d => new Date(d).toLocaleDateString()),
        datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, position: 'top' },
          title: { display: true, text: `Forecast: ${targetModel}` }
        },
        scales: {
          x: { title: { display: true, text: 'Date' } },
          y: { title: { display: true, text: 'Predicted Value' } }
        }
      }
    });
  }

  private updateChart(): void {
    this.chart?.destroy();
    this.createChart();
  }
}


