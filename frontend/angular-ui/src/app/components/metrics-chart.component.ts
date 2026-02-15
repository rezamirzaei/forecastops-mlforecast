import { Component, Input, OnChanges, SimpleChanges, ViewChild, ElementRef, AfterViewInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Chart, registerables } from 'chart.js';

import { AccuracyMetric } from '../models/forecast.models';

Chart.register(...registerables);

@Component({
  selector: 'app-metrics-chart',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="chart-container" *ngIf="metrics.length > 0">
      <canvas #chartCanvas></canvas>
    </div>
  `,
  styles: [`
    .chart-container {
      position: relative;
      height: 300px;
      width: 100%;
      margin: 1rem 0;
    }
  `]
})
export class MetricsChartComponent implements AfterViewInit, OnChanges, OnDestroy {
  @Input() metrics: AccuracyMetric[] = [];
  @ViewChild('chartCanvas') chartCanvas!: ElementRef<HTMLCanvasElement>;

  private chart: Chart | null = null;

  ngAfterViewInit(): void {
    if (this.metrics.length > 0) {
      this.createChart();
    }
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['metrics'] && this.chartCanvas) {
      this.updateChart();
    }
  }

  ngOnDestroy(): void {
    this.chart?.destroy();
  }

  private createChart(): void {
    if (!this.chartCanvas?.nativeElement) return;

    const sortedMetrics = [...this.metrics].sort((a, b) => a.smape - b.smape);

    this.chart = new Chart(this.chartCanvas.nativeElement, {
      type: 'bar',
      data: {
        labels: sortedMetrics.map(m => m.model),
        datasets: [
          {
            label: 'sMAPE',
            data: sortedMetrics.map(m => m.smape),
            backgroundColor: 'rgba(54, 162, 235, 0.7)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1
          },
          {
            label: 'WAPE',
            data: sortedMetrics.map(m => m.wape),
            backgroundColor: 'rgba(255, 99, 132, 0.7)',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, position: 'top' },
          title: { display: true, text: 'Model Comparison (Lower is Better)' }
        },
        scales: {
          y: { beginAtZero: true, title: { display: true, text: 'Error (%)' } }
        }
      }
    });
  }

  private updateChart(): void {
    this.chart?.destroy();
    this.createChart();
  }
}


