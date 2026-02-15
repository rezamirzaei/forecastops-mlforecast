export interface PipelineSummary {
  rows: number;
  unique_series: number;
  start: string;
  end: string;
  trained_models: string[];
}

export interface SeriesResponse {
  series: string[];
  count: number;
}

export interface SP500Company {
  ticker: string;
  symbol: string;
  name: string;
  sector: string;
}

export interface CompaniesResponse {
  companies: SP500Company[];
  sectors: string[];
  count: number;
}

export interface ForecastRequest {
  horizon: number;
  ids?: string[];
  levels: number[];
}

export interface ForecastRecord {
  unique_id: string;
  ds: string;
  model_name: string;
  value: number;
}

export interface HistoryRecord {
  unique_id: string;
  ds: string;
  value: number;
}

export interface ForecastResponse {
  records: ForecastRecord[];
  count: number;
}

export interface HistoryResponse {
  records: HistoryRecord[];
  count: number;
}

export interface AccuracyMetric {
  model: string;
  smape: number;
  wape: number;
}

export interface MetricsResponse {
  metrics: AccuracyMetric[];
  best_model: string | null;
  count: number;
}
