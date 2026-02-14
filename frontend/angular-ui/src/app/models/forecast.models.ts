export interface PipelineSummary {
  rows: number;
  unique_series: number;
  start: string;
  end: string;
  trained_models: string[];
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

export interface ForecastResponse {
  records: ForecastRecord[];
  count: number;
}
