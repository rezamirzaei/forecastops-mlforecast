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
  has_data?: boolean;
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

/** Backtest uses the same shape as forecast (fitted values for known dates). */
export type BacktestResponse = ForecastResponse;

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

// =============== Background Task Types ===============

export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed';
export type TaskType = 'data_update' | 'model_training' | 'full_pipeline';

export interface TaskInfo {
  task_id: string;
  task_type: TaskType;
  status: TaskStatus;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  progress: number;
  message: string;
  result: Record<string, unknown>;
  error: string | null;
  tickers_requested: string[];
}

export interface TaskResponse {
  task: TaskInfo;
  message: string;
}

export interface TasksListResponse {
  tasks: TaskInfo[];
  count: number;
}

export interface DataStats {
  rows: number;
  companies: number;
  start_date: string;
  end_date: string;
}

export interface SystemStatus {
  has_data: boolean;
  has_model: boolean;
  is_busy: boolean;
  current_task: TaskInfo | null;
  data_stats: DataStats | null;
  ready_for_predictions: boolean;
}

export interface TaskStartRequest {
  download: boolean;
  tickers: string[] | null;
}

export interface TrainingRequest {
  tickers: string[] | null;
  download: boolean;
}

