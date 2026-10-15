export interface Candle {
  open: number
  high: number
  low: number
  close: number
  volume: number
  delta: number
}

export interface DepthLevel {
  price: number
  size: number
  side: 'bid' | 'ask'
}

export interface ActiveWall {
  price: number
  side: string
  size: number
  trust_score: number
  is_present: boolean
}

export interface Probabilities {
  down: number
  neutral: number
  up: number
}

export interface ShapFeature {
  feature: string
  shap_value: number
}

export interface ReplayFrame {
  index: number
  total_steps: number
  timestamp_ms: number
  mid_price: number
  best_bid: number
  best_ask: number
  spread: number
  spread_bps: number
  candle: Candle
  depth_levels: DepthLevel[]
  active_walls: ActiveWall[]
  ofi: number
  depth_imbalance: number
  probabilities: Probabilities
  expected_value_bps: number
  action: number // 1: LONG, -1: SHORT, 0: ABSTAIN
  action_label: string
  diagnostic_summary: string
  shap_features: ShapFeature[]
}

export interface Scenario {
  id: string
  name: string
  type: string
  badge: string
  description: string
}

export interface ReplayResponse {
  scenario: string
  offset: number
  limit: number
  total_steps: number
  frames: ReplayFrame[]
  candles: any[]
}
