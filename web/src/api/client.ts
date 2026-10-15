import type { ModelInfo, ReplayResponse, Scenario } from '../types/replay'

const API_BASE = '/api'

export async function fetchHealth(): Promise<{ status: string; service: string }> {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`)
  return res.json()
}

export async function fetchScenarios(): Promise<Scenario[]> {
  const res = await fetch(`${API_BASE}/scenarios`)
  if (!res.ok) throw new Error(`Failed to fetch scenarios: ${res.statusText}`)
  const data = await res.json()
  return data.scenarios
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${API_BASE}/models`)
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`)
  const data = await res.json()
  return data.models
}

export async function fetchReplay(
  scenario: string,
  offset: number = 0,
  limit: number = 150,
  candleSteps: number = 1,
  model: string = 'ensemble'
): Promise<ReplayResponse> {
  const url = `${API_BASE}/replay?scenario=${encodeURIComponent(scenario)}&offset=${offset}&limit=${limit}&candle_steps=${candleSteps}&model=${encodeURIComponent(model)}`
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to fetch replay data: ${res.statusText}`)
  return res.json()
}
