import { useEffect, useState, useRef, useCallback } from 'react'
import { Header } from './components/Header'
import { DecisionBanner } from './components/DecisionBanner'
import { MetricCards } from './components/MetricCards'
import { HeatmapChart } from './components/HeatmapChart'
import { Controls } from './components/Controls'
import { DiagnosticPanel } from './components/DiagnosticPanel'
import { fetchScenarios, fetchModels, fetchReplay } from './api/client'
import type { Scenario, ModelInfo, ReplayFrame } from './types/replay'

export function App() {
  const [scenarios, setScenarios] = useState<Scenario[]>([])
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>('real_binance')
  const [models, setModels] = useState<ModelInfo[]>([])
  const [selectedModelId, setSelectedModelId] = useState<string>('ensemble')
  const [frames, setFrames] = useState<ReplayFrame[]>([])
  const [currentStep, setCurrentStep] = useState<number>(0)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)

  // Playback settings
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [playSpeed, setPlaySpeed] = useState<string>('5x')
  const [windowSize, setWindowSize] = useState<number>(35)
  const [showFootprintHeatmap, setShowFootprintHeatmap] = useState<boolean>(true)
  const [costHurdleBps, setCostHurdleBps] = useState<number>(2.0)

  const playTimerRef = useRef<number | null>(null)

  // 1. Load available scenarios and models on mount
  useEffect(() => {
    fetchScenarios()
      .then((data) => {
        setScenarios(data)
        if (data.length > 0) {
          setSelectedScenarioId(data[0].id)
        }
      })
      .catch((err) => {
        console.error('Failed to load scenarios:', err)
        setError('Failed to connect to backend API server at localhost:8000.')
      })

    fetchModels()
      .then((data) => {
        setModels(data)
        if (data.length > 0) {
          setSelectedModelId(data[0].id)
        }
      })
      .catch((err) => {
        console.error('Failed to load models:', err)
      })
  }, [])

  // 2. Load replay frames whenever selected scenario OR selected model changes
  const loadScenarioData = useCallback(async (scenarioId: string, modelId: string) => {
    setIsLoading(true)
    setError(null)
    setIsPlaying(false)
    try {
      const data = await fetchReplay(scenarioId, 0, 160, 1, modelId)
      setFrames(data.frames)
      setCurrentStep(0)
    } catch (err: any) {
      console.error('Failed to load replay frames:', err)
      setError(`Failed to load scenario ${scenarioId}: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    if (selectedScenarioId && selectedModelId) {
      loadScenarioData(selectedScenarioId, selectedModelId)
    }
  }, [selectedScenarioId, selectedModelId, loadScenarioData])

  // 3. Playback timer
  useEffect(() => {
    if (isPlaying) {
      const delayMs = playSpeed === '1x' ? 120 : playSpeed === '5x' ? 40 : 15
      playTimerRef.current = window.setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= frames.length - 1) {
            setIsPlaying(false)
            return prev
          }
          return prev + 1
        })
      }, delayMs)
    } else {
      if (playTimerRef.current !== null) {
        clearInterval(playTimerRef.current)
        playTimerRef.current = null
      }
    }

    return () => {
      if (playTimerRef.current !== null) {
        clearInterval(playTimerRef.current)
        playTimerRef.current = null
      }
    }
  }, [isPlaying, playSpeed, frames.length])

  const selectedScenario = scenarios.find((s) => s.id === selectedScenarioId) || null
  const currentFrame = frames[currentStep] || null

  return (
    <div className="app-container">
      {/* Swiss Header */}
      <Header
        currentStep={currentStep}
        totalSteps={frames.length}
        selectedScenario={selectedScenario}
      />

      {error ? (
        <div style={{ background: 'rgba(242, 54, 69, 0.1)', border: '1px solid #F23645', padding: '16px', borderRadius: '4px', color: '#F23645', margin: '20px 0' }}>
          <strong>Connection Error:</strong> {error}
          <div style={{ marginTop: '8px', fontSize: '11px', color: '#94A3B8' }}>
            Ensure the Python ASGI backend is running with: <code>python -m uvicorn web_api.server:app --port 8000</code>
          </div>
        </div>
      ) : isLoading ? (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px', color: '#94A3B8', fontFamily: 'var(--font-mono)' }}>
          <span className="status-dot" style={{ marginRight: '10px' }}></span>
          STREAMING BINANCE FUTURES L2 REPLAY & INFERENCE FRAMES ({selectedModelId.toUpperCase()})...
        </div>
      ) : (
        <>
          {/* Decision State Banner */}
          <DecisionBanner
            frame={currentFrame}
            costHurdleBps={costHurdleBps}
          />

          {/* Metric Cards Grid */}
          <MetricCards
            frame={currentFrame}
            costHurdleBps={costHurdleBps}
          />

          {/* Main Stage: Composite Chart + Controls */}
          <div className="main-stage">
            <HeatmapChart
              frames={frames}
              currentIdx={currentStep}
              windowSize={windowSize}
              showFootprintHeatmap={showFootprintHeatmap}
            />

            <Controls
              scenarios={scenarios}
              selectedScenarioId={selectedScenarioId}
              onSelectScenario={setSelectedScenarioId}
              models={models}
              selectedModelId={selectedModelId}
              onSelectModel={setSelectedModelId}
              currentStep={currentStep}
              totalSteps={frames.length}
              onStepChange={setCurrentStep}
              isPlaying={isPlaying}
              onTogglePlay={() => setIsPlaying(!isPlaying)}
              playSpeed={playSpeed}
              onSelectSpeed={setPlaySpeed}
              showFootprintHeatmap={showFootprintHeatmap}
              onToggleFootprintHeatmap={() => setShowFootprintHeatmap(!showFootprintHeatmap)}
              windowSize={windowSize}
              onChangeWindowSize={setWindowSize}
              costHurdleBps={costHurdleBps}
              onChangeCostHurdleBps={setCostHurdleBps}
            />
          </div>

          {/* Diagnostic "Why?" Panel with Model Architecture Tab */}
          <DiagnosticPanel
            frame={currentFrame}
            models={models}
            selectedModelId={selectedModelId}
          />

          {/* Regulatory & Institutional Non-Causal Disclaimer */}
          <div className="disclaimer-bar">
            ⚠️ <b>Note:</b> Diagnostics reflect internal model feature sensitivities, calibrated probabilities, and depth persistence.
            They do not constitute financial advice, trade recommendations, or representations of exchange order integrity.
          </div>
        </>
      )}
    </div>
  )
}

export default App
