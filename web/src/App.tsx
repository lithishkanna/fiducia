import { useEffect, useState, useRef, useCallback } from 'react'
import { Header } from './components/Header'
import { DemoBar } from './components/DemoBar'
import { DecisionBanner } from './components/DecisionBanner'
import { MetricCards } from './components/MetricCards'
import { HeatmapChart } from './components/HeatmapChart'
import { EquityPnLChart } from './components/EquityPnLChart'
import { DepthLadderChart } from './components/DepthLadderChart'
import { SignalTimeChart } from './components/SignalTimeChart'
import { Controls } from './components/Controls'
import { DiagnosticPanel } from './components/DiagnosticPanel'
import { fetchScenarios, fetchModels, fetchReplay } from './api/client'
import type { Scenario, ModelInfo, ReplayFrame } from './types/replay'
import { Flame, TrendingUp, Layers, Activity } from 'lucide-react'

export function App() {
  const [theme, setTheme] = useState<'light' | 'dark'>('light')
  const [activeGraphView, setActiveGraphView] = useState<'heatmap' | 'pnl' | 'depth' | 'signals'>('heatmap')

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

  // Guided Demo Tour state
  const [isTourActive, setIsTourActive] = useState<boolean>(false)
  const [tourNarration, setTourNarration] = useState<string | null>(null)

  const playTimerRef = useRef<number | null>(null)

  // Apply theme to document element
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'))
  }

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
    setIsTourActive(false)
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
            setIsTourActive(false)
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

  // 4. Guided Tour Milestone Narration Engine
  useEffect(() => {
    if (!isTourActive || frames.length === 0) {
      setTourNarration(null)
      return
    }

    const frame = frames[currentStep]
    if (!frame) return

    // Contextual explanation based on microstructure features
    const walls = frame.active_walls || []
    const fleetWall = walls.find((w) => w.trust_score < 35)
    const authWall = walls.find((w) => w.trust_score >= 65)

    if (fleetWall) {
      setTourNarration(
        `[FLEETING WALL DETECTED] ${fleetWall.size.toFixed(1)} BTC ${fleetWall.side.toUpperCase()} wall at $${fleetWall.price.toFixed(2)} scored ${fleetWall.trust_score.toFixed(0)}/100 trust. Model detects spoof flicker and abstains from fake breakout.`
      )
    } else if (authWall) {
      setTourNarration(
        `[AUTHENTIC INSTITUTIONAL WALL] ${authWall.size.toFixed(1)} BTC ${authWall.side.toUpperCase()} wall at $${authWall.price.toFixed(2)} scored ${authWall.trust_score.toFixed(0)}/100 trust. Persistent limit depth confirms genuine support.`
      )
    } else if (frame.action === 1) {
      setTourNarration(
        `[TRADE LONG AUTHORIZED] Net EV (+${frame.expected_value_bps.toFixed(2)} bps) cleared the 2.0 bps taker friction hurdle with P(Up)=${(frame.probabilities.up * 100).toFixed(0)}%. Order executed.`
      )
    } else if (frame.action === -1) {
      setTourNarration(
        `[TRADE SHORT AUTHORIZED] Aggressive sell sweep detected. Net EV (+${frame.expected_value_bps.toFixed(2)} bps) cleared friction barrier with P(Down)=${(frame.probabilities.down * 100).toFixed(0)}%.`
      )
    } else {
      setTourNarration(
        `[DISCIPLINED CAPITAL PRESERVATION] Choppy depth profile. Net EV (${frame.expected_value_bps.toFixed(2)} bps) does not exceed the 2.0 bps friction barrier. System ABSTAINS to protect capital.`
      )
    }
  }, [isTourActive, currentStep, frames])

  const toggleTour = () => {
    if (!isTourActive) {
      setIsTourActive(true)
      setIsPlaying(true)
      setPlaySpeed('5x')
      if (currentStep >= frames.length - 10) {
        setCurrentStep(0)
      }
    } else {
      setIsTourActive(false)
      setIsPlaying(false)
    }
  }

  const selectedScenario = scenarios.find((s) => s.id === selectedScenarioId) || null
  const currentFrame = frames[currentStep] || null

  return (
    <div className="app-container">
      {/* Swiss Frosted Header with Theme Switcher */}
      <Header
        currentStep={currentStep}
        totalSteps={frames.length}
        selectedScenario={selectedScenario}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      {/* Institutional Demo Presentation Ribbon */}
      <DemoBar
        selectedScenarioId={selectedScenarioId}
        onSelectScenario={(id) => {
          setSelectedScenarioId(id)
          setIsTourActive(false)
        }}
        isTourActive={isTourActive}
        onToggleTour={toggleTour}
        tourNarration={tourNarration}
      />

      {error ? (
        <div style={{ background: 'rgba(242, 54, 69, 0.1)', border: '1px solid #F23645', padding: '16px', borderRadius: '4px', color: '#F23645', margin: '20px 0' }}>
          <strong>Connection Error:</strong> {error}
          <div style={{ marginTop: '8px', fontSize: '11px', color: 'var(--text-muted)' }}>
            Ensure the Python ASGI backend is running with: <code>python -m uvicorn web_api.server:app --port 8000</code>
          </div>
        </div>
      ) : isLoading ? (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
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

          {/* Graph View Navigation Bar */}
          <div className="graph-nav-bar">
            <div className="graph-nav-tabs">
              <button
                className={`graph-nav-btn ${activeGraphView === 'heatmap' ? 'active' : ''}`}
                onClick={() => setActiveGraphView('heatmap')}
              >
                <Flame size={13} style={{ color: '#F0B90B' }} />
                <span>1. L2 HEATMAP & FOOTPRINT CANDLESTICKS</span>
              </button>

              <button
                className={`graph-nav-btn ${activeGraphView === 'pnl' ? 'active' : ''}`}
                onClick={() => setActiveGraphView('pnl')}
              >
                <TrendingUp size={13} style={{ color: '#10B981' }} />
                <span>2. STRATEGY EQUITY & PNL CURVE</span>
              </button>

              <button
                className={`graph-nav-btn ${activeGraphView === 'depth' ? 'active' : ''}`}
                onClick={() => setActiveGraphView('depth')}
              >
                <Layers size={13} style={{ color: '#06B6D4' }} />
                <span>3. DEPTH OF MARKET (DOM) LADDER</span>
              </button>

              <button
                className={`graph-nav-btn ${activeGraphView === 'signals' ? 'active' : ''}`}
                onClick={() => setActiveGraphView('signals')}
              >
                <Activity size={13} style={{ color: '#3B82F6' }} />
                <span>4. MICROSTRUCTURE SIGNALS & EV HURDLE</span>
              </button>
            </div>

            <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              SWISS GLASS ENGINE // HIGH-DPI CANVAS
            </div>
          </div>

          {/* Main Stage: Dynamic Graph + Controls */}
          <div className="main-stage">
            {activeGraphView === 'heatmap' && (
              <HeatmapChart
                frames={frames}
                currentIdx={currentStep}
                windowSize={windowSize}
                showFootprintHeatmap={showFootprintHeatmap}
                theme={theme}
              />
            )}

            {activeGraphView === 'pnl' && (
              <EquityPnLChart
                frames={frames}
                currentIdx={currentStep}
                costHurdleBps={costHurdleBps}
                theme={theme}
              />
            )}

            {activeGraphView === 'depth' && (
              <DepthLadderChart
                frame={currentFrame}
                theme={theme}
              />
            )}

            {activeGraphView === 'signals' && (
              <SignalTimeChart
                frames={frames}
                currentIdx={currentStep}
                costHurdleBps={costHurdleBps}
                theme={theme}
              />
            )}

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
