import React from 'react'
import { Play, Pause, ChevronLeft, ChevronRight, RotateCcw, Cpu } from 'lucide-react'
import type { ModelInfo, Scenario } from '../types/replay'

interface ControlsProps {
  scenarios: Scenario[]
  selectedScenarioId: string
  onSelectScenario: (id: string) => void
  models: ModelInfo[]
  selectedModelId: string
  onSelectModel: (id: string) => void
  currentStep: number
  totalSteps: number
  onStepChange: (step: number) => void
  isPlaying: boolean
  onTogglePlay: () => void
  playSpeed: string
  onSelectSpeed: (speed: string) => void
  showFootprintHeatmap: boolean
  onToggleFootprintHeatmap: () => void
  windowSize: number
  onChangeWindowSize: (size: number) => void
  costHurdleBps: number
  onChangeCostHurdleBps: (val: number) => void
}

export const Controls: React.FC<ControlsProps> = ({
  scenarios,
  selectedScenarioId,
  onSelectScenario,
  models,
  selectedModelId,
  onSelectModel,
  currentStep,
  totalSteps,
  onStepChange,
  isPlaying,
  onTogglePlay,
  playSpeed,
  onSelectSpeed,
  showFootprintHeatmap,
  onToggleFootprintHeatmap,
  windowSize,
  onChangeWindowSize,
  costHurdleBps,
  onChangeCostHurdleBps,
}) => {
  const activeModel = models.find((m) => m.id === selectedModelId)

  return (
    <div className="controls-panel">
      {/* 0. Model Architecture Selector */}
      <div>
        <div className="control-group-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Cpu size={12} style={{ color: 'var(--accent-blue)' }} />
          <span>Model Architecture</span>
        </div>
        <select
          className="swiss-select"
          value={selectedModelId}
          onChange={(e) => onSelectModel(e.target.value)}
          style={{ marginBottom: '6px', fontWeight: 600 }}
        >
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name} [{m.badge}]
            </option>
          ))}
        </select>
        {activeModel && (
          <div style={{ fontSize: '10px', color: '#94A3B8', fontFamily: 'var(--font-mono)', lineHeight: 1.4, padding: '4px 6px', background: 'rgba(255,255,255,0.02)', borderRadius: '3px', border: '1px solid var(--border-hairline)' }}>
            <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>Acc: {(activeModel.accuracy * 100).toFixed(1)}%</span> |{' '}
            <span style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>F1: {activeModel.macro_f1.toFixed(3)}</span> |{' '}
            <span>Brier: {activeModel.brier_score.toFixed(3)}</span>
          </div>
        )}
      </div>

      {/* 1. Feed / Scenario Selector */}
      <div>
        <div className="control-group-title">Data Feed Stream</div>
        <select
          className="swiss-select"
          value={selectedScenarioId}
          onChange={(e) => onSelectScenario(e.target.value)}
        >
          {scenarios.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name}
            </option>
          ))}
        </select>
      </div>

      {/* 2. Replay Playback */}
      <div>
        <div className="control-group-title">Replay Timeline</div>
        <div className="btn-row" style={{ marginBottom: '10px' }}>
          <button
            className={`swiss-btn ${isPlaying ? 'active' : ''}`}
            onClick={onTogglePlay}
            style={{ flex: 1.4 }}
          >
            {isPlaying ? <Pause size={13} /> : <Play size={13} />}
            <span>{isPlaying ? 'PAUSE' : 'AUTO-PLAY'}</span>
          </button>

          <button
            className="swiss-btn"
            onClick={() => onStepChange(Math.max(0, currentStep - 1))}
            title="Step Backward"
          >
            <ChevronLeft size={14} />
          </button>

          <button
            className="swiss-btn"
            onClick={() => onStepChange(Math.min(totalSteps - 1, currentStep + 1))}
            title="Step Forward"
          >
            <ChevronRight size={14} />
          </button>

          <button
            className="swiss-btn"
            onClick={() => onStepChange(0)}
            title="Reset to Beginning"
          >
            <RotateCcw size={13} />
          </button>
        </div>

        {/* Playback Speed */}
        <div className="btn-row" style={{ marginBottom: '12px' }}>
          {['1x', '5x', '20x'].map((spd) => (
            <button
              key={spd}
              className={`swiss-btn ${playSpeed === spd ? 'active' : ''}`}
              onClick={() => onSelectSpeed(spd)}
              style={{ flex: 1, padding: '4px 8px' }}
            >
              {spd}
            </button>
          ))}
        </div>

        {/* Scrubber */}
        <div className="slider-container">
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#94A3B8', fontFamily: 'var(--font-mono)' }}>
            <span>SCRUBBER</span>
            <span>{currentStep + 1} / {totalSteps || 1}</span>
          </div>
          <input
            type="range"
            min={0}
            max={Math.max(0, totalSteps - 1)}
            value={currentStep}
            onChange={(e) => onStepChange(parseInt(e.target.value, 10))}
          />
        </div>
      </div>

      {/* 3. Candlestick & Heatmap Configuration */}
      <div>
        <div className="control-group-title">Chart Layering</div>
        <button
          className={`swiss-btn ${showFootprintHeatmap ? 'active' : ''}`}
          onClick={onToggleFootprintHeatmap}
          style={{ width: '100%', marginBottom: '10px' }}
        >
          <span>HEATMAP IN CANDLE: {showFootprintHeatmap ? 'ENABLED' : 'DISABLED'}</span>
        </button>

        <div className="slider-container">
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#94A3B8', fontFamily: 'var(--font-mono)' }}>
            <span>HISTORY WINDOW</span>
            <span>{windowSize} STEPS</span>
          </div>
          <input
            type="range"
            min={15}
            max={50}
            value={windowSize}
            onChange={(e) => onChangeWindowSize(parseInt(e.target.value, 10))}
          />
        </div>
      </div>

      {/* 4. Risk & Cost Parameters */}
      <div>
        <div className="control-group-title">Risk & Cost Friction</div>
        <div className="slider-container">
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#94A3B8', fontFamily: 'var(--font-mono)' }}>
            <span>TAKER FRICTION HURDLE</span>
            <span>{costHurdleBps.toFixed(1)} BPS</span>
          </div>
          <input
            type="range"
            min={0.5}
            max={5.0}
            step={0.5}
            value={costHurdleBps}
            onChange={(e) => onChangeCostHurdleBps(parseFloat(e.target.value))}
          />
        </div>
      </div>
    </div>
  )
}
