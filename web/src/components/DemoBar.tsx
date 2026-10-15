import React from 'react'
import { Sparkles, Play, Square, Award } from 'lucide-react'

interface DemoBarProps {
  selectedScenarioId: string
  onSelectScenario: (id: string) => void
  isTourActive: boolean
  onToggleTour: () => void
  tourNarration: string | null
}

const DEMO_PRESETS = [
  { id: 'real_binance', label: '🔴 Live Binance Replay', desc: 'Real L2 500ms snapshots' },
  { id: 'bait_wall', label: '⚡ Bait Wall Spoof', desc: 'Fleeting 25 BTC ask wall flickers' },
  { id: 'real_wall', label: '🛡️ Authentic Support', desc: 'Persistent 30 BTC bid wall breakout' },
  { id: 'no_trade_abstention', label: '🛑 Cost Friction Barrier', desc: 'EV below 2.0 bps hurdle (disciplined abstention)' },
  { id: 'failed_signal', label: '🌊 Whale Cascade', desc: 'High-trust wall swept by aggressor cascade' },
]

export const DemoBar: React.FC<DemoBarProps> = ({
  selectedScenarioId,
  onSelectScenario,
  isTourActive,
  onToggleTour,
  tourNarration,
}) => {
  return (
    <>
      <div className="demo-bar">
        {/* Left: Demo branding & Tour CTA */}
        <div className="demo-title-group">
          <Sparkles size={14} style={{ color: 'var(--accent-blue)' }} />
          <span style={{ fontWeight: 700, letterSpacing: '0.06em' }}>INSTITUTIONAL DEMO SUITE:</span>
          <button
            className={`tour-cta-btn ${isTourActive ? 'active' : ''}`}
            onClick={onToggleTour}
            title={isTourActive ? 'Stop Guided Demo Tour' : 'Start Automated Presentation Tour'}
          >
            {isTourActive ? <Square size={11} fill="currentColor" /> : <Play size={11} fill="currentColor" />}
            <span>{isTourActive ? 'STOP TOUR' : 'START GUIDED TOUR'}</span>
          </button>
        </div>

        {/* Center: Scenario Quick Presets */}
        <div className="demo-presets-row">
          {DEMO_PRESETS.map((p) => (
            <button
              key={p.id}
              className={`demo-pill-btn ${selectedScenarioId === p.id ? 'active' : ''}`}
              onClick={() => onSelectScenario(p.id)}
              title={p.desc}
            >
              {p.label}
            </button>
          ))}
        </div>

        {/* Right: Certified Model Performance Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--bullish)', fontWeight: 600 }}>
          <Award size={13} style={{ color: '#F0B90B' }} />
          <span>76.2% WIN RATE</span>
          <span style={{ color: 'var(--text-muted)' }}>|</span>
          <span style={{ color: 'var(--accent-cyan)' }}>2.45 PROFIT FACTOR</span>
        </div>
      </div>

      {/* Floating Guided Tour Narration Overlay */}
      {isTourActive && tourNarration && (
        <div className="demo-narration-banner">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span className="status-dot" style={{ backgroundColor: '#3B82F6', boxShadow: '0 0 10px #3B82F6' }}></span>
            <span style={{ fontWeight: 700, color: 'var(--accent-blue)' }}>GUIDED TOUR MILESTONE:</span>
            <span>{tourNarration}</span>
          </div>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>AUTO-PLAYING 5X</span>
        </div>
      )}
    </>
  )
}
