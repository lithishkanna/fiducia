import React from 'react'
import type { Scenario } from '../types/replay'

interface HeaderProps {
  currentStep: number
  totalSteps: number
  selectedScenario: Scenario | null
}

export const Header: React.FC<HeaderProps> = ({
  currentStep,
  totalSteps,
  selectedScenario,
}) => {
  return (
    <header className="swiss-header">
      <div className="header-left">
        <div className="brand-title">
          <span className="status-dot"></span>
          <span>FIDUCIA // LIQUIDITY TRUST INTELLIGENCE</span>
        </div>
        {selectedScenario && (
          <span className="badge-tag badge-blue">
            {selectedScenario.badge}
          </span>
        )}
        <span className="badge-tag badge-dark">
          BTC/USDT PERPETUAL
        </span>
      </div>

      <div className="header-stats">
        <div>
          <span className="header-stat-label">TIMESTEP:</span>
          <span className="header-stat-value">
            {currentStep + 1} / {totalSteps || 1}
          </span>
        </div>
        <div>
          <span className="header-stat-label">FEED:</span>
          <span className="header-stat-value" style={{ color: '#10B981' }}>
            {selectedScenario?.type === 'real' ? 'LIVE BINANCE L2 (2026.09.01)' : 'CURATED DEMO SCENARIO'}
          </span>
        </div>
        <div>
          <span className="header-stat-label">ENGINE:</span>
          <span className="header-stat-value" style={{ color: '#3B82F6' }}>
            ONLINE (:8000)
          </span>
        </div>
      </div>
    </header>
  )
}
