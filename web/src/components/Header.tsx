import React from 'react'
import { Sun, Moon } from 'lucide-react'
import type { Scenario } from '../types/replay'

interface HeaderProps {
  currentStep: number
  totalSteps: number
  selectedScenario: Scenario | null
  theme: 'light' | 'dark'
  onToggleTheme: () => void
}

export const Header: React.FC<HeaderProps> = ({
  currentStep,
  totalSteps,
  selectedScenario,
  theme,
  onToggleTheme,
}) => {
  const isLight = theme === 'light'

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

      <div className="header-right">
        <div style={{ display: 'flex', gap: '14px', fontFamily: 'var(--font-mono)', fontSize: '11px', marginRight: '6px' }}>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>STEP: </span>
            <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
              {currentStep + 1}/{totalSteps || 1}
            </span>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>ENGINE: </span>
            <span style={{ fontWeight: 600, color: 'var(--accent-blue)' }}>
              ONLINE (:8000)
            </span>
          </div>
        </div>

        {/* Theme Toggle Button */}
        <button
          className="theme-toggle-btn"
          onClick={onToggleTheme}
          title={`Switch to ${isLight ? 'Dark Obsidian' : 'Light Frosted Glass'} theme`}
        >
          {isLight ? <Moon size={13} style={{ color: '#2563EB' }} /> : <Sun size={13} style={{ color: '#F0B90B' }} />}
          <span>{isLight ? 'DARK OBSIDIAN' : 'LIGHT FROST'}</span>
        </button>
      </div>
    </header>
  )
}

