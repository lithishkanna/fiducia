import React from 'react'
import type { ReplayFrame } from '../types/replay'

interface DecisionBannerProps {
  frame: ReplayFrame | null
  costHurdleBps: number
}

export const DecisionBanner: React.FC<DecisionBannerProps> = ({ frame, costHurdleBps }) => {
  if (!frame) return null

  const { action, action_label, expected_value_bps, probabilities } = frame

  const bannerClass = action === 1 ? 'long' : action === -1 ? 'short' : 'abstain'
  const icon = action === 1 ? '🟢' : action === -1 ? '🔴' : '⏸️'
  const evColor = expected_value_bps > 0 ? '#089981' : '#F23645'

  return (
    <div className={`decision-banner ${bannerClass}`}>
      <div className="decision-left">
        <span style={{ fontSize: '20px' }}>{icon}</span>
        <div>
          <div className="decision-sub">Model Decision State</div>
          <div className="decision-action-label">{action_label}</div>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '32px' }}>
        <div style={{ textAlign: 'right' }}>
          <div className="decision-sub">Net Expected Value</div>
          <div className="decision-ev" style={{ color: evColor }}>
            {expected_value_bps >= 0 ? `+${expected_value_bps.toFixed(2)}` : expected_value_bps.toFixed(2)} bps
            <span style={{ fontSize: '11px', color: '#94A3B8', marginLeft: '6px' }}>
              (Hurdle: {costHurdleBps.toFixed(1)} bps)
            </span>
          </div>
        </div>

        <div style={{ borderLeft: '1px solid rgba(255, 255, 255, 0.1)', paddingLeft: '24px' }}>
          <div className="decision-sub" style={{ marginBottom: '4px' }}>Calibrated Probabilities (T=10.0)</div>
          <div className="decision-probs">
            <span style={{ color: '#F23645' }}>DOWN: {(probabilities.down * 100).toFixed(1)}%</span>
            <span style={{ color: '#64748B' }}>|</span>
            <span style={{ color: '#94A3B8' }}>NEUTRAL: {(probabilities.neutral * 100).toFixed(1)}%</span>
            <span style={{ color: '#64748B' }}>|</span>
            <span style={{ color: '#089981' }}>UP: {(probabilities.up * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>
    </div>
  )
}
