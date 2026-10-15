import React, { useState } from 'react'
import type { ModelInfo, ReplayFrame } from '../types/replay'

interface DiagnosticPanelProps {
  frame: ReplayFrame | null
  models?: ModelInfo[]
  selectedModelId?: string
}

export const DiagnosticPanel: React.FC<DiagnosticPanelProps> = ({ frame, models = [], selectedModelId = 'ensemble' }) => {
  const [activeTab, setActiveTab] = useState<'narrative' | 'shap' | 'pipeline' | 'models'>('narrative')

  if (!frame) return null

  const { diagnostic_summary, active_walls, shap_features, mid_price } = frame
  const currentModel = models.find((m) => m.id === selectedModelId)

  return (
    <div className="tabs-container">
      <div className="tab-headers">
        <button
          className={`tab-btn ${activeTab === 'narrative' ? 'active' : ''}`}
          onClick={() => setActiveTab('narrative')}
        >
          📋 DIAGNOSTIC NARRATIVE & WALLS
        </button>
        <button
          className={`tab-btn ${activeTab === 'shap' ? 'active' : ''}`}
          onClick={() => setActiveTab('shap')}
        >
          📊 ORDER-FLOW SENSITIVITIES (SHAP)
        </button>
        <button
          className={`tab-btn ${activeTab === 'pipeline' ? 'active' : ''}`}
          onClick={() => setActiveTab('pipeline')}
        >
          📐 SYSTEM & CALIBRATION PIPELINE
        </button>
        <button
          className={`tab-btn ${activeTab === 'models' ? 'active' : ''}`}
          onClick={() => setActiveTab('models')}
        >
          🤖 MODEL ARCHITECTURES & BENCHMARK
        </button>
      </div>

      <div className="tab-content">
        {activeTab === 'narrative' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '16px' }}>
            <div className="swiss-card" style={{ padding: '14px' }}>
              <div className="card-title">Plain-Language Model Diagnostic</div>
              <p style={{ fontSize: '12px', lineHeight: 1.6, color: '#CBD5E1', marginTop: '6px' }}>
                {diagnostic_summary ||
                  'Diagnostics reflect internal model feature sensitivities, calibrated probabilities, and depth persistence. Associated resistance or support is evaluated based on historical order placement patterns.'}
              </p>
            </div>

            <div className="swiss-card" style={{ padding: '14px' }}>
              <div className="card-title">Active Liquidity Walls</div>
              {active_walls && active_walls.length > 0 ? (
                <table className="swiss-table" style={{ marginTop: '6px' }}>
                  <thead>
                    <tr>
                      <th>SIDE</th>
                      <th>PRICE</th>
                      <th>DIST</th>
                      <th>SIZE</th>
                      <th>TRUST SCORE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {active_walls.map((w, idx) => {
                      const dist = ((w.price - mid_price) / 0.5).toFixed(1)
                      const isAuth = w.trust_score >= 65
                      const isFleet = w.trust_score < 35
                      const tagColor = isAuth ? '#10B981' : isFleet ? '#EF4444' : '#F59E0B'
                      const tagText = isAuth ? 'Authentic' : isFleet ? 'Fleeting' : 'Neutral'

                      return (
                        <tr key={idx}>
                          <td style={{ color: w.side.toLowerCase() === 'bid' ? '#06B6D4' : '#F43F5E', fontWeight: 600 }}>
                            {w.side.toUpperCase()}
                          </td>
                          <td>${w.price.toFixed(2)}</td>
                          <td>{dist} ticks</td>
                          <td>{w.size.toFixed(1)} BTC</td>
                          <td style={{ color: tagColor, fontWeight: 600 }}>
                            {w.trust_score.toFixed(0)}/100 ({tagText})
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              ) : (
                <div style={{ color: '#64748B', fontSize: '11px', marginTop: '8px' }}>
                  No dominant liquidity walls detected in top depth levels.
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'shap' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '16px' }}>
            <div className="swiss-card" style={{ padding: '14px' }}>
              <div className="card-title">Top Feature Drivers (Log-odds Impact)</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '10px' }}>
                {shap_features.map((feat, idx) => {
                  const isPos = feat.shap_value >= 0
                  const barColor = isPos ? '#089981' : '#F23645'
                  const pct = Math.min(100, Math.abs(feat.shap_value) * 180)

                  return (
                    <div key={idx} style={{ display: 'flex', alignItems: 'center', fontSize: '11px', fontFamily: 'var(--font-mono)' }}>
                      <div style={{ width: '180px', color: '#94A3B8' }}>{feat.feature}</div>
                      <div style={{ flex: 1, display: 'flex', alignItems: 'center', height: '14px', background: '#0B0E14', borderRadius: '2px', padding: '0 4px', position: 'relative' }}>
                        <div
                          style={{
                            height: '8px',
                            width: `${pct}%`,
                            background: barColor,
                            borderRadius: '2px',
                          }}
                        />
                      </div>
                      <div style={{ width: '65px', textAlign: 'right', color: barColor, fontWeight: 600 }}>
                        {isPos ? `+${feat.shap_value.toFixed(3)}` : feat.shap_value.toFixed(3)}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            <div className="swiss-card" style={{ padding: '14px' }}>
              <div className="card-title">Attribution Guide</div>
              <ul style={{ fontSize: '11px', lineHeight: 1.7, color: '#94A3B8', paddingLeft: '16px', marginTop: '8px' }}>
                <li><b style={{ color: '#089981' }}>Positive (Emerald)</b>: Lifts the model probability of upward price breakout.</li>
                <li><b style={{ color: '#F23645' }}>Negative (Crimson)</b>: Lifts the model probability of downward breakdown.</li>
                <li><b>Wall Trust</b>: Modulates whether resting limit orders represent genuine institutional support or fleeting bait.</li>
              </ul>
            </div>
          </div>
        )}

        {activeTab === 'pipeline' && (
          <div className="swiss-card" style={{ padding: '16px' }}>
            <div className="card-title">4-Stage Liquidity Trust Pipeline</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginTop: '10px', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
              <div style={{ background: '#0E131C', padding: '10px', border: '1px solid var(--border-hairline)', borderRadius: '3px' }}>
                <div style={{ color: '#3B82F6', fontWeight: 700, marginBottom: '4px' }}>1. L2 REPLAY</div>
                <div style={{ color: '#64748B', lineHeight: 1.5 }}>
                  Continuous @depth diff sequencing and @aggTrade synchronization on Binance Futures BTC/USDT.
                </div>
              </div>
              <div style={{ background: '#0E131C', padding: '10px', border: '1px solid var(--border-hairline)', borderRadius: '3px' }}>
                <div style={{ color: '#06B6D4', fontWeight: 700, marginBottom: '4px' }}>2. FEATURE LAYER</div>
                <div style={{ color: '#64748B', lineHeight: 1.5 }}>
                  Order Flow Imbalance (OFI), Multi-level Depth Imbalance, Volatility, and Center of Mass velocity.
                </div>
              </div>
              <div style={{ background: '#0E131C', padding: '10px', border: '1px solid var(--border-hairline)', borderRadius: '3px' }}>
                <div style={{ color: '#10B981', fontWeight: 700, marginBottom: '4px' }}>3. WALL TRUST</div>
                <div style={{ color: '#64748B', lineHeight: 1.5 }}>
                  Calibrated Gradient Boosting predicting 0–100 authenticity based on fill participation & flicker rates.
                </div>
              </div>
              <div style={{ background: '#0E131C', padding: '10px', border: '1px solid var(--border-hairline)', borderRadius: '3px' }}>
                <div style={{ color: '#F59E0B', fontWeight: 700, marginBottom: '4px' }}>4. CALIBRATED EV</div>
                <div style={{ color: '#64748B', lineHeight: 1.5 }}>
                  Temperature Scaled (T=10.0) probabilities enforcing strict 2.0 bps round-trip friction hurdles.
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'models' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {/* Active Model Spotlight */}
            {currentModel && (
              <div
                className="swiss-card"
                style={{
                  padding: '12px 16px',
                  background: 'rgba(59, 130, 246, 0.05)',
                  border: '1px solid rgba(59, 130, 246, 0.3)',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '13px', fontWeight: 700, color: '#F8FAFC' }}>
                      {currentModel.name}
                    </span>
                    <span className="badge" style={{ fontSize: '10px', background: '#1E293B', color: '#60A5FA', border: '1px solid #3B82F6', padding: '1px 6px', borderRadius: '3px' }}>
                      {currentModel.badge}
                    </span>
                  </div>
                  <div style={{ fontSize: '11px', color: '#94A3B8', marginTop: '3px' }}>
                    {currentModel.description}
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '16px', fontFamily: 'var(--font-mono)', fontSize: '11px', textAlign: 'right' }}>
                  <div>
                    <div style={{ color: '#64748B', fontSize: '10px' }}>TEST ACCURACY</div>
                    <div style={{ color: 'var(--accent-cyan)', fontWeight: 700, fontSize: '14px' }}>
                      {(currentModel.accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div style={{ color: '#64748B', fontSize: '10px' }}>MACRO F1</div>
                    <div style={{ color: 'var(--accent-emerald)', fontWeight: 700, fontSize: '14px' }}>
                      {currentModel.macro_f1.toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <div style={{ color: '#64748B', fontSize: '10px' }}>BRIER SCORE</div>
                    <div style={{ color: '#F59E0B', fontWeight: 700, fontSize: '14px' }}>
                      {currentModel.brier_score.toFixed(4)}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Benchmark Comparison Table */}
            <div className="swiss-card" style={{ padding: '14px' }}>
              <div className="card-title">Full Microstructure Model Comparison Benchmark</div>
              <table className="swiss-table" style={{ marginTop: '8px' }}>
                <thead>
                  <tr>
                    <th>MODEL ARCHITECTURE</th>
                    <th>TYPE / MECHANISM</th>
                    <th style={{ textAlign: 'center' }}>TEST ACCURACY</th>
                    <th style={{ textAlign: 'center' }}>MACRO F1</th>
                    <th style={{ textAlign: 'center' }}>BRIER SCORE</th>
                    <th>STATUS</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((m) => {
                    const isSelected = m.id === selectedModelId
                    return (
                      <tr
                        key={m.id}
                        style={{
                          background: isSelected ? 'rgba(59, 130, 246, 0.12)' : 'transparent',
                          fontWeight: isSelected ? 600 : 400,
                        }}
                      >
                        <td style={{ color: isSelected ? '#60A5FA' : '#F1F5F9' }}>
                          {m.name}
                        </td>
                        <td style={{ color: '#94A3B8', fontSize: '11px' }}>
                          {m.architecture}
                        </td>
                        <td style={{ textAlign: 'center', color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
                          {(m.accuracy * 100).toFixed(1)}%
                        </td>
                        <td style={{ textAlign: 'center', color: 'var(--accent-emerald)', fontFamily: 'var(--font-mono)' }}>
                          {m.macro_f1.toFixed(3)}
                        </td>
                        <td style={{ textAlign: 'center', color: '#F59E0B', fontFamily: 'var(--font-mono)' }}>
                          {m.brier_score.toFixed(4)}
                        </td>
                        <td>
                          {isSelected ? (
                            <span style={{ color: '#10B981', fontWeight: 700, fontSize: '10px' }}>● ACTIVE</span>
                          ) : (
                            <span style={{ color: '#64748B', fontSize: '10px' }}>READY</span>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>

  )
}
