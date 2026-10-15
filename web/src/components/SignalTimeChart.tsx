import React, { useEffect, useRef } from 'react'
import type { ReplayFrame } from '../types/replay'

interface SignalTimeChartProps {
  frames: ReplayFrame[]
  currentIdx: number
  costHurdleBps?: number
  theme?: 'light' | 'dark'
}

export const SignalTimeChart: React.FC<SignalTimeChartProps> = ({
  frames,
  currentIdx,
  costHurdleBps = 2.0,
  theme = 'dark',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const isLight = theme === 'light'

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || frames.length === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const width = canvas.clientWidth
    const height = canvas.clientHeight

    canvas.width = width * dpr
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)

    // Clear background
    ctx.fillStyle = isLight ? '#FFFFFF' : '#0B0E14'
    ctx.fillRect(0, 0, width, height)

    const padding = { top: 25, right: 80, bottom: 40, left: 20 }
    const plotWidth = width - padding.left - padding.right
    const totalHeight = height - padding.top - padding.bottom

    // Split into top subplot (Probabilities: 55% height) and bottom subplot (EV: 40% height)
    const probHeight = totalHeight * 0.55
    const gap = totalHeight * 0.08
    const evHeight = totalHeight * 0.37
    const evTop = padding.top + probHeight + gap

    const idxToX = (idx: number) => padding.left + (idx / Math.max(1, frames.length - 1)) * plotWidth

    // ----------------------------------------------------
    // SUBPLOT 1: CALIBRATED PROBABILITIES [0.0 - 1.0]
    // ----------------------------------------------------
    // Grid lines for probs
    ctx.strokeStyle = isLight ? 'rgba(0, 0, 0, 0.06)' : 'rgba(255, 255, 255, 0.05)'
    ctx.lineWidth = 1

    ;[0.0, 0.25, 0.5, 0.75, 1.0].forEach((pVal) => {
      const y = padding.top + probHeight - pVal * probHeight
      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(padding.left + plotWidth, y)
      ctx.stroke()

      ctx.fillStyle = isLight ? '#64748B' : '#94A3B8'
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillText(`${(pVal * 100).toFixed(0)}%`, padding.left + plotWidth + 8, y + 3)
    })

    // Subplot 1 title
    ctx.fillStyle = isLight ? '#334155' : '#CBD5E1'
    ctx.font = 'bold 10px "JetBrains Mono", monospace'
    ctx.fillText('MODEL CALIBRATED PROBABILITIES [DOWN vs NEUTRAL vs UP]', padding.left + 6, padding.top + 14)

    // P(Neutral) line
    ctx.strokeStyle = isLight ? '#94A3B8' : '#475569'
    ctx.lineWidth = 1.2
    ctx.beginPath()
    frames.forEach((f, idx) => {
      const x = idxToX(idx)
      const y = padding.top + probHeight - f.probabilities.neutral * probHeight
      if (idx === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // P(Up) line
    ctx.strokeStyle = isLight ? '#059669' : '#10B981'
    ctx.lineWidth = 2.0
    ctx.beginPath()
    frames.forEach((f, idx) => {
      const x = idxToX(idx)
      const y = padding.top + probHeight - f.probabilities.up * probHeight
      if (idx === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // P(Down) line
    ctx.strokeStyle = isLight ? '#E11D48' : '#F43F5E'
    ctx.lineWidth = 2.0
    ctx.beginPath()
    frames.forEach((f, idx) => {
      const x = idxToX(idx)
      const y = padding.top + probHeight - f.probabilities.down * probHeight
      if (idx === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // ----------------------------------------------------
    // SUBPLOT 2: NET EXPECTED VALUE (EV in bps)
    // ----------------------------------------------------
    const maxEv = 8.0
    const minEv = -8.0
    const evSpan = maxEv - minEv
    const evZeroY = evTop + evHeight * (maxEv / evSpan)

    ctx.fillStyle = isLight ? '#334155' : '#CBD5E1'
    ctx.font = 'bold 10px "JetBrains Mono", monospace'
    ctx.fillText(`NET EXPECTED VALUE (EV) vs ${costHurdleBps.toFixed(1)} BPS TAKER FRICTION`, padding.left + 6, evTop + 14)

    // Zero line
    ctx.strokeStyle = isLight ? 'rgba(0, 0, 0, 0.2)' : 'rgba(255, 255, 255, 0.2)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(padding.left, evZeroY)
    ctx.lineTo(padding.left + plotWidth, evZeroY)
    ctx.stroke()

    // Hurdle lines (+/- costHurdleBps)
    const hurdleUpY = evTop + evHeight * ((maxEv - costHurdleBps) / evSpan)
    const hurdleDownY = evTop + evHeight * ((maxEv + costHurdleBps) / evSpan)

    ctx.strokeStyle = 'rgba(245, 158, 11, 0.6)'
    ctx.setLineDash([3, 3])
    ctx.beginPath()
    ctx.moveTo(padding.left, hurdleUpY)
    ctx.lineTo(padding.left + plotWidth, hurdleUpY)
    ctx.moveTo(padding.left, hurdleDownY)
    ctx.lineTo(padding.left + plotWidth, hurdleDownY)
    ctx.stroke()
    ctx.setLineDash([])

    ctx.fillStyle = '#F59E0B'
    ctx.font = '9px "JetBrains Mono", monospace'
    ctx.fillText(`+${costHurdleBps.toFixed(1)} bps Hurdle`, padding.left + plotWidth + 8, hurdleUpY + 3)
    ctx.fillText(`-${costHurdleBps.toFixed(1)} bps Hurdle`, padding.left + plotWidth + 8, hurdleDownY + 3)

    // EV bars
    const barWidth = Math.max(2, (plotWidth / frames.length) * 0.7)
    frames.forEach((f, idx) => {
      const x = idxToX(idx) - barWidth / 2
      const evVal = f.expected_value_bps || 0
      const barH = (Math.abs(evVal) / evSpan) * evHeight

      if (f.action === 1) {
        ctx.fillStyle = '#10B981'
        ctx.fillRect(x, evZeroY - barH, barWidth, barH)
      } else if (f.action === -1) {
        ctx.fillStyle = '#F43F5E'
        ctx.fillRect(x, evZeroY, barWidth, barH)
      } else {
        ctx.fillStyle = isLight ? '#94A3B8' : '#334155'
        if (evVal >= 0) {
          ctx.fillRect(x, evZeroY - barH, barWidth, barH)
        } else {
          ctx.fillRect(x, evZeroY, barWidth, barH)
        }
      }
    })

    // Current cursor line
    const currX = idxToX(currentIdx)
    ctx.strokeStyle = '#F0B90B'
    ctx.lineWidth = 1
    ctx.setLineDash([3, 3])
    ctx.beginPath()
    ctx.moveTo(currX, padding.top)
    ctx.lineTo(currX, padding.top + totalHeight)
    ctx.stroke()
    ctx.setLineDash([])
  }, [frames, currentIdx, costHurdleBps, isLight])

  return (
    <div className="glass-panel" style={{ height: '540px', display: 'flex', flexDirection: 'column', padding: '14px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ color: '#3B82F6' }}>●</span>
          <span style={{ fontWeight: 700, fontSize: '13px', color: 'var(--text-primary)' }}>
            MICROSTRUCTURE SIGNAL DYNAMICS & CALIBRATED EV HURDLES
          </span>
          <span className="badge-tag badge-blue">TEMPERATURE SCALED T=10.0</span>
        </div>

        <div style={{ display: 'flex', gap: '16px', fontFamily: 'var(--font-mono)', fontSize: '10px' }}>
          <span style={{ color: 'var(--bullish)', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ width: '8px', height: '8px', background: 'var(--bullish)', borderRadius: '2px' }}></span> P(UP)
          </span>
          <span style={{ color: 'var(--bearish)', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ width: '8px', height: '8px', background: 'var(--bearish)', borderRadius: '2px' }}></span> P(DOWN)
          </span>
          <span style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ width: '8px', height: '8px', background: 'var(--text-muted)', borderRadius: '2px' }}></span> P(NEUTRAL)
          </span>
          <span style={{ color: '#F59E0B', display: 'flex', alignItems: 'center', gap: '4px' }}>
            --- 2.0 BPS HURDLE
          </span>
        </div>
      </div>

      <div style={{ flex: 1, position: 'relative' }}>
        <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>
    </div>
  )
}
