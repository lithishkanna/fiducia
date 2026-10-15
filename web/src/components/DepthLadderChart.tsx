import React, { useEffect, useRef } from 'react'
import type { ReplayFrame } from '../types/replay'

interface DepthLadderChartProps {
  frame: ReplayFrame | null
  theme?: 'light' | 'dark'
}

export const DepthLadderChart: React.FC<DepthLadderChartProps> = ({ frame, theme = 'dark' }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const isLight = theme === 'light'

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !frame) return

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

    const bids = frame.depth_levels.filter((d) => d.side === 'bid').sort((a, b) => b.price - a.price)
    const asks = frame.depth_levels.filter((d) => d.side === 'ask').sort((a, b) => a.price - b.price)

    if (bids.length === 0 || asks.length === 0) return

    // Calculate cumulative sizes
    let cumBid = 0
    const cumBids = bids.map((b) => {
      cumBid += b.size
      return { price: b.price, size: b.size, cum: cumBid }
    })

    let cumAsk = 0
    const cumAsks = asks.map((a) => {
      cumAsk += a.size
      return { price: a.price, size: a.size, cum: cumAsk }
    })

    const maxCum = Math.max(cumBid, cumAsk, 15)

    const padding = { top: 30, right: 30, bottom: 40, left: 30 }
    const plotWidth = width - padding.left - padding.right
    const plotHeight = height - padding.top - padding.bottom
    const midX = padding.left + plotWidth / 2

    // 1. Center Mid-Price divider
    ctx.strokeStyle = isLight ? 'rgba(0, 0, 0, 0.1)' : 'rgba(255, 255, 255, 0.1)'
    ctx.lineWidth = 1
    ctx.setLineDash([4, 4])
    ctx.beginPath()
    ctx.moveTo(midX, padding.top)
    ctx.lineTo(midX, padding.top + plotHeight)
    ctx.stroke()
    ctx.setLineDash([])

    // Mid price badge
    ctx.fillStyle = '#F0B90B'
    ctx.font = 'bold 11px "JetBrains Mono", monospace'
    ctx.textAlign = 'center'
    ctx.fillText(`MID: $${frame.mid_price.toFixed(2)}`, midX, padding.top - 10)

    // 2. Cumulative Bid Curve (Left to Center)
    const bidStepH = plotHeight / Math.max(1, cumBids.length)
    const cumBidGrad = ctx.createLinearGradient(midX, 0, padding.left, 0)
    cumBidGrad.addColorStop(0, isLight ? 'rgba(5, 150, 105, 0.35)' : 'rgba(6, 182, 212, 0.35)')
    cumBidGrad.addColorStop(1, 'rgba(6, 182, 212, 0.05)')

    ctx.fillStyle = cumBidGrad
    ctx.beginPath()
    ctx.moveTo(midX, padding.top)

    cumBids.forEach((b, i) => {
      const y = padding.top + i * bidStepH
      const w = (b.cum / maxCum) * (plotWidth / 2 - 20)
      const x = midX - w
      ctx.lineTo(x, y)
    })
    ctx.lineTo(midX, padding.top + plotHeight)
    ctx.closePath()
    ctx.fill()

    // Bid step line
    ctx.strokeStyle = isLight ? '#059669' : '#06B6D4'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(midX, padding.top)
    cumBids.forEach((b, i) => {
      const y = padding.top + i * bidStepH
      const w = (b.cum / maxCum) * (plotWidth / 2 - 20)
      const x = midX - w
      ctx.lineTo(x, y)
    })
    ctx.stroke()

    // 3. Cumulative Ask Curve (Center to Right)
    const askStepH = plotHeight / Math.max(1, cumAsks.length)
    const cumAskGrad = ctx.createLinearGradient(midX, 0, padding.left + plotWidth, 0)
    cumAskGrad.addColorStop(0, isLight ? 'rgba(225, 29, 72, 0.35)' : 'rgba(244, 63, 94, 0.35)')
    cumAskGrad.addColorStop(1, 'rgba(244, 63, 94, 0.05)')

    ctx.fillStyle = cumAskGrad
    ctx.beginPath()
    ctx.moveTo(midX, padding.top)

    cumAsks.forEach((a, i) => {
      const y = padding.top + i * askStepH
      const w = (a.cum / maxCum) * (plotWidth / 2 - 20)
      const x = midX + w
      ctx.lineTo(x, y)
    })
    ctx.lineTo(midX, padding.top + plotHeight)
    ctx.closePath()
    ctx.fill()

    // Ask step line
    ctx.strokeStyle = isLight ? '#E11D48' : '#F43F5E'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(midX, padding.top)
    cumAsks.forEach((a, i) => {
      const y = padding.top + i * askStepH
      const w = (a.cum / maxCum) * (plotWidth / 2 - 20)
      const x = midX + w
      ctx.lineTo(x, y)
    })
    ctx.stroke()

    // 4. Price & Size text annotations
    ctx.font = '10px "JetBrains Mono", monospace'
    cumBids.forEach((b, i) => {
      const y = padding.top + i * bidStepH + 12
      ctx.fillStyle = isLight ? '#065F46' : '#67E8F9'
      ctx.textAlign = 'right'
      ctx.fillText(`$${b.price.toFixed(2)} (${b.size.toFixed(1)} BTC)`, midX - 10, y)
    })

    cumAsks.forEach((a, i) => {
      const y = padding.top + i * askStepH + 12
      ctx.fillStyle = isLight ? '#9F1239' : '#FDA4AF'
      ctx.textAlign = 'left'
      ctx.fillText(`$${a.price.toFixed(2)} (${a.size.toFixed(1)} BTC)`, midX + 10, y)
    })

    // 5. Active Wall Highlighting
    if (frame.active_walls && frame.active_walls.length > 0) {
      frame.active_walls.forEach((wall) => {
        const isAuth = wall.trust_score >= 65
        const isFleet = wall.trust_score < 35
        const wallColor = isAuth ? '#10B981' : isFleet ? '#EF4444' : '#F59E0B'
        const badgeLabel = `WALL: ${wall.size.toFixed(1)} BTC [${wall.trust_score.toFixed(0)}/100 ${isAuth ? 'AUTHENTIC' : isFleet ? 'FLEETING' : 'NEUTRAL'}]`

        const isBid = wall.side.toLowerCase() === 'bid'
        const targetArr = isBid ? cumBids : cumAsks
        const matchIdx = targetArr.findIndex((lvl) => Math.abs(lvl.price - wall.price) < 0.26)
        if (matchIdx >= 0) {
          const y = padding.top + matchIdx * (isBid ? bidStepH : askStepH)
          ctx.strokeStyle = wallColor
          ctx.lineWidth = 2
          ctx.beginPath()
          ctx.moveTo(isBid ? padding.left : midX, y)
          ctx.lineTo(isBid ? midX : padding.left + plotWidth, y)
          ctx.stroke()

          ctx.fillStyle = wallColor
          ctx.textAlign = isBid ? 'left' : 'right'
          ctx.fillText(badgeLabel, isBid ? padding.left + 8 : padding.left + plotWidth - 8, y - 4)
        }
      })
    }
  }, [frame, isLight])

  if (!frame) return null

  const totalBid = frame.depth_levels.filter((d) => d.side === 'bid').reduce((acc, c) => acc + c.size, 0)
  const totalAsk = frame.depth_levels.filter((d) => d.side === 'ask').reduce((acc, c) => acc + c.size, 0)
  const totalDepth = totalBid + totalAsk || 1
  const bidPct = (totalBid / totalDepth) * 100
  const askPct = (totalAsk / totalDepth) * 100

  return (
    <div className="glass-panel" style={{ height: '540px', display: 'flex', flexDirection: 'column', padding: '14px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ color: '#06B6D4' }}>●</span>
          <span style={{ fontWeight: 700, fontSize: '13px', color: 'var(--text-primary)' }}>
            DEPTH OF MARKET (DOM) LIQUIDITY LADDER
          </span>
          <span className="badge-tag badge-blue">TOP-10 L2 CUMULATIVE DEPTH</span>
        </div>

        {/* Ratio bar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
          <span style={{ color: 'var(--bullish)', fontWeight: 600 }}>BIDS {bidPct.toFixed(0)}%</span>
          <div style={{ width: '120px', height: '8px', background: 'var(--bg-subtle)', borderRadius: '4px', overflow: 'hidden', display: 'flex' }}>
            <div style={{ width: `${bidPct}%`, background: 'var(--bullish)' }}></div>
            <div style={{ width: `${askPct}%`, background: 'var(--bearish)' }}></div>
          </div>
          <span style={{ color: 'var(--bearish)', fontWeight: 600 }}>ASKS {askPct.toFixed(0)}%</span>
        </div>
      </div>

      <div style={{ flex: 1, position: 'relative' }}>
        <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>
    </div>
  )
}
