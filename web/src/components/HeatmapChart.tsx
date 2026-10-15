import React, { useEffect, useRef, useState, useCallback } from 'react'
import type { ReplayFrame } from '../types/replay'

interface HeatmapChartProps {
  frames: ReplayFrame[]
  currentIdx: number
  windowSize?: number
  showFootprintHeatmap?: boolean
}

export const HeatmapChart: React.FC<HeatmapChartProps> = ({
  frames,
  currentIdx,
  windowSize = 35,
  showFootprintHeatmap = true,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [hoverInfo, setHoverInfo] = useState<{
    x: number
    y: number
    price: number
    timeLabel: string
    depthSize: number
    side?: string
  } | null>(null)

  // Determine historical window slice
  const startIdx = Math.max(0, currentIdx - windowSize + 1)
  const windowFrames = frames.slice(startIdx, currentIdx + 1)
  const currentFrame = frames[currentIdx] || null

  const renderChart = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !currentFrame || windowFrames.length === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const width = canvas.clientWidth
    const height = canvas.clientHeight

    // Scale canvas for high-DPI crisp rendering
    canvas.width = width * dpr
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)

    // Layout Dimensions
    const rightMargin = 85
    const bottomHeight = 90
    const mainHeight = height - bottomHeight - 20
    const plotWidth = width - rightMargin - 15
    const plotLeft = 15

    // Clear background
    ctx.fillStyle = '#0B0E14'
    ctx.fillRect(0, 0, width, height)

    // Price Grid Definition around current mid
    const currentMid = currentFrame.mid_price
    const priceSpanTicks = 24
    const tickSize = 0.5
    const maxPrice = currentMid + priceSpanTicks * tickSize
    const minPrice = currentMid - priceSpanTicks * tickSize
    const priceRange = maxPrice - minPrice

    const priceToY = (price: number) => {
      const norm = (maxPrice - price) / priceRange
      return 15 + norm * (mainHeight - 15)
    }

    const colWidth = plotWidth / windowFrames.length
    const candleWidth = Math.max(4, colWidth * 0.68)

    // 1. Render Background Liquidity Heatmap
    // Collect all price levels and normalize depth
    let maxAbsDepth = 12.0
    windowFrames.forEach((f) => {
      f.depth_levels.forEach((lvl) => {
        if (lvl.size > maxAbsDepth) maxAbsDepth = lvl.size
      })
    })

    windowFrames.forEach((frame, colIdx) => {
      const x = plotLeft + colIdx * colWidth

      // Depth lookup map for this step
      const depthMap: Record<number, { size: number; side: 'bid' | 'ask' }> = {}
      frame.depth_levels.forEach((lvl) => {
        const roundedP = Math.round(lvl.price * 2) / 2
        depthMap[roundedP] = lvl
      })

      // Draw background vertical depth strips
      for (let p = minPrice; p <= maxPrice; p += tickSize) {
        const roundedP = Math.round(p * 2) / 2
        const yTop = priceToY(roundedP + tickSize / 2)
        const yBottom = priceToY(roundedP - tickSize / 2)
        const cellHeight = Math.abs(yBottom - yTop)

        const depth = depthMap[roundedP]
        if (depth) {
          const intensity = Math.min(1.0, depth.size / (maxAbsDepth * 0.75))
          if (depth.side === 'bid') {
            // Cool cyan-blue
            ctx.fillStyle = `rgba(6, 182, 212, ${0.12 + intensity * 0.55})`
          } else {
            // Warm amber-red
            ctx.fillStyle = `rgba(239, 68, 68, ${0.12 + intensity * 0.55})`
          }
          ctx.fillRect(x, Math.min(yTop, yBottom), colWidth, cellHeight)
        }
      }
    })

    // 2. Grid lines & Price Axis
    ctx.strokeStyle = '#181F2C'
    ctx.lineWidth = 1
    ctx.fillStyle = '#64748B'
    ctx.font = '10px "JetBrains Mono", monospace'

    // Horizontal price grid lines
    const numPriceLines = 10
    const priceStep = priceRange / numPriceLines
    for (let i = 0; i <= numPriceLines; i++) {
      const p = minPrice + i * priceStep
      const y = priceToY(p)

      ctx.beginPath()
      ctx.moveTo(plotLeft, y)
      ctx.lineTo(plotLeft + plotWidth, y)
      ctx.stroke()

      // Price label on right
      ctx.fillText(`$${p.toFixed(2)}`, plotLeft + plotWidth + 8, y + 3)
    }

    // 3. Render Foreground Candlesticks (with Heatmap-Colored Footprint)
    windowFrames.forEach((frame, colIdx) => {
      const colCenterX = plotLeft + colIdx * colWidth + colWidth / 2
      const candle = frame.candle
      const isBullish = candle.close >= candle.open
      const contourColor = isBullish ? '#089981' : '#F23645'

      const yHigh = priceToY(candle.high)
      const yLow = priceToY(candle.low)
      const yOpen = priceToY(candle.open)
      const yClose = priceToY(candle.close)

      const bodyTop = Math.min(yOpen, yClose)
      const bodyBottom = Math.max(yOpen, yClose)
      const bodyHeight = Math.max(2, bodyBottom - bodyTop)

      // Draw Wick
      ctx.strokeStyle = isBullish ? '#10B981' : '#EF4444'
      ctx.lineWidth = 1.2
      ctx.beginPath()
      ctx.moveTo(colCenterX, yHigh)
      ctx.lineTo(colCenterX, bodyTop)
      ctx.moveTo(colCenterX, bodyBottom)
      ctx.lineTo(colCenterX, yLow)
      ctx.stroke()

      const candleLeft = colCenterX - candleWidth / 2

      if (showFootprintHeatmap) {
        // "HEATMAP IN CANDLE": Slices inside the candle body colored by resting liquidity & volume profile
        const depthMap: Record<number, { size: number; side: 'bid' | 'ask' }> = {}
        frame.depth_levels.forEach((lvl) => {
          depthMap[Math.round(lvl.price * 2) / 2] = lvl
        })

        // Draw segmented internal heatmap cells
        const minBodyPrice = Math.min(candle.open, candle.close)
        const maxBodyPrice = Math.max(candle.open, candle.close)

        for (let p = minBodyPrice; p <= maxBodyPrice + 0.01; p += tickSize) {
          const roundedP = Math.round(p * 2) / 2
          const sliceYTop = priceToY(roundedP + tickSize / 2)
          const sliceYBottom = priceToY(roundedP - tickSize / 2)
          const sliceH = Math.max(1, Math.abs(sliceYBottom - sliceYTop))

          const depth = depthMap[roundedP]
          if (depth) {
            const intensity = Math.min(1.0, depth.size / (maxAbsDepth * 0.75))
            if (depth.side === 'bid') {
              ctx.fillStyle = `rgba(14, 165, 233, ${0.4 + intensity * 0.55})` // Vibrant Cyan
            } else {
              ctx.fillStyle = `rgba(244, 63, 94, ${0.4 + intensity * 0.55})` // Vibrant Crimson
            }
          } else {
            // Neutral candle body fill
            ctx.fillStyle = isBullish ? 'rgba(8, 153, 129, 0.45)' : 'rgba(242, 54, 69, 0.45)'
          }

          // Clip slice inside candle body
          const yRender = Math.max(bodyTop, Math.min(sliceYTop, sliceYBottom))
          const hRender = Math.min(bodyHeight, sliceH)
          ctx.fillRect(candleLeft, yRender, candleWidth, hRender)
        }

        // Draw Candle Outline Contour
        ctx.strokeStyle = contourColor
        ctx.lineWidth = 1.6
        ctx.strokeRect(candleLeft, bodyTop, candleWidth, bodyHeight)
      } else {
        // Traditional solid candle
        ctx.fillStyle = isBullish ? 'rgba(8, 153, 129, 0.85)' : 'rgba(242, 54, 69, 0.85)'
        ctx.fillRect(candleLeft, bodyTop, candleWidth, bodyHeight)
        ctx.strokeStyle = contourColor
        ctx.lineWidth = 1.2
        ctx.strokeRect(candleLeft, bodyTop, candleWidth, bodyHeight)
      }
    })

    // 4. Tracking Dotted Mid-Price Line
    ctx.strokeStyle = '#F0B90B'
    ctx.lineWidth = 1.4
    ctx.setLineDash([3, 3])
    ctx.beginPath()
    windowFrames.forEach((frame, idx) => {
      const x = plotLeft + idx * colWidth + colWidth / 2
      const y = priceToY(frame.mid_price)
      if (idx === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()
    ctx.setLineDash([]) // Reset

    // 5. Active Liquidity Wall Lines & Badges
    const activeWalls = currentFrame.active_walls || []
    activeWalls.forEach((wall) => {
      const y = priceToY(wall.price)
      if (y < 15 || y > mainHeight) return

      let strokeColor = '#94A3B8'
      let badgeBg = '#1E2538'
      let badgeLabel = `WALL ${wall.size.toFixed(1)} BTC`

      if (wall.trust_score >= 65) {
        strokeColor = '#10B981'
        badgeBg = 'rgba(8, 153, 129, 0.25)'
        badgeLabel = `AUTHENTIC ${wall.trust_score.toFixed(0)}/100 (${wall.size.toFixed(1)} BTC)`
      } else if (wall.trust_score < 35) {
        strokeColor = '#EF4444'
        badgeBg = 'rgba(242, 54, 69, 0.25)'
        badgeLabel = `FLEETING ${wall.trust_score.toFixed(0)}/100 (${wall.size.toFixed(1)} BTC)`
      }

      // Horizontal wall guideline
      ctx.strokeStyle = strokeColor
      ctx.lineWidth = 1.8
      ctx.setLineDash([5, 4])
      ctx.beginPath()
      ctx.moveTo(plotLeft, y)
      ctx.lineTo(plotLeft + plotWidth, y)
      ctx.stroke()
      ctx.setLineDash([])

      // Right-side badge
      ctx.fillStyle = badgeBg
      ctx.fillRect(plotLeft + plotWidth - 195, y - 9, 190, 18)
      ctx.strokeStyle = strokeColor
      ctx.lineWidth = 1
      ctx.strokeRect(plotLeft + plotWidth - 195, y - 9, 190, 18)

      ctx.fillStyle = strokeColor
      ctx.font = 'bold 10px "JetBrains Mono", monospace'
      ctx.fillText(badgeLabel, plotLeft + plotWidth - 190, y + 4)
    })

    // 6. Current Mid-Price Guideline & Right Tag
    const currentY = priceToY(currentMid)
    ctx.strokeStyle = '#F0B90B'
    ctx.lineWidth = 1.0
    ctx.setLineDash([2, 2])
    ctx.beginPath()
    ctx.moveTo(plotLeft, currentY)
    ctx.lineTo(plotLeft + plotWidth, currentY)
    ctx.stroke()
    ctx.setLineDash([])

    // Right tag for mid price
    ctx.fillStyle = '#F0B90B'
    ctx.fillRect(plotLeft + plotWidth + 4, currentY - 9, 72, 18)
    ctx.fillStyle = '#0B0E14'
    ctx.font = 'bold 10px "JetBrains Mono", monospace'
    ctx.fillText(`$${currentMid.toFixed(2)}`, plotLeft + plotWidth + 8, currentY + 4)

    // 7. Bottom OFI & Volume Delta Subplot
    const bottomTop = mainHeight + 20
    ctx.fillStyle = '#0E131C'
    ctx.fillRect(plotLeft, bottomTop, plotWidth, bottomHeight)

    ctx.strokeStyle = '#181F2C'
    ctx.lineWidth = 1
    ctx.strokeRect(plotLeft, bottomTop, plotWidth, bottomHeight)

    // Zero center line
    const zeroY = bottomTop + bottomHeight / 2
    ctx.strokeStyle = '#283348'
    ctx.beginPath()
    ctx.moveTo(plotLeft, zeroY)
    ctx.lineTo(plotLeft + plotWidth, zeroY)
    ctx.stroke()

    // OFI bars
    const maxAbsOfi = Math.max(
      1.5,
      ...windowFrames.map((f) => Math.abs(f.ofi || 0))
    )

    windowFrames.forEach((frame, idx) => {
      const x = plotLeft + idx * colWidth + colWidth * 0.15
      const barW = colWidth * 0.7
      const ofiVal = frame.ofi || 0
      const barH = (Math.abs(ofiVal) / maxAbsOfi) * (bottomHeight / 2 - 4)

      ctx.fillStyle = ofiVal >= 0 ? '#089981' : '#F23645'
      if (ofiVal >= 0) {
        ctx.fillRect(x, zeroY - barH, barW, barH)
      } else {
        ctx.fillRect(x, zeroY, barW, barH)
      }
    })

    // Label for bottom subplot
    ctx.fillStyle = '#94A3B8'
    ctx.font = '9px "JetBrains Mono", monospace'
    ctx.fillText('ORDER FLOW IMBALANCE (OFI)', plotLeft + 6, bottomTop + 14)
    ctx.fillText('NOW', plotLeft + plotWidth - 28, bottomTop + bottomHeight - 6)
  }, [currentFrame, windowFrames, showFootprintHeatmap, windowSize])

  useEffect(() => {
    renderChart()
  }, [renderChart])

  // Mouse move handler for hover tooltip and crosshair
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const canvas = canvasRef.current
    if (!canvas || !currentFrame) return

    const rect = canvas.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    const mouseY = e.clientY - rect.top

    const rightMargin = 85
    const bottomHeight = 90
    const mainHeight = canvas.clientHeight - bottomHeight - 20
    const plotWidth = canvas.clientWidth - rightMargin - 15
    const plotLeft = 15

    if (mouseX >= plotLeft && mouseX <= plotLeft + plotWidth && mouseY >= 15 && mouseY <= mainHeight) {
      const currentMid = currentFrame.mid_price
      const priceSpanTicks = 24
      const tickSize = 0.5
      const maxPrice = currentMid + priceSpanTicks * tickSize
      const minPrice = currentMid - priceSpanTicks * tickSize
      const priceRange = maxPrice - minPrice

      const norm = (mouseY - 15) / (mainHeight - 15)
      const hoverPrice = maxPrice - norm * priceRange

      const colIdx = Math.floor(((mouseX - plotLeft) / plotWidth) * windowFrames.length)
      const targetFrame = windowFrames[Math.max(0, Math.min(windowFrames.length - 1, colIdx))]

      // Find depth at this rounded price
      const roundedP = Math.round(hoverPrice * 2) / 2
      const depthLvl = targetFrame?.depth_levels.find((l) => Math.abs(l.price - roundedP) < 0.26)

      const timeOffset = (windowFrames.length - 1 - colIdx) * 500
      const timeLabel = timeOffset === 0 ? 'Now' : `-${timeOffset}ms`

      setHoverInfo({
        x: mouseX,
        y: mouseY,
        price: roundedP,
        timeLabel,
        depthSize: depthLvl ? depthLvl.size : 0,
        side: depthLvl?.side,
      })
    } else {
      setHoverInfo(null)
    }
  }

  const handleMouseLeave = () => {
    setHoverInfo(null)
  }

  return (
    <div className="chart-container" ref={containerRef}>
      <div className="chart-header">
        <div className="chart-title-box">
          <span style={{ color: '#F0B90B' }}>●</span>
          <span>COMPOSITE ORDER-BOOK HEATMAP & FOOTPRINT CANDLESTICKS</span>
          <span className="badge-tag badge-dark" style={{ marginLeft: '8px' }}>
            {showFootprintHeatmap ? 'HEATMAP FOOTPRINT ACTIVE' : 'STANDARD CANDLES'}
          </span>
        </div>
        <div style={{ display: 'flex', gap: '16px', fontFamily: 'var(--font-mono)', fontSize: '11px', color: '#94A3B8' }}>
          <span>CYAN = RESTING BIDS</span>
          <span>CRIMSON = RESTING ASKS</span>
          <span>CANDLE INTERIOR = HEATMAP DENSITY</span>
        </div>
      </div>

      <div className="canvas-wrapper" onMouseMove={handleMouseMove} onMouseLeave={handleMouseLeave}>
        <canvas
          ref={canvasRef}
          style={{ width: '100%', height: '520px' }}
        />

        {hoverInfo && (
          <div
            className="chart-tooltip"
            style={{
              left: Math.min(window.innerWidth - 300, hoverInfo.x + 15),
              top: Math.max(10, hoverInfo.y - 45),
            }}
          >
            <div>
              <span style={{ color: '#64748B' }}>PRICE: </span>
              <span style={{ color: '#F0B90B', fontWeight: 600 }}>${hoverInfo.price.toFixed(2)}</span>
              <span style={{ color: '#64748B', marginLeft: '8px' }}>TIME: </span>
              <span>{hoverInfo.timeLabel}</span>
            </div>
            {hoverInfo.depthSize > 0 && (
              <div style={{ marginTop: '2px' }}>
                <span style={{ color: '#64748B' }}>RESTING DEPTH: </span>
                <span style={{ color: hoverInfo.side === 'bid' ? '#06B6D4' : '#F43F5E', fontWeight: 600 }}>
                  {hoverInfo.depthSize.toFixed(2)} BTC ({hoverInfo.side?.toUpperCase()})
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
