import React from 'react'
import type { ReplayFrame } from '../types/replay'

interface MetricCardsProps {
  frame: ReplayFrame | null
  costHurdleBps: number
}

export const MetricCards: React.FC<MetricCardsProps> = ({ frame, costHurdleBps }) => {
  if (!frame) return null

  const { mid_price, best_bid, best_ask, spread, spread_bps, active_walls, ofi, depth_imbalance } = frame

  // Dominant Wall
  let wallTrust = 50
  let wallSize = 0
  let wallSide = 'NONE'
  let wallStatus = 'NO WALL'
  let wallColor = '#94A3B8'

  if (active_walls && active_walls.length > 0) {
    const topWall = active_walls.reduce((prev, curr) => (curr.size > prev.size ? curr : prev), active_walls[0])
    wallTrust = topWall.trust_score
    wallSize = topWall.size
    wallSide = topWall.side.toUpperCase()

    if (wallTrust >= 65) {
      wallStatus = 'AUTHENTIC'
      wallColor = '#10B981'
    } else if (wallTrust < 35) {
      wallStatus = 'FLEETING'
      wallColor = '#EF4444'
    } else {
      wallStatus = 'NEUTRAL'
      wallColor = '#F59E0B'
    }
  }

  const ofiColor = ofi >= 0 ? '#089981' : '#F23645'

  return (
    <div className="metric-grid">
      <div className="swiss-card">
        <div className="card-title">BTC/USDT Mid Price</div>
        <div className="card-value">${mid_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
        <div className="card-sub">
          Bid: ${best_bid.toFixed(2)} | Ask: ${best_ask.toFixed(2)}
        </div>
      </div>

      <div className="swiss-card">
        <div className="card-title">Spread Friction</div>
        <div className="card-value">
          ${spread.toFixed(2)}{' '}
          <span style={{ fontSize: '13px', color: '#94A3B8', fontWeight: 500 }}>
            ({spread_bps.toFixed(2)} bps)
          </span>
        </div>
        <div className="card-sub">
          Taker Hurdle: {costHurdleBps.toFixed(1)} bps
        </div>
      </div>

      <div className="swiss-card">
        <div className="card-title">Dominant Wall Trust</div>
        <div className="card-value" style={{ color: wallColor }}>
          {wallTrust.toFixed(0)}
          <span style={{ fontSize: '13px', color: '#64748B' }}>/100 </span>
          <span
            style={{
              fontSize: '11px',
              border: `1px solid ${wallColor}66`,
              padding: '2px 6px',
              borderRadius: '3px',
              marginLeft: '6px',
              color: wallColor,
            }}
          >
            {wallStatus}
          </span>
        </div>
        <div className="card-sub">
          {wallSide} Wall: {wallSize.toFixed(1)} BTC
        </div>
      </div>

      <div className="swiss-card">
        <div className="card-title">Order Flow Imbalance (OFI)</div>
        <div className="card-value" style={{ color: ofiColor }}>
          {ofi >= 0 ? `+${ofi.toFixed(2)}` : ofi.toFixed(2)}
        </div>
        <div className="card-sub">
          Depth Imbalance: {depth_imbalance >= 0 ? `+${depth_imbalance.toFixed(2)}` : depth_imbalance.toFixed(2)}
        </div>
      </div>
    </div>
  )
}
