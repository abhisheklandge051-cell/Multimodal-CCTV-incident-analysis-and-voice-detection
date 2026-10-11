// src/utils/incidentMeta.js
// Maps incident class names to colors, emojis, and severity

export const INCIDENT_CLASSES = [
  'Abuse', 'CarAccident', 'Explosion', 'Fighting', 'Normal', 'Riot', 'Shooting'
]

const META = {
  Abuse:       { color: '#ef4444', borderColor: '#7f1d1d', glowColor: 'rgba(239,68,68,0.3)',   emoji: '👊', severity: 'critical', badgeClass: 'badge-abuse'       },
  CarAccident: { color: '#f97316', borderColor: '#7c2d12', glowColor: 'rgba(249,115,22,0.3)',  emoji: '🚗', severity: 'high',     badgeClass: 'badge-caraccident' },
  Explosion:   { color: '#eab308', borderColor: '#713f12', glowColor: 'rgba(234,179,8,0.3)',   emoji: '💥', severity: 'critical', badgeClass: 'badge-explosion'   },
  Fighting:    { color: '#f43f5e', borderColor: '#881337', glowColor: 'rgba(244,63,94,0.3)',   emoji: '⚔️', severity: 'high',     badgeClass: 'badge-fighting'    },
  Normal:      { color: '#22c55e', borderColor: '#14532d', glowColor: 'rgba(34,197,94,0.2)',   emoji: '✅', severity: 'low',      badgeClass: 'badge-normal'      },
  Riot:        { color: '#a855f7', borderColor: '#4c1d95', glowColor: 'rgba(168,85,247,0.3)',  emoji: '🔥', severity: 'critical', badgeClass: 'badge-riot'        },
  Shooting:    { color: '#dc2626', borderColor: '#7f1d1d', glowColor: 'rgba(220,38,38,0.3)',   emoji: '🔫', severity: 'critical', badgeClass: 'badge-shooting'    },
}

export function getIncidentMeta(label) {
  return META[label] || { color: '#94a3b8', borderColor: '#334155', glowColor: 'rgba(148,163,184,0.2)', emoji: '⚠️', severity: 'medium', badgeClass: '' }
}

export function getSeverityClass(label) {
  const meta = getIncidentMeta(label)
  return `severity-${meta.severity}`
}

export function parseDate(ts) {
  if (!ts) return null
  if (ts instanceof Date) return isNaN(ts.getTime()) ? null : ts

  let str = String(ts).trim()
  if (!str) return null

  // Handle pure numeric timestamps (seconds or milliseconds)
  if (/^\d+(\.\d+)?$/.test(str)) {
    const num = Number(str)
    return new Date(num < 1e11 ? num * 1000 : num)
  }

  // Replace space with T (e.g. '2026-09-21 09:51:29' -> '2026-09-21T09:51:29')
  str = str.replace(' ', 'T')

  // If missing timezone specifier, append 'Z' because the backend generates UTC datetimes
  if (!/[Zz]$/.test(str) && !/[+-]\d{2}(:?\d{2})?$/.test(str)) {
    str += 'Z'
  }

  const d = new Date(str)
  return isNaN(d.getTime()) ? null : d
}

export function formatTime(ts) {
  const d = parseDate(ts)
  if (!d) return '—'
  return d.toLocaleTimeString('en-IN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  })
}

export function formatDateTime(ts) {
  const d = parseDate(ts)
  if (!d) return '—'
  return d.toLocaleString('en-IN', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  })
}

export function formatRelativeTime(ts) {
  const d = parseDate(ts)
  if (!d) return '—'
  const diffSec = Math.floor((Date.now() - d.getTime()) / 1000)
  if (diffSec < 5) return 'just now'
  if (diffSec < 60) return `${diffSec}s ago`
  const diffMin = Math.floor(diffSec / 60)
  if (diffMin < 60) return `${diffMin}m ago`
  const diffHours = Math.floor(diffMin / 60)
  if (diffHours < 24) return `${diffHours}h ago`
  const diffDays = Math.floor(diffHours / 24)
  return `${diffDays}d ago`
}

