// src/components/NotificationToast.jsx
// Renders floating incident toasts powered by react-hot-toast
import { useEffect } from 'react'
import toast, { Toaster } from 'react-hot-toast'
import { useWS } from '../context/WebSocketContext'
import { getIncidentMeta } from '../utils/incidentMeta'

export function IncidentToastListener() {
  const { lastEvent } = useWS()

  useEffect(() => {
    if (!lastEvent || lastEvent.label === 'Normal') return
    const meta = getIncidentMeta(lastEvent.label)

    const decision = Number(
      lastEvent.decision !== undefined && lastEvent.decision !== null
        ? lastEvent.decision
        : (Number(lastEvent.confidence || 0) * 100)
    )
    const modalityStr = String(lastEvent.modality_status || '').toUpperCase()
    const isVA = (
      modalityStr.includes('V+A') ||
      modalityStr.includes('MULTIMODAL') ||
      modalityStr.includes('VIDEO+AUDIO') ||
      (modalityStr.includes('VIDEO') && modalityStr.includes('AUDIO'))
    )
    const reqThreshold = isVA ? 70.0 : 80.0
    const isHighDecision = decision >= reqThreshold

    toast.custom(
      (t) => (
        <div
          onClick={() => toast.dismiss(t.id)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            background: '#111827',
            border: `1px solid ${meta.borderColor}`,
            borderRadius: 12,
            padding: '12px 16px',
            cursor: 'pointer',
            minWidth: 280,
            boxShadow: `0 4px 24px rgba(0,0,0,0.5), 0 0 12px ${meta.glowColor}`,
            opacity: t.visible ? 1 : 0,
            transition: 'opacity 0.3s ease',
          }}
        >
          <span style={{ fontSize: '1.6rem' }}>{meta.emoji}</span>
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontWeight: 700, color: meta.color, fontSize: '0.9rem' }}>
                🚨 {lastEvent.label.toUpperCase()}
              </span>
              {isHighDecision && (
                <span style={{
                  fontSize: '0.65rem', background: 'rgba(6,182,212,0.2)',
                  color: '#38bdf8', padding: '1px 5px', borderRadius: 4, fontWeight: 700
                }}>
                  🔔 NTFY
                </span>
              )}
            </div>
            <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: 2 }}>
              {decision.toFixed(1)}% decision · {lastEvent.owners_notified} notified
            </div>
            <div style={{ fontSize: '0.72rem', color: '#475569', marginTop: 1 }}>
              {lastEvent.modality_status}
            </div>
          </div>
        </div>
      ),
      { duration: 6000, position: 'top-right' }
    )
  }, [lastEvent])

  return (
    <Toaster
      position="top-right"
      toastOptions={{ style: {} }}
    />
  )
}
