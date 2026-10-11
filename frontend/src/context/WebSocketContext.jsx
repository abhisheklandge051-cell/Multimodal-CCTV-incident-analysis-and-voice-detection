// src/context/WebSocketContext.jsx
import { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react'

// Derive WS URL from current page host so it works in any deployment
function getWsUrl() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  // In dev (Vite proxy) the backend is on :8000; in prod it's the same host
  const host = import.meta.env.VITE_API_HOST || 'localhost:8000'
  return `${proto}//${host}/ws/incidents`
}

const WebSocketContext = createContext(null)

export function WebSocketProvider({ children }) {
  const [wsStatus, setWsStatus] = useState('connecting') // 'connecting' | 'connected' | 'disconnected'
  const [incidents, setIncidents] = useState([])
  const [lastEvent, setLastEvent] = useState(null)
  const wsRef = useRef(null)
  const reconnectTimer = useRef(null)

  // Derived boolean kept for backward-compat with existing consumers
  const connected = wsStatus === 'connected'

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setWsStatus('connecting')
    const ws = new WebSocket(getWsUrl())
    wsRef.current = ws

    ws.onopen = () => {
      setWsStatus('connected')
      clearTimeout(reconnectTimer.current)
    }

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'incident') {
          setLastEvent(msg)
          setIncidents(prev => [msg, ...prev].slice(0, 200))
        }
      } catch (_) {}
    }

    ws.onclose = () => {
      setWsStatus('disconnected')
      reconnectTimer.current = setTimeout(connect, 3000)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  const ping = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send('ping')
    }
  }

  return (
    <WebSocketContext.Provider value={{ connected, wsStatus, incidents, lastEvent, ping }}>
      {children}
    </WebSocketContext.Provider>
  )
}

export const useWS = () => useContext(WebSocketContext)
