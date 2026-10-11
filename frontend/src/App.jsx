// src/App.jsx
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { WebSocketProvider } from './context/WebSocketContext'
import { IncidentToastListener } from './components/NotificationToast'
import Sidebar from './components/Sidebar'
import Topbar from './components/Topbar'
import Dashboard from './pages/SecurityDashboard'
import OwnersPage from './pages/OwnersPage'
import IncidentHistory from './pages/IncidentHistory'
import AlertSettings from './pages/AlertSettings'
import LiveFeed from './pages/LiveFeed'
import AuthorizedUsers from './pages/AuthorizedUsers'

export default function App() {
  return (
    <BrowserRouter>
      <WebSocketProvider>
        <IncidentToastListener />
        <div className="app-shell">
          <Sidebar />
          <main className="main-content">
            <Topbar />
            <Routes>
              <Route path="/"                 element={<Dashboard />} />
              <Route path="/live"             element={<LiveFeed />} />
              <Route path="/authorized"       element={<AuthorizedUsers />} />
              <Route path="/authorized-users" element={<AuthorizedUsers />} />
              <Route path="/owners"           element={<OwnersPage />} />
              <Route path="/incidents"        element={<IncidentHistory />} />
              <Route path="/settings"         element={<AlertSettings />} />
            </Routes>
          </main>
        </div>
      </WebSocketProvider>
    </BrowserRouter>
  )
}
