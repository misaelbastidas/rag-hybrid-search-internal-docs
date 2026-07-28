import { useEffect, useRef, useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function sourceLabel(source) {
  const location = source.section_heading || (source.page_num ? `page ${source.page_num}` : '')
  return location ? `${source.source_path} (${location})` : source.source_path
}

function confidenceLevel(score) {
  if (score >= 0.7) return 'high'
  if (score >= 0.4) return 'medium'
  return 'low'
}

function SourcesPanel({ sources }) {
  const [open, setOpen] = useState(false)
  if (!sources || sources.length === 0) return null

  return (
    <div className="sources">
      <button className="sources-toggle" onClick={() => setOpen(!open)}>
        {open ? 'Hide' : 'Show'} {sources.length} source{sources.length !== 1 ? 's' : ''}
      </button>
      {open && (
        <ol className="sources-list">
          {sources.map((s) => (
            <li key={s.id}>
              <div className="source-label">{sourceLabel(s)}</div>
              <div className="source-text">{s.text}</div>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

function ConfidenceBadges({ confidence }) {
  if (!confidence) return null
  const items = [
    ['confidence', confidence.composite],
    ['retrieval', confidence.retrieval_confidence],
    ['citations', confidence.citation_coverage],
    ['completeness', confidence.completeness],
  ]
  return (
    <div className="confidence-row">
      {items.map(([label, value]) => (
        <span key={label} className={`badge badge-${confidenceLevel(value)}`}>
          {label} {value.toFixed(2)}
        </span>
      ))}
    </div>
  )
}

function Message({ message }) {
  if (message.role === 'user') {
    return (
      <div className="message message-user">
        <div className="bubble bubble-user">{message.content}</div>
      </div>
    )
  }

  if (message.loading) {
    return (
      <div className="message message-assistant">
        <div className="bubble bubble-assistant bubble-loading">
          <span className="dot" />
          <span className="dot" />
          <span className="dot" />
        </div>
      </div>
    )
  }

  if (message.error) {
    return (
      <div className="message message-assistant">
        <div className="bubble bubble-assistant bubble-error">{message.error}</div>
      </div>
    )
  }

  if (!message.answered) {
    return (
      <div className="message message-assistant">
        <div className="bubble bubble-assistant bubble-declined">{message.declineMessage}</div>
        <SourcesPanel sources={message.sources} />
      </div>
    )
  }

  return (
    <div className="message message-assistant">
      <div className="bubble bubble-assistant">{message.content}</div>
      <ConfidenceBadges confidence={message.confidence} />
      <SourcesPanel sources={message.sources} />
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function sendMessage(e) {
    e.preventDefault()
    const question = input.trim()
    if (!question || sending) return

    const userMessage = { id: crypto.randomUUID(), role: 'user', content: question }
    const loadingMessage = { id: crypto.randomUUID(), role: 'assistant', loading: true }
    setMessages((prev) => [...prev, userMessage, loadingMessage])
    setInput('')
    setSending(true)

    try {
      const response = await fetch(`${API_URL}/v1/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
      if (!response.ok) throw new Error(`Request failed (${response.status})`)
      const data = await response.json()

      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMessage.id
            ? {
                id: m.id,
                role: 'assistant',
                answered: data.answered,
                content: data.answer,
                declineMessage: data.message,
                sources: data.sources,
                confidence: data.confidence,
              }
            : m
        )
      )
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMessage.id
            ? { id: m.id, role: 'assistant', error: `Could not reach the API: ${err.message}` }
            : m
        )
      )
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>RAG Pipeline · Hybrid Search</h1>
        <p>Ask questions about the indexed internal documents.</p>
      </header>

      <main className="chat">
        {messages.length === 0 && (
          <div className="empty-state">Ask a question to get started.</div>
        )}
        {messages.map((m) => (
          <Message key={m.id} message={m} />
        ))}
        <div ref={bottomRef} />
      </main>

      <form className="composer" onSubmit={sendMessage}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={sending}
        />
        <button type="submit" disabled={sending || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}
