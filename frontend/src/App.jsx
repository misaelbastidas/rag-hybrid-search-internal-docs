import { useEffect, useRef, useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const EXAMPLE_QUESTIONS = [
  '¿Cuántos días de vacaciones tengo con 1 año de antigüedad?',
  '¿Hay un programa de lealtad para clientes?',
  '¿Cuánto tiempo toma resolver una queja de un cliente?',
]

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

function Avatar() {
  return (
    <div className="avatar" aria-hidden="true">
      N58
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
        <Avatar />
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
        <Avatar />
        <div className="bubble bubble-assistant bubble-error">{message.error}</div>
      </div>
    )
  }

  if (!message.answered) {
    return (
      <div className="message message-assistant">
        <Avatar />
        <div className="message-body">
          <div className="bubble bubble-assistant bubble-declined">{message.declineMessage}</div>
          <SourcesPanel sources={message.sources} />
        </div>
      </div>
    )
  }

  return (
    <div className="message message-assistant">
      <Avatar />
      <div className="message-body">
        <div className="bubble bubble-assistant">{message.content}</div>
        <ConfidenceBadges confidence={message.confidence} />
        <SourcesPanel sources={message.sources} />
      </div>
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

  async function sendQuestion(question) {
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

  function handleSubmit(e) {
    e.preventDefault()
    sendQuestion(input.trim())
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            N58
          </div>
          <div>
            <h1>Naguara58 Assistant</h1>
            <p>Ask about company policies — HR, finance, operations, and more.</p>
          </div>
        </div>
      </header>

      <main className="chat">
        {messages.length === 0 && (
          <div className="empty-state">
            <p className="empty-state-title">How can I help?</p>
            <p className="empty-state-hint">Ask in English or Spanish — try one of these:</p>
            <div className="example-chips">
              {EXAMPLE_QUESTIONS.map((q) => (
                <button key={q} className="chip" onClick={() => sendQuestion(q)} disabled={sending}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m) => (
          <Message key={m.id} message={m} />
        ))}
        <div ref={bottomRef} />
      </main>

      <form className="composer" onSubmit={handleSubmit}>
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
