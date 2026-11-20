import {
  AssistantRuntimeProvider,
  AuiIf,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useExternalStoreRuntime,
  type AppendMessage,
  type ThreadMessageLike,
} from "@assistant-ui/react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";

import "./style.css";

type Citation = {
  passage_id: string;
  quote?: string;
};

type ChatMessage = {
  message_id: string;
  role: "user" | "assistant";
  content: string;
  metadata: {
    citations?: Citation[];
    human_review_status?: "pending" | "approved" | "rejected";
  };
  created_at: string;
};

type ChatSession = {
  session_id: string;
  title: string;
  updated_at: string;
};

type ReviewRequest = {
  question_id: string;
  question: string;
  answer: string;
  citations: Citation[];
  grounding_f1: number | null;
  trigger_reason: "grounding_threshold" | "manual";
  created_at: string;
};

type ActivityEvent = {
  event_type: string;
  created_at: string;
  session_id?: string | null;
  question_id?: string | null;
  details: Record<string, unknown>;
};

type View = "chat" | "activity" | "review";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  const payload = response.status === 204 ? null : await response.json();
  if (!response.ok) {
    const detail =
      typeof payload?.detail === "string"
        ? payload.detail
        : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return payload as T;
}

function messageText(message: ChatMessage): string {
  const parts = [message.content];
  if (message.metadata.human_review_status === "pending") {
    parts.push("Answer withheld until human review is complete.");
  } else if (message.metadata.human_review_status === "approved") {
    parts.push("Approved by a human reviewer.");
  } else if (message.metadata.human_review_status === "rejected") {
    parts.push("Not approved by a human reviewer.");
  }
  if (message.metadata.citations?.length) {
    parts.push(
      `Sources\n${message.metadata.citations
        .map((citation) => `- ${citation.passage_id}: ${citation.quote || "Policy passage cited"}`)
        .join("\n")}`,
    );
  }
  return parts.filter(Boolean).join("\n\n");
}

function toThreadMessage(message: ChatMessage): ThreadMessageLike {
  return {
    id: message.message_id,
    role: message.role,
    content: [{ type: "text", text: messageText(message) }],
    createdAt: new Date(message.created_at),
  };
}

function ChatThread({
  messages,
  onNew,
  isRunning,
  isDisabled,
}: {
  messages: ChatMessage[];
  onNew: (message: AppendMessage) => Promise<void>;
  isRunning: boolean;
  isDisabled: boolean;
}) {
  const convertMessage = useCallback(toThreadMessage, []);
  const runtime = useExternalStoreRuntime({
    messages,
    convertMessage,
    onNew,
    isRunning,
    isDisabled,
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Root className="thread">
        <ThreadPrimitive.Viewport className="thread-viewport" autoScroll>
          <AuiIf condition={(state) => state.thread.isEmpty}>
            <section className="welcome">
              <span className="welcome-mark" aria-hidden="true">P</span>
              <h2>What would you like to know?</h2>
              <p>Ask about your policies. Answers are grounded in evidence and include citations.</p>
            </section>
          </AuiIf>
          <ThreadPrimitive.Messages>
            {({ message }) => (
              <MessagePrimitive.Root
                className={`chat-message ${message.role === "user" ? "user-message" : "assistant-message"}`}
              >
                <span className="message-avatar" aria-hidden="true">
                  {message.role === "user" ? "You" : "PE"}
                </span>
                <div className="message-body">
                  <span className="message-author">
                    {message.role === "user" ? "You" : "Policy Evidence Agent"}
                  </span>
                  <div className="message-content"><MessagePrimitive.Parts /></div>
                </div>
              </MessagePrimitive.Root>
            )}
          </ThreadPrimitive.Messages>
          <ThreadPrimitive.ViewportFooter className="thread-footer">
            <ComposerPrimitive.Root className="composer">
              <ComposerPrimitive.Input
                className="composer-input"
                placeholder="Ask a policy question..."
                aria-label="Your message"
                maxLength={4000}
                rows={1}
              />
              <ComposerPrimitive.Send className="send-button">Send</ComposerPrimitive.Send>
            </ComposerPrimitive.Root>
            <p className="composer-hint">Policy answers below 50% grounding F1 are automatically held for review.</p>
          </ThreadPrimitive.ViewportFooter>
        </ThreadPrimitive.Viewport>
      </ThreadPrimitive.Root>
    </AssistantRuntimeProvider>
  );
}

function App() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [reviewRequests, setReviewRequests] = useState<ReviewRequest[]>([]);
  const [selectedReviewId, setSelectedReviewId] = useState<string | null>(null);
  const [editedAnswer, setEditedAnswer] = useState("");
  const [activity, setActivity] = useState<ActivityEvent[]>([]);
  const [activeView, setActiveView] = useState<View>("chat");
  const [liveMode, setLiveMode] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [status, setStatus] = useState("");
  const [isError, setIsError] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">("light");

  const selectedReview = useMemo(
    () => reviewRequests.find((item) => item.question_id === selectedReviewId) ?? null,
    [reviewRequests, selectedReviewId],
  );

  useEffect(() => {
    const savedTheme = localStorage.getItem("policyEvidenceTheme");
    const initialTheme =
      savedTheme === "light" || savedTheme === "dark"
        ? savedTheme
        : window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light";
    setTheme(initialTheme);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("policyEvidenceTheme", theme);
  }, [theme]);

  useEffect(() => {
    let active = true;
    async function initialize() {
      try {
        const result = await request<{ sessions: ChatSession[] }>("/sessions");
        let nextSessions = result.sessions;
        if (nextSessions.length === 0) {
          const created = await request<{ session_id: string }>("/sessions", { method: "POST" });
          nextSessions = await request<{ sessions: ChatSession[] }>("/sessions").then(
            (value) => value.sessions,
          );
          localStorage.setItem("policyEvidenceSession", created.session_id);
        }
        if (!active) return;
        setSessions(nextSessions);
        const remembered = localStorage.getItem("policyEvidenceSession");
        const selected =
          nextSessions.find((item) => item.session_id === remembered) ?? nextSessions[0];
        if (selected) await selectSession(selected.session_id, active);
        await refreshReviewCount();
      } catch (error) {
        if (active) showError(error);
      } finally {
        if (active) setIsLoading(false);
      }
    }
    void initialize();
    return () => {
      active = false;
    };
  }, []);

  function showStatus(message: string, error = false) {
    setStatus(message);
    setIsError(error);
  }

  function showError(error: unknown) {
    showStatus(error instanceof Error ? error.message : String(error), true);
  }

  async function selectSession(sessionId: string, active = true) {
    const session = await request<{
      session_id: string;
      title: string;
      messages: ChatMessage[];
    }>(`/sessions/${encodeURIComponent(sessionId)}`);
    if (!active) return;
    setSelectedSessionId(session.session_id);
    setMessages(session.messages);
    localStorage.setItem("policyEvidenceSession", session.session_id);
    setActiveView("chat");
    showStatus("Saved");
  }

  async function refreshSessions() {
    const result = await request<{ sessions: ChatSession[] }>("/sessions");
    setSessions(result.sessions);
  }

  async function refreshReviewCount(): Promise<number> {
    const result = await request<{ requests: ReviewRequest[] }>("/review-requests");
    setReviewRequests(result.requests);
    setSelectedReviewId((current) =>
      result.requests.some((item) => item.question_id === current)
        ? current
        : result.requests[0]?.question_id ?? null,
    );
    return result.requests.length;
  }

  async function createSession() {
    try {
      const session = await request<{ session_id: string }>("/sessions", { method: "POST" });
      await refreshSessions();
      await selectSession(session.session_id);
      showStatus("New chat");
    } catch (error) {
      showError(error);
    }
  }

  async function deleteSession(sessionId: string) {
    const session = sessions.find((item) => item.session_id === sessionId);
    if (!session || !window.confirm(`Delete "${session.title}" and all its messages?`)) return;
    try {
      await request<void>(`/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
      const remaining = sessions.filter((item) => item.session_id !== sessionId);
      setSessions(remaining);
      if (sessionId === selectedSessionId) {
        localStorage.removeItem("policyEvidenceSession");
        if (remaining.length > 0) {
          await selectSession(remaining[0].session_id);
        } else {
          const created = await request<{ session_id: string }>("/sessions", { method: "POST" });
          await refreshSessions();
          await selectSession(created.session_id);
        }
      }
      showStatus("Chat deleted");
    } catch (error) {
      showError(error);
    }
  }

  const sendMessage = useCallback(async (message: AppendMessage) => {
    const content = message.content
      .filter((part) => part.type === "text")
      .map((part) => part.text)
      .join("")
      .trim();
    if (!content || !selectedSessionId) return;

    const sessionId = selectedSessionId;
    const pendingMessageId = crypto.randomUUID();
    const now = new Date().toISOString();
    const userMessage: ChatMessage = {
      message_id: crypto.randomUUID(),
      role: "user",
      content,
      metadata: {},
      created_at: now,
    };
    const pendingMessage: ChatMessage = {
      message_id: pendingMessageId,
      role: "assistant",
      content: "Searching policy evidence...",
      metadata: {},
      created_at: now,
    };
    setMessages((current) => [...current, userMessage, pendingMessage]);
    setIsRunning(true);
    showStatus("Searching policy evidence...");
    try {
      const result = await request<{ message: ChatMessage; human_review_status?: string }>(
        `/sessions/${encodeURIComponent(sessionId)}/messages`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            content,
            live_mode: liveMode,
          }),
        },
      );
      setMessages((current) =>
        current.map((item) => item.message_id === pendingMessageId ? result.message : item),
      );
      try {
        await refreshSessions();
        if (result.human_review_status === "pending") {
          await refreshReviewCount();
          showStatus("Answer withheld pending human approval");
        } else {
          showStatus("Saved");
        }
      } catch (error) {
        showError(error);
      }
    } catch (error) {
      const failure: ChatMessage = {
        ...pendingMessage,
        content: error instanceof Error ? error.message : String(error),
      };
      setMessages((current) =>
        current.map((item) => item.message_id === pendingMessageId ? failure : item),
      );
      showError(error);
    } finally {
      setIsRunning(false);
    }
  }, [liveMode, selectedSessionId]);

  async function openActivity() {
    setActiveView("activity");
    try {
      const result = await request<{ events: ActivityEvent[] }>("/activity?limit=100");
      setActivity(result.events);
      showStatus(`${result.events.length} recent events`);
    } catch (error) {
      showError(error);
    }
  }

  async function openReviews() {
    setActiveView("review");
    try {
      const count = await refreshReviewCount();
      showStatus(`${count} pending`);
    } catch (error) {
      showError(error);
    }
  }

  function chooseReview(item: ReviewRequest) {
    setSelectedReviewId(item.question_id);
    setEditedAnswer(item.answer);
  }

  async function submitReviewDecision(approved: boolean) {
    if (!selectedReview) return;
    if (!approved && !window.confirm("Reject this answer?")) return;
    const body: { approved: boolean; edited_answer?: string } = { approved };
    if (approved && editedAnswer.trim() !== selectedReview.answer) {
      body.edited_answer = editedAnswer.trim();
    }
    try {
      await request(
        `/review-requests/${encodeURIComponent(selectedReview.question_id)}/decision`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      );
      setSelectedReviewId(null);
      await refreshReviewCount();
      showStatus(approved ? "Answer approved" : "Answer rejected");
    } catch (error) {
      showError(error);
    }
  }

  const busy = isRunning || isLoading;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">P</div>
          <div>
            <strong>Policy Evidence</strong>
            <p>Grounded answers. Clear sources.</p>
          </div>
        </div>
        <button className="new-chat-button" onClick={() => void createSession()} disabled={busy}>
          <span aria-hidden="true">+</span> New chat
        </button>
        <nav className="primary-nav" aria-label="Workspace">
          <button className={activeView === "activity" ? "nav-item active" : "nav-item"} onClick={() => void openActivity()}>
            <span>Activity log</span>
          </button>
          <button className={activeView === "review" ? "nav-item active" : "nav-item"} onClick={() => void openReviews()}>
            <span>Human review</span><span className="count-badge">{reviewRequests.length}</span>
          </button>
        </nav>
        <div className="sidebar-heading">Your chats</div>
        <div className="session-list" aria-label="Saved chats">
          {sessions.map((session) => (
            <div className={`session-row ${session.session_id === selectedSessionId && activeView === "chat" ? "selected" : ""}`} key={session.session_id}>
              <button
                className="session-select"
                onClick={() => void selectSession(session.session_id).catch(showError)}
                disabled={busy}
              >
                <span className="session-title">{session.title}</span>
                <span className="session-date">{new Date(session.updated_at).toLocaleDateString()}</span>
              </button>
              <button
                className="delete-chat"
                aria-label={`Delete chat: ${session.title}`}
                title="Delete chat"
                onClick={() => void deleteSession(session.session_id)}
                disabled={busy}
              >
                ×
              </button>
            </div>
          ))}
          {sessions.length === 0 && <p className="sidebar-empty">No saved chats yet.</p>}
        </div>
        <div className="sidebar-footer">Policy answers include evidence and citations.</div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div className="page-heading">
            <h1>
              {activeView === "chat"
                ? sessions.find((item) => item.session_id === selectedSessionId)?.title ?? "New chat"
                : activeView === "activity" ? "Activity log" : "Human review"}
            </h1>
            <p>
              {activeView === "chat"
                ? "Ask follow-ups. Answers are grounded in policy evidence."
                : activeView === "activity"
                  ? "Recent chat, review, and application events."
                  : "Policy answers below 50% grounding F1 are held until approved."}
            </p>
          </div>
          <div className="topbar-tools">
            <label className="mode-toggle" title="Use LiteLLM for uncached messages. Usage may apply.">
              <input type="checkbox" checked={liveMode} onChange={(event) => setLiveMode(event.target.checked)} />
              <span className="toggle-track" aria-hidden="true"></span>
              <span>Live</span>
            </label>
            <label className="mode-toggle" title="Switch between light and dark appearance.">
              <input
                type="checkbox"
                checked={theme === "dark"}
                onChange={(event) => setTheme(event.target.checked ? "dark" : "light")}
              />
              <span className="toggle-track" aria-hidden="true"></span>
              <span>Dark</span>
            </label>
            <span className={isError ? "status error" : "status"} role="status">{status}</span>
          </div>
        </header>

        {activeView === "chat" && (
          <ChatThread
            messages={messages}
            onNew={sendMessage}
            isRunning={isRunning}
            isDisabled={!selectedSessionId || busy}
          />
        )}

        {activeView === "activity" && (
          <section className="activity-view" aria-label="Recent activity">
            {activity.length === 0 && <p className="empty-panel">No activity recorded yet.</p>}
            {activity.map((event, index) => (
              <article className="activity-item" key={`${event.event_type}-${event.created_at}-${index}`}>
                <div className="activity-marker" aria-hidden="true"></div>
                <div className="activity-content">
                  <strong>{event.event_type}</strong>
                  <time>{new Date(event.created_at).toLocaleString()}</time>
                  <p>
                    {[
                      event.session_id && `session=${event.session_id}`,
                      event.question_id && `reference=${event.question_id}`,
                      Object.keys(event.details ?? {}).length > 0 && JSON.stringify(event.details),
                    ].filter(Boolean).join(" · ") || "No additional details"}
                  </p>
                </div>
              </article>
            ))}
          </section>
        )}

        {activeView === "review" && (
          <section className="review-view" aria-label="Human review inbox">
            <div className="review-list">
              <div className="review-list-heading">
                <strong>Pending answers</strong>
                <span>{reviewRequests.length}</span>
              </div>
              {reviewRequests.length === 0 && <p className="empty-panel">No answers are waiting for approval.</p>}
              {reviewRequests.map((item) => (
                <button
                  className={`review-item ${item.question_id === selectedReviewId ? "selected" : ""}`}
                  key={item.question_id}
                  onClick={() => chooseReview(item)}
                >
                  <strong>{item.question || "Untitled question"}</strong>
                  <span>{new Date(item.created_at).toLocaleString()}</span>
                </button>
              ))}
            </div>
            <div className="review-detail">
              {selectedReview ? (
                <>
                  <div className="review-eyebrow">Answer review</div>
                  <h2>Question</h2>
                  <p className="review-question">{selectedReview.question}</p>
                  {selectedReview.trigger_reason === "grounding_threshold" ? (
                    <p className="review-explanation">
                      Grounding F1: {((selectedReview.grounding_f1 ?? 0) * 100).toFixed(0)}%.
                      The score is below 50%, so this answer is held until review.
                    </p>
                  ) : (
                    <p className="review-explanation">This turn was manually submitted for approval.</p>
                  )}
                  <label htmlFor="review-answer">Draft answer <span>— edit before approving</span></label>
                  <textarea
                    id="review-answer"
                    maxLength={4000}
                    value={editedAnswer}
                    onChange={(event) => setEditedAnswer(event.target.value)}
                  />
                  {selectedReview.citations.length > 0 && (
                    <div className="review-citations">
                      <strong>Sources</strong>
                      {selectedReview.citations.map((citation) => (
                        <blockquote key={citation.passage_id}>
                          <b>{citation.passage_id}</b>
                          {citation.quote && <span>{citation.quote}</span>}
                        </blockquote>
                      ))}
                    </div>
                  )}
                  <div className="review-actions">
                    <button className="approve-button" onClick={() => void submitReviewDecision(true)}>Approve answer</button>
                    <button className="reject-button" onClick={() => void submitReviewDecision(false)}>Reject</button>
                  </div>
                </>
              ) : (
                <div className="review-empty">
                  <span className="review-empty-mark" aria-hidden="true">✓</span>
                  <h2>Review queue</h2>
                  <p>Choose an answer from the list to inspect its evidence and make a decision.</p>
                </div>
              )}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
