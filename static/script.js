document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const startBtn = document.getElementById('start-demo-btn');
    const loadingIndicator = document.getElementById('loading-indicator');
    const landingState = document.getElementById('landing-state');
    const dashboardState = document.getElementById('dashboard-state');
    
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    const graphIframe = document.getElementById('graph-iframe');
    const contractText = document.getElementById('raw-contract-text');
    
    const chatMessages = document.getElementById('chat-messages');
    const queryInput = document.getElementById('query-input');
    const sendBtn = document.getElementById('send-btn');
    const endChatBtn = document.getElementById('end-chat-btn');

    // Session ID for conversation memory (assigned after pipeline build)
    let sessionId = null;

    // 1. Start Demo / Build Pipeline
    startBtn.addEventListener('click', () => {
        startBtn.classList.add('hidden');
        loadingIndicator.classList.remove('hidden');
        const progressText = document.getElementById('progress-text');

        try {
            const eventSource = new EventSource('/api/build/stream');
            
            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.status === 'progress') {
                    if (progressText) {
                        progressText.textContent = data.message;
                    }
                } else if (data.status === 'complete') {
                    eventSource.close();
                    
                    // Populate Contract text
                    contractText.textContent = data.contract;
                    
                    // Set Iframe source to load the generated graph
                    graphIframe.src = '/api/graph.html';

                    // Create a chat session for conversation memory
                    fetch('/api/session', { method: 'POST' })
                        .then(r => r.json())
                        .then(d => { sessionId = d.session_id; })
                        .catch(() => { /* session will be auto-created server-side */ });

                    // Transition to dashboard
                    landingState.classList.add('hidden');
                    dashboardState.classList.remove('hidden');
                } else if (data.status === 'error') {
                    eventSource.close();
                    alert('Error building infrastructure: ' + data.message);
                    startBtn.classList.remove('hidden');
                    loadingIndicator.classList.add('hidden');
                }
            };

            eventSource.onerror = function(error) {
                console.error("EventSource failed:", error);
                // Sometimes the browser closes the connection cleanly
                if (eventSource.readyState === EventSource.CLOSED) {
                    return;
                }
                eventSource.close();
                alert('Connection to server lost during build.');
                startBtn.classList.remove('hidden');
                loadingIndicator.classList.add('hidden');
            };
            
        } catch (error) {
            alert('Error initializing build stream: ' + error.message);
            startBtn.classList.remove('hidden');
            loadingIndicator.classList.add('hidden');
        }
    });

    // 2. Tab Switching Logic
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active from all
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.add('hidden'));
            
            // Add active to clicked
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-tab');
            document.getElementById(targetId).classList.remove('hidden');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // 3. Chat Logic
    const addMessage = (content, sender = 'bot', contexts = null, triplets = null) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}`;
        
        let html = '';
        if (sender === 'bot') {
            const parsedContent = marked.parse(content);
            html += `<div class="avatar">🤖</div><div class="bubble">${parsedContent}`;
            
            // Add expandable context if available
            if (contexts || triplets) {
                let contextHtml = '';
                if (triplets && triplets.length > 0) {
                    contextHtml += `<div style="margin-bottom:8px"><strong>Extracted Triplets:</strong><br>`;
                    triplets.forEach(t => contextHtml += `<span style="color:var(--accent)">${t}</span><br>`);
                    contextHtml += `</div>`;
                }
                if (contexts && contexts.length > 0) {
                    contextHtml += `<div><strong>Semantic & Graph-Expanded Chunks:</strong><br>`;
                    contexts.forEach((c, i) => contextHtml += `<pre>[Excerpt ${i+1}] ${c.substring(0, 150)}...</pre>`);
                    contextHtml += `</div>`;
                }

                if (contextHtml) {
                    html += `
                        <div class="context-accordion">
                            <div class="accordion-header" onclick="this.nextElementSibling.classList.toggle('hidden')">
                                <span>🔍 View Retrieval Process</span>
                            </div>
                            <div class="accordion-content hidden">
                                ${contextHtml}
                            </div>
                        </div>
                    `;
                }
            }
            html += `</div>`;
        } else {
            html += `<div class="bubble"><p>${content}</p></div><div class="avatar">👤</div>`;
        }
        
        msgDiv.innerHTML = html;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };

    const handleSend = async () => {
        const query = queryInput.value.trim();
        if (!query) return;

        // Add user message
        addMessage(query, 'user');
        queryInput.value = '';
        
        // Disable input
        queryInput.disabled = true;
        sendBtn.disabled = true;
        
        // Show loading bubble
        const loadingId = 'loading-' + Date.now();
        const loadingHtml = `<div id="${loadingId}" class="message bot"><div class="avatar">🤖</div><div class="bubble"><p>...</p></div></div>`;
        chatMessages.insertAdjacentHTML('beforeend', loadingHtml);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, session_id: sessionId })
            });
            
            const data = await response.json();
            
            // Remove loading
            document.getElementById(loadingId).remove();
            
            if (response.ok) {
                addMessage(data.answer, 'bot', data.contexts, data.triplets);
            } else {
                addMessage('Error: ' + data.detail, 'bot');
            }
        } catch (error) {
            document.getElementById(loadingId).remove();
            addMessage('Failed to connect to server.', 'bot');
        }

        // Re-enable input
        queryInput.disabled = false;
        sendBtn.disabled = false;
        queryInput.focus();
    };

    sendBtn.addEventListener('click', handleSend);
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSend();
    });

    // End Chat -> resets the dashboard back to landing
    endChatBtn.addEventListener('click', () => {
        if(confirm("Are you sure you want to end the session?")) {
            // Hard refresh is easiest to clear state
            window.location.reload();
        }
    });
});
