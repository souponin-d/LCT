<script>
  import { onDestroy, onMount } from 'svelte';

  let processedData = null;
  let lastUpdated = null;
  let connectionStatus = 'connecting';
  let socket;
  let testClicks = 0;

  const reconnectDelay = 1500;

  const statusMessages = {
    connected: 'Соединение установлено',
    connecting: 'Устанавливаем соединение…',
    disconnected: 'Соединение отсутствует'
  };

  function connect() {
    if (socket) {
      socket.close();
      socket = undefined;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws/processed`);
    socket = ws;
    connectionStatus = 'connecting';

    ws.addEventListener('open', () => {
      if (socket === ws) {
        connectionStatus = 'connected';
      }
    });

    ws.addEventListener('message', (event) => {
      if (socket !== ws) return;
      try {
        processedData = JSON.parse(event.data);
        lastUpdated = new Date();
      } catch (err) {
        processedData = { error: 'Не удалось разобрать сообщение', raw: event.data };
      }
    });

    const scheduleReconnect = () => {
      if (socket === ws) {
        connectionStatus = 'disconnected';
        setTimeout(() => {
          if (socket === ws) {
            connect();
          }
        }, reconnectDelay);
      }
    };

    ws.addEventListener('close', scheduleReconnect);
    ws.addEventListener('error', () => {
      ws.close();
    });
  }

  function handleTestClick() {
    testClicks += 1;
  }

  onMount(() => {
    connect();
  });

  onDestroy(() => {
    if (socket) {
      socket.close();
      socket = undefined;
    }
  });
</script>

<style>
  :global(:root) {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    line-height: 1.5;
    font-weight: 400;
    color: #0f172a;
    background-color: #f8fafc;
  }

  :global(body) {
    margin: 0;
    min-height: 100vh;
  }

  button {
    cursor: pointer;
    border: none;
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    font-weight: 600;
    background-color: #2563eb;
    color: white;
    transition: background-color 0.2s ease;
  }

  button:hover {
    background-color: #1d4ed8;
  }

  .card {
    background: white;
    border-radius: 0.75rem;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.1);
    padding: 1.5rem;
  }

  .status {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.95rem;
    font-weight: 500;
    margin-bottom: 1rem;
  }

  .status-dot {
    width: 0.75rem;
    height: 0.75rem;
    border-radius: 9999px;
    background: #9ca3af;
  }

  .status-dot.connected {
    background: #16a34a;
  }

  .status-dot.connecting {
    background: #facc15;
  }

  .status-dot.disconnected {
    background: #dc2626;
  }

  pre {
    margin: 0;
    overflow-x: auto;
    background: #0f172a;
    color: #f8fafc;
    padding: 1rem;
    border-radius: 0.75rem;
  }

  main {
    max-width: 960px;
    margin: 0 auto;
    padding: 3rem 1.5rem;
  }

  h1 {
    margin-top: 0;
    font-size: 2rem;
    margin-bottom: 1rem;
  }

  .controls {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 1.5rem;
  }

  .small-text {
    font-size: 0.85rem;
    color: #475569;
  }
</style>

<main>
  <div class="card">
    <h1>Processed Data Monitor</h1>
    <div class="controls">
      <div class="status">
        <span class={`status-dot ${connectionStatus}`}></span>
        <span>{statusMessages[connectionStatus] || 'Неизвестный статус'}</span>
      </div>
      <button type="button" on:click={handleTestClick}>Test</button>
    </div>
    <p class="small-text">
      {#if lastUpdated}
        Последнее обновление: {lastUpdated.toLocaleTimeString()}
      {:else}
        Ожидание данных от бэкенда…
      {/if}
    </p>
    <p class="small-text">Количество нажатий на кнопку Test: {testClicks}</p>
    {#if processedData}
      <pre>{JSON.stringify(processedData, null, 2)}</pre>
    {:else}
      <pre>{'{\n  "status": "ожидаем processed_data.json"\n}'}</pre>
    {/if}
  </div>
</main>
