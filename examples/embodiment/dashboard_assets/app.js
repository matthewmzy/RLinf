const state = {
  configsLoaded: false,
  lastStatus: null,
};

function qs(selector) {
  return document.querySelector(selector);
}

function formatValue(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  const num = Number(value);
  if (Math.abs(num) >= 1000) {
    return num.toLocaleString("zh-CN", { maximumFractionDigits: 0 });
  }
  if (Math.abs(num) >= 100) {
    return num.toFixed(1);
  }
  if (Math.abs(num) >= 10) {
    return num.toFixed(2);
  }
  return num.toFixed(digits);
}

function formatMaybeInt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toLocaleString("zh-CN", { maximumFractionDigits: 0 });
}

function setFeedback(message, isError = false) {
  const target = qs("#actionFeedback");
  target.textContent = message || "";
  target.style.color = isError ? "var(--danger)" : "var(--muted)";
}

async function apiPost(path, payload = {}) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || data.message || "请求失败");
  }
  return data;
}

function applyStatusBadge(status) {
  const badge = qs("#statusBadge");
  const normalized = (status || "idle").toLowerCase();
  badge.textContent = normalized;
  badge.className = `status-badge status-${normalized}`;
}

function updateMeta(session, summary) {
  qs("#metaConfig").textContent = session?.config_name || "-";
  qs("#metaPid").textContent = session?.pid || "-";
  qs("#metaPhase").textContent = summary?.phase || session?.phase || "-";
  const maxSteps = summary?.max_steps ? ` / ${summary.max_steps}` : "";
  qs("#metaStep").textContent =
    summary?.global_step !== undefined && summary?.global_step !== null
      ? `${summary.global_step}${maxSteps}`
      : "-";
  qs("#metaSession").textContent = session?.session_dir || "-";
  qs("#metaLog").textContent = session?.log_path || "-";
  qs("#metaBufferDir").textContent = session?.live_buffer_dir || "-";
}

function renderCards(summary, session) {
  const cards = [
    {
      label: "训练状态",
      value: (summary.status || "-").toUpperCase(),
      sub: `Phase: ${summary.phase || "-"}`,
    },
    {
      label: "Global Step",
      value:
        summary.global_step !== undefined && summary.global_step !== null
          ? `${summary.global_step}${summary.max_steps ? ` / ${summary.max_steps}` : ""}`
          : "-",
      sub: session?.stop_requested
        ? `停止请求已发送${session.save_buffers_on_stop ? "，将保留 buffer" : "，将丢弃 buffer"}`
        : "训练循环进度",
    },
    {
      label: "Replay Buffer",
      value: formatMaybeInt(summary.replay_buffer_size),
      sub: `样本 ${formatMaybeInt(summary.replay_buffer_samples)}`,
    },
    {
      label: "Demo Buffer",
      value: formatMaybeInt(summary.demo_buffer_size),
      sub: `样本 ${formatMaybeInt(summary.demo_buffer_samples)}`,
    },
    {
      label: "Actor Loss",
      value: formatValue(summary.actor_loss),
      sub: `Update ${summary.actor_update_step ?? "-"}`,
    },
    {
      label: "Critic Loss",
      value: formatValue(summary.critic_loss),
      sub: `Alpha ${formatValue(summary.alpha)}`,
    },
    {
      label: "Env Reward",
      value: formatValue(summary.env_reward),
      sub: `Episode Len ${formatValue(summary.env_episode_length, 1)}`,
    },
    {
      label: "Env Success",
      value:
        summary.env_success === null || summary.env_success === undefined
          ? "-"
          : `${formatValue(Number(summary.env_success) * 100, 1)}%`,
      sub:
        session?.cleanup_result === "kept_live_buffers"
          ? "buffer 已保留"
          : session?.cleanup_result === "discarded_live_buffers"
            ? "buffer 已丢弃"
            : session?.persist_buffers
              ? "实时持久化开启"
              : "仅缓存",
    },
  ];

  qs("#cardGrid").innerHTML = cards
    .map(
      (card) => `
        <article class="metric-card">
          <div class="metric-label">${card.label}</div>
          <div class="metric-value">${card.value}</div>
          <div class="metric-sub">${card.sub}</div>
        </article>
      `
    )
    .join("");
}

function buildPath(points, width, height, padding) {
  if (!points || points.length === 0) {
    return null;
  }
  const values = points.map((point) => Number(point.y));
  const xs = points.map((point) => Number(point.x));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...values);
  const maxY = Math.max(...values);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const chartWidth = width - padding * 2;
  const chartHeight = height - padding * 2;

  const coords = points.map((point) => {
    const x = padding + ((Number(point.x) - minX) / rangeX) * chartWidth;
    const y =
      height - padding - ((Number(point.y) - minY) / rangeY) * chartHeight;
    return [x, y];
  });

  const line = coords
    .map(([x, y], index) => `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`)
    .join(" ");
  const fill = `${line} L ${coords[coords.length - 1][0].toFixed(2)} ${(height - padding).toFixed(2)} L ${coords[0][0].toFixed(2)} ${(height - padding).toFixed(2)} Z`;
  return { line, fill, minY, maxY };
}

function renderSparkline(containerId, points, { alt = false, second = null } = {}) {
  const container = qs(containerId);
  if (!points || points.length === 0) {
    container.innerHTML = `<div class="empty-state">暂无数据</div>`;
    return;
  }
  const width = container.clientWidth || 320;
  const height = 180;
  const padding = 18;
  const firstPath = buildPath(points, width, height, padding);
  if (!firstPath) {
    container.innerHTML = `<div class="empty-state">暂无数据</div>`;
    return;
  }

  let secondSvg = "";
  if (second && second.length > 0) {
    const secondPath = buildPath(second, width, height, padding);
    if (secondPath) {
      secondSvg = `
        <path class="path-line alt" d="${secondPath.line}"></path>
      `;
    }
  }

  container.innerHTML = `
    <svg class="svg-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <line class="grid-line" x1="${padding}" y1="${padding}" x2="${width - padding}" y2="${padding}"></line>
      <line class="grid-line" x1="${padding}" y1="${height / 2}" x2="${width - padding}" y2="${height / 2}"></line>
      <line class="grid-line" x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}"></line>
      <path class="path-fill" d="${firstPath.fill}"></path>
      <path class="path-line ${alt ? "alt" : ""}" d="${firstPath.line}"></path>
      ${secondSvg}
      <text class="axis-label" x="${padding}" y="${padding - 4}">${formatValue(firstPath.maxY, 2)}</text>
      <text class="axis-label" x="${padding}" y="${height - 6}">${formatValue(firstPath.minY, 2)}</text>
    </svg>
  `;
}

function renderUpdateTable(updates) {
  const body = qs("#updateTableBody");
  if (!updates || updates.length === 0) {
    body.innerHTML = `
      <tr>
        <td colspan="7" style="color: var(--muted); padding: 16px 0;">暂无 update 数据</td>
      </tr>
    `;
    return;
  }

  body.innerHTML = updates
    .slice()
    .reverse()
    .map((entry) => {
      const metrics = entry.metrics || {};
      const replay = entry.replay_buffer || {};
      const demo = entry.demo_buffer || {};
      return `
        <tr>
          <td>${entry.update_step ?? "-"}</td>
          <td>${entry.global_step ?? "-"}</td>
          <td>${formatValue(metrics["sac/critic_loss"])}</td>
          <td>${formatValue(metrics["sac/actor_loss"])}</td>
          <td>${formatValue(metrics["sac/alpha"])}</td>
          <td>${formatMaybeInt(replay.size)}</td>
          <td>${formatMaybeInt(demo.size)}</td>
        </tr>
      `;
    })
    .join("");
}

function renderLogs(logs) {
  qs("#logTail").textContent = logs && logs.length ? logs.join("\n") : "暂无日志";
}

function setChartValues(summary) {
  qs("#actorLossValue").textContent = formatValue(summary.actor_loss);
  qs("#criticLossValue").textContent = formatValue(summary.critic_loss);
  qs("#alphaValue").textContent = formatValue(summary.alpha);
  qs("#replayValue").textContent = formatMaybeInt(summary.replay_buffer_size);
  qs("#demoValue").textContent = formatMaybeInt(summary.demo_buffer_size);
  const envReward = formatValue(summary.env_reward);
  const envSuccess =
    summary.env_success === null || summary.env_success === undefined
      ? "-"
      : `${formatValue(Number(summary.env_success) * 100, 1)}%`;
  qs("#envValue").textContent = `${envReward} / ${envSuccess}`;
}

function updateButtons(session, summary) {
  const status = (summary.status || "idle").toLowerCase();
  const runningStates = ["running", "launching", "stopping"];
  const isBusy = runningStates.includes(status);

  qs("#startBtn").disabled = isBusy;
  qs("#stopSaveBtn").disabled = !isBusy || status === "stopping";
  qs("#stopDiscardBtn").disabled = !isBusy || status === "stopping";
  qs("#forceStopBtn").disabled = !isBusy;
}

function populateConfigOptions(configs) {
  const select = qs("#configSelect");
  const currentValue = select.value;
  select.innerHTML = configs
    .map((config) => `<option value="${config}">${config}</option>`)
    .join("");
  if (currentValue && configs.includes(currentValue)) {
    select.value = currentValue;
  } else if (configs.includes("realworld_a2d_sac_psi_safe_explore")) {
    select.value = "realworld_a2d_sac_psi_safe_explore";
  }
}

async function fetchState() {
  try {
    const res = await fetch("/api/state");
    const data = await res.json();
    populateConfigOptions(data.available_configs || []);
    if (!state.configsLoaded) {
      state.configsLoaded = true;
    }

    const session = data.session || {};
    const summary = data.summary || {};
    applyStatusBadge(summary.status || session.status || "idle");
    updateMeta(session, summary);
    renderCards(summary, session);
    setChartValues(summary);
    updateButtons(session, summary);

    renderSparkline("#chart-actor-loss", data.charts?.actor_loss || []);
    renderSparkline("#chart-critic-loss", data.charts?.critic_loss || []);
    renderSparkline("#chart-alpha", data.charts?.alpha || []);
    renderSparkline("#chart-replay", data.charts?.replay_buffer_size || []);
    renderSparkline("#chart-demo", data.charts?.demo_buffer_size || []);
    renderSparkline("#chart-env", data.charts?.env_reward || [], {
      second: data.charts?.env_success || [],
    });

    renderUpdateTable(data.updates || []);
    renderLogs(data.logs || []);
  } catch (error) {
    setFeedback(`刷新状态失败：${error.message}`, true);
  }
}

async function startTraining() {
  try {
    setFeedback("正在启动训练...");
    const payload = {
      config_name: qs("#configSelect").value,
      experiment_name: qs("#experimentName").value.trim(),
      extra_overrides: qs("#extraOverrides").value.trim(),
      persist_buffers: qs("#persistBuffers").checked,
    };
    const result = await apiPost("/api/start", payload);
    setFeedback(result.message || "训练已启动。");
    await fetchState();
  } catch (error) {
    setFeedback(error.message, true);
  }
}

async function stopTraining(saveBuffers) {
  try {
    setFeedback(
      saveBuffers
        ? "已请求停止并保留全部 buffer..."
        : "已请求停止并在退出后丢弃 buffer..."
    );
    const result = await apiPost("/api/stop", { save_buffers: saveBuffers });
    setFeedback(result.message || "停止请求已发送。");
    await fetchState();
  } catch (error) {
    setFeedback(error.message, true);
  }
}

async function forceStopTraining() {
  try {
    setFeedback("正在发送强制终止信号...");
    const result = await apiPost("/api/force-stop", {});
    setFeedback(result.message || "已发送强制终止。");
    await fetchState();
  } catch (error) {
    setFeedback(error.message, true);
  }
}

function bindActions() {
  qs("#startBtn").addEventListener("click", startTraining);
  qs("#stopSaveBtn").addEventListener("click", () => stopTraining(true));
  qs("#stopDiscardBtn").addEventListener("click", () => stopTraining(false));
  qs("#forceStopBtn").addEventListener("click", forceStopTraining);

  window.addEventListener("keydown", (event) => {
    const active = document.activeElement;
    const typingTarget =
      active &&
      (active.tagName === "INPUT" ||
        active.tagName === "TEXTAREA" ||
        active.tagName === "SELECT");
    if (typingTarget) {
      return;
    }
    if (!event.shiftKey) {
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      startTraining();
    } else if (event.key.toLowerCase() === "s") {
      event.preventDefault();
      stopTraining(true);
    } else if (event.key.toLowerCase() === "d") {
      event.preventDefault();
      stopTraining(false);
    } else if (event.key.toLowerCase() === "x") {
      event.preventDefault();
      forceStopTraining();
    }
  });
}

bindActions();
fetchState();
setInterval(fetchState, 1000);
