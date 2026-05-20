import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { exec } from 'child_process';

let panel: vscode.WebviewPanel | undefined;
let currentSummaryStage: string = 'frontend';

export function activate(context: vscode.ExtensionContext) {
  const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';

  context.subscriptions.push(
    vscode.commands.registerCommand('socmate.openWorkflowEditor', () => {
      if (panel) {
        panel.reveal();
        return;
      }

      panel = vscode.window.createWebviewPanel(
        'socmateWorkflow',
        'SoCMate',
        vscode.ViewColumn.One,
        {
          enableScripts: true,
          retainContextWhenHidden: true,
          localResourceRoots: [
            vscode.Uri.file(path.join(context.extensionPath, 'dist')),
          ],
        }
      );

      const webviewJs = panel.webview.asWebviewUri(
        vscode.Uri.file(path.join(context.extensionPath, 'dist', 'webview.js'))
      );
      const webviewCss = panel.webview.asWebviewUri(
        vscode.Uri.file(path.join(context.extensionPath, 'dist', 'webview.css'))
      );

      panel.webview.html = getWebviewContent(webviewJs, webviewCss);

      panel.webview.onDidReceiveMessage(
        (msg) => {
          if (msg.type === 'requestGraph') {
            fetchGraphStructure(workspaceRoot, msg.graphName || 'frontend');
          } else if (msg.type === 'requestTraces') {
            fetchNodeTraces(workspaceRoot, msg.nodeId);
          } else if (msg.type === 'requestSummary') {
            currentSummaryStage = msg.stage || 'frontend';
            fetchSummary(workspaceRoot, currentSummaryStage);
          }
        },
        undefined,
        context.subscriptions
      );

      panel.onDidDispose(() => {
        panel = undefined;
        stopExecutionPolling();
      });

      // Initial load
      fetchGraphStructure(workspaceRoot, 'frontend');
      startExecutionPolling(workspaceRoot);
    })
  );

  // Watch for graph file changes
  const watcher = vscode.workspace.createFileSystemWatcher(
    new vscode.RelativePattern(workspaceRoot, 'orchestrator/langgraph/*.py')
  );

  watcher.onDidChange(() => {
    if (panel) {
      fetchGraphStructure(workspaceRoot, 'frontend');
    }
  });

  context.subscriptions.push(watcher);
}

function fetchGraphStructure(root: string, graphName: string) {
  const script = `
import sys, json
sys.path.insert(0, '${root.replace(/\\/g, '\\\\')}')
from orchestrator.mcp_server import _introspect_graph, _project_root
print(json.dumps(_introspect_graph('${graphName}', _project_root())))
`.trim();

  exec(
    `python3 -c "${script.replace(/"/g, '\\"')}"`,
    { cwd: root, timeout: 10000 },
    (err, stdout, stderr) => {
      if (err) {
        vscode.window.showErrorMessage(`Graph introspection failed: ${stderr || err.message}`);
        return;
      }
      try {
        const data = JSON.parse(stdout);
        panel?.webview.postMessage({ type: 'graphUpdate', data, graphName });
      } catch (e) {
        vscode.window.showErrorMessage(`Failed to parse graph data: ${e}`);
      }
    }
  );
}

function getWebviewContent(webviewJs: vscode.Uri, webviewCss: vscode.Uri): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SoCMate</title>
  <link rel="stylesheet" href="${webviewCss}">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body, #root { width: 100%; height: 100%; overflow: hidden; }
    body { font-family: var(--vscode-font-family); background: var(--vscode-editor-background); color: var(--vscode-editor-foreground); }
  </style>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="${webviewJs}"></script>
</body>
</html>`;
}

export function deactivate() {
  stopExecutionPolling();
}

// ---------------------------------------------------------------------------
// Execution status polling
// ---------------------------------------------------------------------------

let pollingInterval: ReturnType<typeof setInterval> | undefined;

function startExecutionPolling(root: string) {
  stopExecutionPolling();
  pollingInterval = setInterval(() => {
    if (!panel) {
      stopExecutionPolling();
      return;
    }
    fetchExecutionStatus(root);
    fetchSummary(root, currentSummaryStage);
  }, 3000);
}

function stopExecutionPolling() {
  if (pollingInterval) {
    clearInterval(pollingInterval);
    pollingInterval = undefined;
  }
}

function fetchNodeTraces(root: string, nodeId: string) {
  const escapedRoot = root.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
  const escapedNode = nodeId.replace(/'/g, "\\'");
  const script = `
import sys, json
sys.path.insert(0, '${escapedRoot}')
from orchestrator.telemetry.reader import get_node_traces
import os
db_path = os.path.join('${escapedRoot}', '.socmate', 'traces.db')
print(json.dumps(get_node_traces(db_path, '${escapedNode}')))
`.trim();

  exec(
    `python3 -c "${script.replace(/"/g, '\\"')}"`,
    { cwd: root, timeout: 10000 },
    (err, stdout) => {
      if (err || !stdout.trim()) {
        panel?.webview.postMessage({ type: 'traceUpdate', nodeId, traces: [] });
        return;
      }
      try {
        const traces = JSON.parse(stdout);
        panel?.webview.postMessage({ type: 'traceUpdate', nodeId, traces });
      } catch {
        panel?.webview.postMessage({ type: 'traceUpdate', nodeId, traces: [] });
      }
    }
  );
}

function fetchExecutionStatus(root: string) {
  const script = `
import sys, json
sys.path.insert(0, '${root.replace(/\\/g, '\\\\')}')
from orchestrator.langgraph.event_stream import read_events
events = read_events('${root.replace(/\\/g, '\\\\')}')
status = {}
for e in events:
    node = e.get('node') or e.get('block')
    event_type = e.get('event', '')
    if node:
        if 'start' in event_type or 'enter' in event_type:
            status[node] = 'running'
        elif 'end' in event_type or 'exit' in event_type:
            if e.get('error'):
                status[node] = 'failed'
            else:
                status[node] = 'done'
print(json.dumps(status))
`.trim();

  exec(
    `python3 -c "${script.replace(/"/g, '\\"')}"`,
    { cwd: root, timeout: 5000 },
    (err, stdout) => {
      if (err || !stdout.trim()) return;
      try {
        const status = JSON.parse(stdout);
        panel?.webview.postMessage({ type: 'executionUpdate', status });
      } catch {
        // Silently ignore parse errors during polling
      }
    }
  );
}

// ---------------------------------------------------------------------------
// Summary file polling (reads .socmate/summary_{stage}.md directly from disk)
// ---------------------------------------------------------------------------

function fetchSummary(root: string, stage: string) {
  const summaryPath = path.join(root, '.socmate', `summary_${stage}.md`);
  try {
    if (fs.existsSync(summaryPath)) {
      const content = fs.readFileSync(summaryPath, 'utf-8');
      const stat = fs.statSync(summaryPath);
      const updated = stat.mtimeMs / 1000;
      panel?.webview.postMessage({
        type: 'summaryUpdate',
        stage,
        content,
        updated,
      });
    } else {
      panel?.webview.postMessage({
        type: 'summaryUpdate',
        stage,
        content: '',
        updated: null,
      });
    }
  } catch {
    // Silently ignore file read errors during polling
  }
}
