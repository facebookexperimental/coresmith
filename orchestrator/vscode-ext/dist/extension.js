"use strict";
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/extension.ts
var extension_exports = {};
__export(extension_exports, {
  activate: () => activate,
  deactivate: () => deactivate
});
module.exports = __toCommonJS(extension_exports);
var vscode = __toESM(require("vscode"));
var path = __toESM(require("path"));
var fs = __toESM(require("fs"));
var import_child_process = require("child_process");
var panel;
var currentSummaryStage = "frontend";
function activate(context) {
  const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || "";
  context.subscriptions.push(
    vscode.commands.registerCommand("socmate.openWorkflowEditor", () => {
      if (panel) {
        panel.reveal();
        return;
      }
      panel = vscode.window.createWebviewPanel(
        "socmateWorkflow",
        "SoCMate",
        vscode.ViewColumn.One,
        {
          enableScripts: true,
          retainContextWhenHidden: true,
          localResourceRoots: [
            vscode.Uri.file(path.join(context.extensionPath, "dist"))
          ]
        }
      );
      const webviewJs = panel.webview.asWebviewUri(
        vscode.Uri.file(path.join(context.extensionPath, "dist", "webview.js"))
      );
      const webviewCss = panel.webview.asWebviewUri(
        vscode.Uri.file(path.join(context.extensionPath, "dist", "webview.css"))
      );
      panel.webview.html = getWebviewContent(webviewJs, webviewCss);
      panel.webview.onDidReceiveMessage(
        (msg) => {
          if (msg.type === "requestGraph") {
            fetchGraphStructure(workspaceRoot, msg.graphName || "frontend");
          } else if (msg.type === "requestTraces") {
            fetchNodeTraces(workspaceRoot, msg.nodeId);
          } else if (msg.type === "requestSummary") {
            currentSummaryStage = msg.stage || "frontend";
            fetchSummary(workspaceRoot, currentSummaryStage);
          }
        },
        void 0,
        context.subscriptions
      );
      panel.onDidDispose(() => {
        panel = void 0;
        stopExecutionPolling();
      });
      fetchGraphStructure(workspaceRoot, "frontend");
      startExecutionPolling(workspaceRoot);
    })
  );
  const watcher = vscode.workspace.createFileSystemWatcher(
    new vscode.RelativePattern(workspaceRoot, "orchestrator/langgraph/*.py")
  );
  watcher.onDidChange(() => {
    if (panel) {
      fetchGraphStructure(workspaceRoot, "frontend");
    }
  });
  context.subscriptions.push(watcher);
}
function fetchGraphStructure(root, graphName) {
  const script = `
import sys, json
sys.path.insert(0, '${root.replace(/\\/g, "\\\\")}')
from orchestrator.mcp_server import _introspect_graph, _project_root
print(json.dumps(_introspect_graph('${graphName}', _project_root())))
`.trim();
  (0, import_child_process.exec)(
    `python3 -c "${script.replace(/"/g, '\\"')}"`,
    { cwd: root, timeout: 1e4 },
    (err, stdout, stderr) => {
      if (err) {
        vscode.window.showErrorMessage(`Graph introspection failed: ${stderr || err.message}`);
        return;
      }
      try {
        const data = JSON.parse(stdout);
        panel?.webview.postMessage({ type: "graphUpdate", data, graphName });
      } catch (e) {
        vscode.window.showErrorMessage(`Failed to parse graph data: ${e}`);
      }
    }
  );
}
function getWebviewContent(webviewJs, webviewCss) {
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
function deactivate() {
  stopExecutionPolling();
}
var pollingInterval;
function startExecutionPolling(root) {
  stopExecutionPolling();
  pollingInterval = setInterval(() => {
    if (!panel) {
      stopExecutionPolling();
      return;
    }
    fetchExecutionStatus(root);
    fetchSummary(root, currentSummaryStage);
  }, 3e3);
}
function stopExecutionPolling() {
  if (pollingInterval) {
    clearInterval(pollingInterval);
    pollingInterval = void 0;
  }
}
function fetchNodeTraces(root, nodeId) {
  const escapedRoot = root.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
  const escapedNode = nodeId.replace(/'/g, "\\'");
  const script = `
import sys, json
sys.path.insert(0, '${escapedRoot}')
from orchestrator.telemetry.reader import get_node_traces
import os
db_path = os.path.join('${escapedRoot}', '.socmate', 'traces.db')
print(json.dumps(get_node_traces(db_path, '${escapedNode}')))
`.trim();
  (0, import_child_process.exec)(
    `python3 -c "${script.replace(/"/g, '\\"')}"`,
    { cwd: root, timeout: 1e4 },
    (err, stdout) => {
      if (err || !stdout.trim()) {
        panel?.webview.postMessage({ type: "traceUpdate", nodeId, traces: [] });
        return;
      }
      try {
        const traces = JSON.parse(stdout);
        panel?.webview.postMessage({ type: "traceUpdate", nodeId, traces });
      } catch {
        panel?.webview.postMessage({ type: "traceUpdate", nodeId, traces: [] });
      }
    }
  );
}
function fetchExecutionStatus(root) {
  const script = `
import sys, json
sys.path.insert(0, '${root.replace(/\\/g, "\\\\")}')
from orchestrator.langgraph.event_stream import read_events
events = read_events('${root.replace(/\\/g, "\\\\")}')
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
  (0, import_child_process.exec)(
    `python3 -c "${script.replace(/"/g, '\\"')}"`,
    { cwd: root, timeout: 5e3 },
    (err, stdout) => {
      if (err || !stdout.trim())
        return;
      try {
        const status = JSON.parse(stdout);
        panel?.webview.postMessage({ type: "executionUpdate", status });
      } catch {
      }
    }
  );
}
function fetchSummary(root, stage) {
  const summaryPath = path.join(root, ".socmate", `summary_${stage}.md`);
  try {
    if (fs.existsSync(summaryPath)) {
      const content = fs.readFileSync(summaryPath, "utf-8");
      const stat = fs.statSync(summaryPath);
      const updated = stat.mtimeMs / 1e3;
      panel?.webview.postMessage({
        type: "summaryUpdate",
        stage,
        content,
        updated
      });
    } else {
      panel?.webview.postMessage({
        type: "summaryUpdate",
        stage,
        content: "",
        updated: null
      });
    }
  } catch {
  }
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  activate,
  deactivate
});
