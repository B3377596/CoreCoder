# CoreCoder

> 一个可读、可懂、可 fork 的 AI 编程 Agent。从 Claude Code 逆向提取架构模式，用 Python 重建为 ~2,000 行代码库。

[English](README.md) | 中文 | [Claude Code 源码深度导读（7 篇）](article/)

[![PyPI](https://img.shields.io/pypi/v/corecoder)](https://pypi.org/project/corecoder/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 两种运行模式

CoreCoder 支持两种互补的运行模式：

### 普通模式（REPL）

交互式对话循环，跟 Claude Code 类似的体验。每轮对话自动注入仓库概览作为上下文。

```
$ corecoder -m deepseek-chat

You > 读一下 main.py，修掉拼错的 import

  > read_file(file_path='main.py')
  > edit_file(file_path='main.py', ...)

--- a/main.py
+++ b/main.py
@@ -1 +1 @@
-from utils import halper
+from utils import helper
```

### Plan 模式（DAG 编排）

输入一个高层目标，LLM Planner 自动拆解为带依赖关系的任务图，Scheduler 按拓扑顺序逐个执行。每个任务带有 **允许/禁止/停止条件** 的边界约束，防止 agent 越界。

```
You > /plan 给这个项目加一个前端页面调用计算器

Plan: 给这个项目加一个简单的前端页面  [0/5]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [  ] 安装 Flask Web 框架
    [  ] 创建 REST API 端点
      [  ] 构建前端页面
        [  ] 集成服务器启动到主入口
          [  ] 端到端验证
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[>>] 安装 Flask Web 框架
  > bash(command='cd test && uv add flask')
[OK] 安装 Flask Web 框架 [22.3s]
[>>] 创建 REST API 端点
  > write_file(file_path='test/api.py', ...)
[OK] 创建 REST API 端点 [15.1s]
...
```

---

## 架构

### 整体分层

```
用户输入
  │
  ├─ 普通模式 ──→ Agent.chat() ──→ ReAct 循环 ──→ LLM
  │
  └─ Plan 模式 ──→ Orchestrator
                    ├─ LLMPlanner        ← 目标 → 任务图
                    ├─ Scheduler         ← DAG 拓扑调度
                    │   └─ Executor      ← 调用 Agent.chat()
                    │       ├─ ContextOrchestrator  ← 动态上下文组装
                    │       └─ Verifier             ← 验证执行结果
                    └─ RecoveryManager   ← 重试/回滚
```

### State-Centric 运行时（核心设计）

**旧架构**（chat-history centric）：所有上下文（repo 摘要、约束、工作记忆）被永久追加到对话历史中——这是错误的。

**新架构**（state-centric）：上下文在每次 LLM 推理前从 `SessionState` **动态重建**为 ephemeral 消息前缀，不污染持久对话历史。

```
build_runtime_messages(state) =
  [system]              ← 稳定规则（来自 prompt.py）
  [assistant(memory)]   ← 已完成步骤 + 关键决策（ephemeral）
  [assistant(repo)]     ← 仓库结构认知（ephemeral）
  [assistant(runtime)]  ← 执行边界：ALLOWED/FORBIDDEN/STOP WHEN（ephemeral）
  [user]                ← 当前任务指令
  ... persistent_history ← 仅真实对话：user/assistant/tool 消息
```

**为什么这样做？**
- Ephemeral 前缀不会被压缩进对话摘要（避免把 repo 文件列表总结成无意义文本）
- 仓库上下文可以在发现变化时刷新，而不是永远看旧信息
- 持久历史只保留真实对话，undo/checkpoint 精确作用在对话边界上

### 项目结构

```
corecoder/
├── agent.py                    # ReAct 循环（State-Centric）
├── cli.py                      # CLI REPL + Plan 模式入口
├── config.py                   # 配置（环境变量 → 模型/API）
├── prompt.py                   # 系统提示词
│
├── runtime/                    # 运行时状态管理 ★新增
│   ├── state.py                # SessionState 数据类
│   └── assembler.py            # 动态消息组装器
│
├── llm/                        # LLM 接口层
│   ├── types.py                # ToolCall, LLMResponse, SSEEvent
│   └── client.py               # LLM（OpenAI 兼容）+ LiteLLM（多厂商）
│
├── repo/                       # 仓库智能
│   ├── index.py                # RepoIndex 符号/依赖索引
│   └── shadow.py               # ShadowGit 影子仓库（checkpoint/undo/diff）
│
├── history/                    # 对话历史管理
│   ├── compression.py          # ContextManager 三层压缩（snip→summarize→collapse）
│   └── session.py              # 会话存续（支持 v1/v2 格式）
│
├── tools/                      # 工具实现
│   ├── bash.py                 # Shell 命令（含危险命令拦截）
│   ├── read.py / write.py      # 文件读写
│   ├── edit.py                 # 搜索替换编辑（唯一匹配保证）
│   ├── glob_tool.py            # 文件匹配（自动过滤噪声目录）
│   ├── grep.py                 # 内容搜索
│   ├── repo_info.py            # 仓库结构化索引查询
│   └── agent.py                # 子 Agent 工具
│
└── orchestration/              # DAG 编排层
    ├── orchestrator.py         # 顶层 Orchestrator
    ├── storage.py              # 持久化
    ├── observability.py        # 结构化日志
    ├── viz.py                  # 图可视化
    │
    ├── dag/                    # 图结构与状态
    │   ├── models.py           # TaskNode, ExecutionResult
    │   ├── graph.py            # TaskGraph（DAG）
    │   ├── memory.py           # WorkingMemory + MemoryInjector
    │   └── recovery.py         # RecoveryManager 重试/回滚
    │
    ├── engine/                 # 执行引擎
    │   ├── scheduler.py        # 依赖感知调度器
    │   ├── planner.py          # Planner（Static + LLM）
    │   ├── executor.py         # Executor（包装 ReAct 循环）
    │   └── verifier.py         # 验证层（文件创建/内容/补丁分析）
    │
    ├── context/                # 上下文编排
    │   ├── orchestrator.py     # ContextOrchestrator 主入口
    │   ├── models.py           # ContextFragment, TokenBudget
    │   ├── layers.py           # 6 个上下文层（Task/WorkingMemory/Failure/Constraint/ExecutionPolicy/System）
    │   ├── pipeline.py         # 分阶段流水线（rank→dedup→compress→budget）
    │   ├── ranker.py           # 多信号相关性排序
    │   ├── retriever.py        # 图感知仓库检索
    │   └── policies.py         # 7 种执行状态策略
    │
    └── retrieval/              # 符号级仓库检索
        ├── models.py           # RankedFile, FileSummary
        ├── symbol_index.py     # SymbolOwnershipGraph 符号→文件双向索引
        ├── summaries.py        # FileSummaryManager（启发式 + LLM 可选）
        ├── task_intent.py      # TaskIntentAnalyzer 任务意图分类
        ├── query_planner.py    # RetrievalQueryPlanner 查询扩展
        ├── dependency_graph.py # BidirectionalDepGraph 依赖邻域扩展
        └── ranker.py           # StructuredRanker 多因素排序
```

### SessionState — 运行时认知中心

```python
@dataclass
class SessionState:
    persistent_history: list[dict]   # 仅真实对话（user/assistant/tool）
    repo_summary: str = ""           # 仓库结构摘要（session-long）
    active_files: list[str]          # 当前任务相关文件（task-long）
    completed_steps: list[str]       # 已完成步骤（compactable）
    important_decisions: list[str]   # 关键决策（compactable）
    constraints: list[str]           # 约束条件（task-long）
    failures: list[str]              # 失败记录（capped at 10）
    allowed_actions: list[str]       # 允许操作（execution-long）
    forbidden_actions: list[str]     # 禁止操作（execution-long）
    stop_conditions: str             # 停止条件（execution-long）
```

### 上下文组装管道

```
ContextRequest（from Scheduler）
  │
  ├─ ContextLayer 各层 produce()
  │   ├─ TaskContextLayer         → 当前任务描述
  │   ├─ WorkingMemoryContextLayer→ 已完成工作
  │   ├─ FailureMemoryContextLayer→ 历史失败
  │   ├─ ConstraintContextLayer   → 约束条件
  │   └─ ExecutionPolicyContextLayer → ALLOWED/FORBIDDEN/STOP WHEN
  │
  ├─ RepositoryContextRetriever.retrieve()
  │   ├─ TaskIntentAnalyzer       → 分类任务类型
  │   ├─ RetrievalQueryPlanner   → 扩展查询
  │   ├─ SymbolOwnershipGraph     → 符号→文件路由
  │   ├─ BidirectionalDepGraph    → 依赖邻域扩展
  │   ├─ FileSummaryManager       → 语义摘要匹配
  │   └─ StructuredRanker         → 多因素排序
  │
  ├─ Pipeline: rank → deduplicate → compress → budget
  │
  ├─ _assemble_user_message()     → Goal + Task（用户指令）
  ├─ _assemble_context_message()  → 所有非 TASK 内容（观察/日志用）
  └─ _extract_state_updates()     → SessionState 字段 dict
```

### 上下文压缩（三层）

压缩**仅作用于 persistent_history**（真实对话），ephemeral 前缀不在压缩范围内：

| 层 | 触发阈值 | 策略 |
|---|---|---|
| Layer 1 (snip) | 50% token 上限 | 截断 >1500 字符的 tool 输出 |
| Layer 2 (summarize) | 70% token 上限 | LLM 总结旧对话，保留后 8 条消息 |
| Layer 3 (collapse) | 90% token 上限 | 硬重置：保留后 4 条 + 摘要 |

---

## 安装与使用

### 安装

```bash
pip install corecoder
```

### 配置模型

支持任意 OpenAI 兼容 API。环境变量或项目 `.env` 文件均可：

```bash
# DeepSeek V3
export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com
corecoder -m deepseek-chat

# Kimi K2.5
export OPENAI_API_KEY=你的key OPENAI_BASE_URL=https://api.moonshot.ai/v1
corecoder -m kimi-k2.5

# Claude（通过 OpenRouter）
export OPENAI_API_KEY=你的key OPENAI_BASE_URL=https://openrouter.ai/api/v1
corecoder -m anthropic/claude-opus-4-6

# OpenAI GPT-5
export OPENAI_API_KEY=sk-...
corecoder -m gpt-5

# 本地 Ollama
export OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1
corecoder -m qwen3:32b

# 单次模式（不进入 REPL）
corecoder -p "给 parse_config() 加上错误处理"
```

### REPL 命令

| 命令 | 作用 |
|------|------|
| `/plan <目标>` | 进入 Plan 模式：LLM 拆解目标为任务图并执行 |
| `/model` | 查看当前模型 |
| `/model <名称>` | 切换模型 |
| `/compact` | 手动压缩对话历史 |
| `/tokens` | 查看 token 用量和费用估算 |
| `/diff` | 查看本次会话修改的文件 |
| `/undo` | 撤销上一次用户对话（恢复文件和对话状态） |
| `/save` | 保存会话到 `~/.corecoder/sessions/` |
| `/sessions` | 列出已保存的会话 |
| `/resume <id>` | 恢复已保存的会话 |
| `/reset` | 清空对话历史 |
| `quit` | 退出 |

### 会话恢复

```bash
# 列出已保存的会话
corecoder --list-sessions

# 恢复指定会话
corecoder --resume session_20260527_143000_a1b2c3d4
```

---

## 作为库使用

```python
from corecoder import Agent, LLM

llm = LLM(
    model="deepseek-chat",
    api_key="sk-...",
    base_url="https://api.deepseek.com",
)
agent = Agent(llm=llm)

# 普通对话
response = await agent.chat("找出项目里所有的 TODO 注释")

# 带结构化状态更新（编排模式）
response = await agent.chat(
    "实现 /calculate 端点",
    state_updates={
        "current_goal": "给项目加前端页面",
        "current_task": "创建 REST API 端点",
        "repo_summary": "## Project Files\n- calculator.py: 计算逻辑\n- api.py: Flask 路由",
        "allowed_actions": ["修改 api.py", "导入 calculator"],
        "forbidden_actions": ["修改 calculator.py 逻辑"],
        "stop_conditions": "api.py 包含 /calculate 路由",
    },
)

# 查看状态
print(agent.state.persistent_history)  # 真实对话
print(agent.state.completed_steps)     # 工作记忆
```

### 添加自定义工具

```python
from corecoder.tools.base import Tool

class HttpTool(Tool):
    name = "http"
    description = "发起 HTTP GET 请求并返回响应体。"
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    }

    def execute(self, url: str) -> str:
        import urllib.request
        return urllib.request.urlopen(url).read().decode()[:5000]

# 注入自定义工具
agent = Agent(llm=llm, tools=[HttpTool(), *agent.tools])
```

---

## 核心设计模式

| 设计模式 | 实现位置 | 说明 |
|---|---|---|
| **State-Centric 运行时** | `runtime/state.py`, `runtime/assembler.py` | SessionState 替代 chat-history accumulation；ephemeral 上下文每轮动态重建 |
| **搜索替换编辑** | `tools/edit.py` | old_string 在文件中唯一匹配才执行，防止误修改 |
| **并行工具执行** | `agent.py` — `asyncio.gather` | 单轮多个 tool_call 并发执行 |
| **三层上下文压缩** | `history/compression.py` | snip → summarize → collapse，仅压缩真实对话 |
| **子 Agent 隔离** | `tools/agent.py` | 子 Agent 有独立上下文窗口 |
| **危险命令拦截** | `tools/bash.py` | 拦截 `rm -rf`、fork bomb、curl pipe 等 |
| **ShadowGit 影子仓库** | `repo/shadow.py` | checkpoint/undo/diff 用真实 git snapshot，不触碰用户 .git |
| **符号级仓库检索** | `orchestration/retrieval/` | 无 embedding：符号→文件路由 + 依赖扩展 + 多因素排序 |
| **DAG 任务编排** | `orchestration/` | LLM Planner 拆解目标 → Scheduler 调度 → Executor 执行 → Verifier 验证 |

---

## FAQ

**CoreCoder 支持 Skill / MCP / Hook 吗？**

不支持，这是刻意的。CoreCoder 只保留可运行的最小核心。Skill、MCP、Hook 是 Claude Code 在上层加的特性；如果你想加，参考 [架构导读系列](article/)。

---

## License

MIT。Fork，改造，造你自己的东西。

---

作者 **[何宇峰](https://github.com/he-yufeng)** · Agentic AI Researcher @ Moonshot AI (Kimi)

[Claude Code 源码分析（知乎 17 万阅读，6000 收藏）](https://zhuanlan.zhihu.com/p/1898797658343862272)
