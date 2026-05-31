# CoreCoder

> 原名 **NanoCoder**——为避免与 [Nano-Collective/nanocoder](https://github.com/Nano-Collective/nanocoder) 混淆而更名。

[![PyPI](https://img.shields.io/pypi/v/corecoder)](https://pypi.org/project/corecoder/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Claude Code 核心架构的极简 Python 复刻——约 2000 行源码，完整可运行。**

CoreCoder 不是另一个 AI 编程工具。它是一个**教学蓝图**，类比于 [nanoGPT](https://github.com/karpathy/nanoGPT)，是 AI coding agent 领域的最小可读实现。阅读它，魔改它，构建你自己的 agent。

```
$ corecoder -m kimi-k2.5

You > 读一下 main.py，修好那个坏掉的 import

  > read_file(file_path='main.py')
  > edit_file(file_path='main.py', ...)

--- a/main.py
+++ b/main.py
@@ -1 +1 @@
-from utils import halper
+from utils import helper

修好了：halper → helper。
```

---

## 架构概览

CoreCoder 精炼了 Claude Code 的 **7 个关键架构模式**，全部在 ~2000 行 Python 中完整实现：

```
入口: cli.py（CLI 参数解析 + REPL / 单次 / Plan 三种运行模式）
  └─ Agent（agent/core.py）── 核心 ReAct 循环 + Think-Execute 外层
       ├─ LLM（llm/client.py）            ── OpenAI / LiteLLM 双后端，SSE 流式
       ├─ Runtime（agent/runtime/）        ── State-Centric 状态管理 + 动态消息组装
       ├─ Tools（tools/）                  ── 8 个内置工具
       ├─ Context（context/）              ── 3 层上下文压缩 + 动态编排
       ├─ Shadow Git（codebase/shadow.py） ── 独立 git 仓库做 checkpoint / undo / diff
       ├─ RepoIndex（codebase/indexing/）  ── AST 符号提取 + 依赖分析
       ├─ Retrieval（retrieval/）          ── 零 Embedding 的符号化仓库检索
       ├─ Workflow（agent/workflow/）      ── DAG 任务编排（Plan 模式）
       └─ MCP（mcp/client.py）            ── MCP 协议支持（JSON-RPC over stdio）
```

| 模式 | Claude Code 源码规模 | CoreCoder 实现 |
|------|---------------------|----------------|
| Search-and-Replace 编辑（唯一匹配 + diff） | FileEditTool | `tools/edit.py`——70 行 |
| 并行工具执行 | StreamingToolExecutor | `agent/core.py`——asyncio.gather |
| 3 层上下文压缩 | HISTORY_SNIP → Microcompact → CONTEXT_COLLAPSE | `context/compression.py`——145 行 |
| 子智能体隔离上下文 | AgentTool（1397 行） | `tools/agent.py`——50 行 |
| 危险命令拦截 | BashTool（1143 行） | `tools/bash.py`——95 行 |
| 会话持久化 | QueryEngine（1295 行） | `context/session.py`——65 行 |
| 动态系统提示词构建 | prompts.ts（914 行） | `prompt.py`——35 行 |

---

## 安装

```bash
pip install corecoder
```

可选依赖：

```bash
pip install corecoder[litellm]    # 100+ 非 OpenAI 提供商支持
pip install corecoder[tiktoken]   # 精确 token 计数
pip install corecoder[dev]        # 测试框架
```

---

## 快速开始

任意兼容 OpenAI API 的模型均可使用。通过环境变量或项目根目录的 `.env` 文件配置：

```bash
# Kimi K2.5
export OPENAI_API_KEY=your-key OPENAI_BASE_URL=https://api.moonshot.ai/v1
corecoder -m kimi-k2.5

# Claude Opus 4.6（通过 OpenRouter）
export OPENAI_API_KEY=your-key OPENAI_BASE_URL=https://openrouter.ai/api/v1
corecoder -m anthropic/claude-opus-4-6

# GPT-5
export OPENAI_API_KEY=sk-...
corecoder -m gpt-5

# DeepSeek V3
export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com
corecoder -m deepseek-chat

# Qwen 3.5
export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
corecoder -m qwen-max

# Ollama（本地模型）
export OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 CORECODER_MODEL=qwen2.5-coder
corecoder
```

三种运行模式：

```bash
corecoder                          # 交互式 REPL（默认）
corecoder -p "修好 main.py 的 bug" # 单次执行
corecoder -P "实现用户登录功能"     # Plan 模式（LLM 制定计划 → DAG 执行）
corecoder -r <session-id>          # 恢复之前的会话
```

---

## 核心设计

### 1. State-Centric 运行时（而非 Chat-History-Centric）

这是 CoreCoder 最关键的架构决策。传统 coding agent 把所有信息（仓库结构、工作记忆、执行约束）直接塞进聊天历史，导致三个问题：压缩破坏上下文、undo 困难、LLM 混乱。

CoreCoder 的方案：**运行时认知走 SessionState 命名字段，瞬态上下文每次 turn 重新组装**。

`SessionState`（`agent/runtime/state.py`）按生命周期分四层：

| 层级 | 字段 | 生命周期 |
|------|------|---------|
| 持久对话 | `persistent_history` | 整个会话 |
| 仓库认知 | `repo_summary` | 整个会话，惰性刷新 |
| 任务上下文 | `active_files`, `active_symbols`, `current_task`, `current_goal` | 每个任务重置 |
| 工作记忆 | `completed_steps`, `important_decisions`, `failures`, `constraints` | 每个任务，可压缩 |
| 执行边界 | `allowed_actions`, `forbidden_actions`, `stop_conditions` | 每次执行重置 |

每轮 LLM 调用前，`assembler`（`agent/runtime/assembler.py`）从零构建消息列表：

```
[system]              ← 稳定的系统提示词
[assistant(memory)]   ← [WORKING MEMORY] 已完成步骤 + 关键决策
[assistant(repo)]     ← [REPOSITORY CONTEXT] 仓库结构 + 活跃文件
[assistant(runtime)]  ← [EXECUTION CONSTRAINTS] 允许/禁止/停止条件
... persistent_history（仅真实对话）
```

瞬态前缀**绝不写入 persistent_history**。这个分离保证了压缩只触真实对话，checkpoint/undo 只操作对话边界。

### 2. ReAct 循环（agent/core.py）

`Agent.chat()` 的核心流程：

1. 用户消息追加到 `persistent_history`
2. ShadowGit 做快照 checkpoint
3. 估算瞬态前缀 token 开销，必要时压缩
4. 最多 50 轮的 ReAct 循环：
   - 调用 LLM（SSE 流式）
   - 如果 LLM 返回纯文本 → 结束，返回给用户
   - 如果 LLM 返回工具调用 → 并行执行，结果追加到 persistent_history
   - 回到循环
5. 重建 RepoIndex（文件可能已被修改）

SSE 模式下，工具调用一旦 JSON 参数完整到达，立刻 `asyncio.create_task` 启动后台执行，与 LLM 剩余输出流重叠——减少总延迟。

### 3. Think-Execute 外层循环（agent/runtime/staged.py）

在基础 ReAct 循环之上，CoreCoder 实现了一个 **Think-Execute 双层结构**：

```
外层 (Think-Execute):  ThinkEngine → StagePlan → StageExecutor → Evaluation → 状态更新
内层 (ReAct):         现有 Agent.chat() 约束在单个 Stage 内
```

每个阶段有独立的**工具白名单**和**步骤预算**：

| 阶段 | 可用工具 | 最大步骤 | 用途 |
|------|---------|---------|------|
| `understand` | repo_info, glob, grep, read_file | 8 | 建立仓库高层理解 |
| `locate` | repo_info, glob, grep, read_file | 8 | 定位相关文件/符号 |
| `analyze` | read_file, grep, glob, repo_info | 8 | 理解实现细节 |
| `modify` | read_file, edit_file, write_file, grep, bash | 10 | 执行代码修改 |
| `verify` | read_file, grep, bash | 6 | 验证修改结果 |
| `recover` | repo_info, glob, grep, read_file, bash | 8 | 从失败中恢复/换方案 |
| `finalize` | 无工具 | 1 | 生成最终答案 |

`ThinkEngine` 在每个阶段开始前评估当前证据是否足够："我有没有理解仓库结构？有没有找到目标文件？有没有看到实现细节？"如果还缺，就规划下一个需要的阶段；如果都齐了，进入 finalize。

### 4. 工具系统

所有工具继承 `Tool` 抽象基类（`tools/base.py`），需要提供 `name`、`description`、`parameters`（JSON Schema）和 `execute()` 方法。

**8 个内置工具**（`tools/__init__.py`）：

| 工具 | 文件 | 核心功能 |
|------|------|---------|
| `bash` | `bash.py` 95 行 | Shell 执行 + 危险命令拦截（9 条正则）+ 超时 + cd 跟踪 |
| `read_file` | `read.py` | 按行号读取文件，支持 offset/limit |
| `write_file` | `write.py` | 创建或覆盖文件 |
| `edit_file` | `edit.py` 70 行 | **Search-and-Replace**：old_string 在文件中必须出现恰好一次 |
| `glob` | `glob_tool.py` | 文件名模式匹配，过滤噪音目录 |
| `grep` | `grep.py` | 正则内容搜索，200 匹配上限 |
| `repo_info` | `repo_info.py` | 对仓库索引的结构化查询（符号、导入、依赖） |
| `agent` | `agent.py` 50 行 | 创建独立的子智能体，有隔离的上下文窗口 |

**edit_file 的原理**（Claude Code 的核心创新）：不是让 LLM 输出整个文件，而是指定 `old_string`（要替换的原文）和 `new_string`（新内容）。工具验证 `old_string` 在文件中出现次数：

- `count == 0` → 报错："文件中找不到这段文字"
- `count > 1` → 报错："这段文字出现了 N 次，请包含更多上下文使其唯一"
- `count == 1` → 执行替换，返回 unified diff

### 5. 三层上下文压缩（context/compression.py）

只压缩 `persistent_history`，瞬态前缀通过 `estimate_ephemeral_tokens()` 计入阈值但本身不参与压缩。

| 层级 | 触发阈值（占 max_context_tokens） | 策略 |
|------|--------------------------------|------|
| Layer 1 "snip" | 50% | 超过 1500 字符的工具输出截断为首尾各 3 行 |
| Layer 2 "summarize" | 70% + 消息数 > 10 | LLM 摘要旧对话，保留最近 8 条完整 |
| Layer 3 "collapse" | 90% + 消息数 > 4 | 紧急模式：保留最后 4 条 + 硬摘要 |

Token 计数优先使用 `tiktoken`（`cl100k_base` 编码器），fallback 到 `len(text) // 3` 估算。

### 6. Shadow Git（codebase/shadow.py）

在 `~/.corecoder/shadow/<project-md5-hash>/` 维护一个**独立的 git 仓库**。核心技巧：

```python
env["GIT_DIR"] = self.shadow_dir       # git 对象存储在此
env["GIT_WORK_TREE"] = self.work_tree  # 操作的是真实项目文件
```

这样 checkpoint/undo/diff 完全不影响用户的 `.git`。每次用户 turn 前自动 `git add -A` + `git commit`，undo 时 `git reset --hard` 恢复文件 + 截断 `persistent_history`。

### 7. 仓库索引（codebase/indexing/index.py）

`RepoIndex` 在 `<project>/.corecoder/` 下构建项目结构知识：

- **符号提取**：AST 解析类、函数、方法及其签名
- **依赖分析**：解析 `pyproject.toml` / `requirements.txt` / `package.json` 的外部依赖 + 文件间 import 关系
- **框架检测**：自动识别 FastAPI、Flask、Django、React 等
- **增量更新**：基于文件 mtime 变化检测，有改动才重建

### 8. 零 Embedding 的符号化检索（retrieval/）

不依赖 embedding 模型，通过 **AST 符号索引** + **依赖图** + **启发式文件摘要** 做仓库理解：

- `SymbolOwnershipGraph`：符号 ↔ 文件 双向索引，支持模糊匹配
- `FileSummaryManager`：启发式文件用途分类（入口点/配置/测试等），可选 LLM 生成摘要
- `StructuredRanker`：多因素文件评分——编辑距离、所有权分数、依赖深度、摘要匹配、最近修改时间、任务对齐

---

## CLI 命令参考

进入 REPL 后（默认模式），支持以下命令：

| 命令 | 功能 |
|------|------|
| `/help` | 显示帮助 |
| `/plan <目标>` | 进入 Plan 模式：LLM 制定任务图 → DAG 编排执行 |
| `/reset` | 清空对话历史和 checkpoint |
| `/undo` | 撤销上一次用户 turn（ShadowGit reset + 截断历史） |
| `/model` | 显示当前模型 |
| `/model <名称>` | 运行时切换模型 |
| `/tokens` | 查看累计 token 使用量和估算费用 |
| `/compact` | 手动触发上下文压缩 |
| `/diff` | 查看本次会话修改的文件及 diff |
| `/save` | 保存当前会话到磁盘 |
| `/sessions` | 列出所有已保存会话 |
| `quit` / `exit` | 退出 |

输入技巧：
- `Enter` 提交消息
- `Esc + Enter` 插入换行（粘贴代码时使用）

---

## 配置

配置加载优先级（后者覆盖前者）：

1. `Config` dataclass 默认值（`config.py`）
2. `~/.corecoder/.env`（全局用户配置）
3. 从当前目录向上逐级查找 `.env`（项目级配置）
4. 环境变量（最高优先级）

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| `CORECODER_MODEL` | `gpt-4o` | 模型名称 |
| `CORECODER_API_KEY` | — | API Key（优先于 OPENAI_API_KEY） |
| `OPENAI_API_KEY` | — | OpenAI 兼容 API Key |
| `DEEPSEEK_API_KEY` | — | DeepSeek API Key（最低优先级） |
| `OPENAI_BASE_URL` | — | API 基础 URL |
| `CORECODER_BASE_URL` | — | API 基础 URL（备用） |
| `CORECODER_MAX_TOKENS` | `4096` | 单次 LLM 回复的最大 token 数 |
| `CORECODER_TEMPERATURE` | `0` | 生成温度 |
| `CORECODER_MAX_CONTEXT` | `128000` | 上下文窗口大小（决定压缩阈值） |
| `CORECODER_PROVIDER` | `openai` | 设为 `litellm` 使用 LiteLLM 后端 |
| `CORECODER_DEBUG` | `false` | 启用调试日志 |

---

## Plan 模式（DAG 任务编排）

通过 `/plan <目标>` 或 `-P` 参数启动。完整流水线：

```
用户输入 Goal
  → LLMPlanner：LLM 将目标分解为 TaskGraph（有向无环图，加边时在线环检测）
    → 展示任务图给用户
      → Orchestrator：协调整个流水线
        → Scheduler：拓扑排序，找到就绪任务（依赖全完成）
          → MemoryInjector：为每个任务构建工作记忆（目标/依赖/失败/约束）
            → Executor：对每个任务调用 Agent.chat()
              → Verifier：检查 patch、文件存在性
                → RecoveryManager：失败时决定 retry / skip / replan / abort
    → 完成
```

`RecoveryManager` 恢复策略：
- **Retry**（指数退避 1s→2s→4s→8s…max 60s）：网络超时、工具执行临时失败
- **Skip**：该任务可选，或下游任务能容忍它失败
- **Replan**：多个任务连续失败，让 LLM 重新规划
- **Abort**：不可恢复错误，终止流水线

---

## MCP 支持

通过 JSON-RPC over stdio 连接 MCP 服务器，自动发现其工具并包装为 CoreCoder `Tool` 实例。配置文件路径：`~/.corecoder/mcp.json`。

---

## 项目结构

```
corecoder/
├── __init__.py               # 惰性导入，暴露 Agent / LLM / Config / ALL_TOOLS
├── __main__.py               # python -m corecoder 入口
├── cli.py                    # CLI 参数解析 + REPL + Plan + 单次 三种模式
├── config.py                 # 从环境变量和 .env 文件加载配置
├── prompt.py                 # 系统提示词生成
│
├── agent/
│   ├── core.py               # Agent 类——ReAct 循环 + SSE 执行 + undo
│   ├── runtime/
│   │   ├── state.py          # SessionState dataclass（所有运行时认知）
│   │   ├── assembler.py      # 动态消息组装（瞬态前缀 + persistent_history）
│   │   └── staged.py         # Think-Execute 双层运行时（~1700 行，最复杂的模块）
│   ├── workflow/
│   │   ├── orchestrator.py   # Orchestrator——顶层 DAG 流水线协调器
│   │   ├── planner.py        # LLMPlanner / StaticPlanner
│   │   ├── scheduler.py      # Scheduler——依赖感知的任务调度
│   │   ├── executor.py       # Executor——Agent 包装器，桥接 任务节点 ↔ ReAct
│   │   └── verifier.py       # 验证层（PatchAnalysis + VerificationPolicyEngine）
│   └── dag/
│       ├── models.py         # TaskNode / ExecutionResult / RetryPolicy / TaskStatus
│       ├── graph.py          # TaskGraph——有向无环图 + 在线环检测
│       ├── memory.py         # WorkingMemory + MemoryInjector
│       └── recovery.py       # RecoveryManager（retry / skip / replan / abort）
│
├── llm/
│   ├── types.py              # ToolCall / LLMResponse / SSEEvent 数据结构
│   └── client.py             # LLM（OpenAI 兼容）+ LiteLLM（100+ 提供商）
│
├── tools/
│   ├── base.py               # Tool 抽象基类
│   ├── __init__.py           # ALL_TOOLS 注册表 + get_tool() 查找
│   ├── bash.py               # Shell 执行 + 危险命令拦截 + cd 跟踪
│   ├── read.py               # 按行号读取文件
│   ├── write.py              # 创建/覆盖文件
│   ├── edit.py               # Search-and-Replace 编辑（唯一性约束）
│   ├── glob_tool.py          # 文件模式匹配
│   ├── grep.py               # 正则内容搜索
│   ├── repo_info.py          # 仓库索引结构化查询
│   └── agent.py              # 子智能体 spawn
│
├── context/
│   ├── models.py             # ContextFragment / ContextBundle / ContextRequest
│   ├── orchestrator.py       # ContextOrchestrator——动态上下文组装引擎
│   ├── compression.py        # 3 层压缩（snip → summarize → collapse）
│   ├── session.py            # 会话持久化（save / load / list）
│   └── retriever.py          # RepositoryContextRetriever
│
├── retrieval/                # 零 Embedding 符号化仓库检索
│   ├── models.py             # SymbolInfo / FileSummary / RankedFile
│   ├── symbol_index.py       # SymbolOwnershipGraph（符号↔文件双向索引）
│   ├── summaries.py          # FileSummaryManager（启发式 + LLM 摘要）
│   ├── task_understanding.py # TaskUnderstandingAnalyzer
│   └── ranker.py             # StructuredRanker——多因素文件评分
│
├── codebase/
│   ├── shadow.py             # ShadowGit——独立 git 仓库做 checkpoint/undo/diff
│   └── indexing/
│       └── index.py          # RepoIndex——AST 符号提取 + 依赖分析
│
├── mcp/
│   └── client.py             # MCP 客户端（JSON-RPC over stdio）
│
└── infra/                    # 基础设施
    ├── storage.py            # JSONStorage 图持久化
    ├── observability.py      # 带类型的工序日志
    └── viz.py                # 终端任务图可视化
```

---

## 设计原则

- **Less is more**：每个模块只保留核心逻辑，删掉装饰代码
- **可读性优先于性能**：代码是给人读的，不是给机器优化的
- **瞬态与持久分离**：上下文注入和真实对话严格分开
- **零魔法**：没有复杂的元编程、没有隐式依赖注入、没有魔法常量
- **可魔改**：每个模块足够独立，可以单独替换而不影响其他模块

---

## 许可证

MIT License
