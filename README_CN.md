#UPDATE README_CN.md
基于原nanocoder修改而来

# CoreCoder

> 原名 **NanoCoder**，为避免与 [Nano-Collective/nanocoder](https://github.com/Nano-Collective/nanocoder) 混淆而改名。旧链接自动跳转到这里。


[English](README.md) | [中文](README_CN.md) | [Claude Code 源码深度导读（7 篇）](article/)

[![PyPI](https://img.shields.io/pypi/v/corecoder)](https://pypi.org/project/corecoder/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/he-yufeng/CoreCoder/actions/workflows/ci.yml/badge.svg)](https://github.com/he-yufeng/CoreCoder/actions)

**51万行 TypeScript → ~1,400 行 Python。**

我逆向了 Claude Code 泄露的全部源码，然后把不承重的部分全扔掉，用 Python 重建了核心。成果：**Claude Code 的每一个关键架构模式，浓缩在一个下午能读完的代码库里。**

CoreCoder 不仅是一个 AI 编程工具。它是一份**蓝图**，编程 Agent 领域的 [nanoGPT](https://github.com/karpathy/nanoGPT)。读懂它，fork 它，然后造你自己的。

---

```
$ corecoder -m kimi-k2.5

You > 读一下 main.py，修掉拼错的 import

  > read_file(file_path='main.py')
  > edit_file(file_path='main.py', ...)

--- a/main.py
+++ b/main.py
@@ -1 +1 @@
-from utils import halper
+from utils import helper

修好了：halper → helper。
```

## 你能得到什么

Claude Code 51 万行源码提炼出来的 7 个核心模式：

| 设计模式 | Claude Code | CoreCoder |
|---|---|---|
| 搜索替换编辑（唯一匹配 + diff） | FileEditTool | `tools/edit.py` — 70 行 |
| 并行工具执行 | StreamingToolExecutor（530行） | `agent.py` — ThreadPool |
| 三层上下文压缩 | HISTORY_SNIP → Microcompact → CONTEXT_COLLAPSE | `context.py` — 145 行 |
| 子代理隔离上下文 | AgentTool（1,397行） | `tools/agent.py` — 50 行 |
| 危险命令拦截 | BashTool（1,143行） | `tools/bash.py` — 95 行 |
| 会话持久化 | QueryEngine（1,295行） | `session.py` — 65 行 |
| 动态系统提示词 | prompts.ts（914行） | `prompt.py` — 35 行 |

每个模式都是可运行的实现，不是流程图，不是博客文章。

## 安装

```bash
pip install corecoder
```

选你的模型，任何 OpenAI 兼容 API 都行。可以 `export` 环境变量，也可以在项目根目录放一个 `.env` 文件：

```bash
# Kimi K2.5
export OPENAI_API_KEY=你的key OPENAI_BASE_URL=https://api.moonshot.ai/v1
corecoder -m kimi-k2.5

# Claude Opus 4.6（通过 OpenRouter）
export OPENAI_API_KEY=你的key OPENAI_BASE_URL=https://openrouter.ai/api/v1
corecoder -m anthropic/claude-opus-4-6

# OpenAI GPT-5
export OPENAI_API_KEY=sk-...
corecoder -m gpt-5

# DeepSeek V3
export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com
corecoder -m deepseek-chat

# Qwen 3.5
export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
corecoder -m qwen-max

# Ollama（本地）
export OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1
corecoder -m qwen3:32b

# 单次模式
corecoder -p "给 parse_config() 加上错误处理"
```

## 架构

整个项目一目了然：

```
corecoder/
├── __init__.py / __main__.py      # 包入口
├── cli.py                          # CLI REPL
├── agent.py                        # ReAct 循环
├── config.py                       # 配置
├── prompt.py                       # System prompt
│
├── llm/                            # LLM 接口
│   ├── types.py                    # ToolCall, LLMResponse, SSEEvent
│   └── client.py                   # LLM + LiteLLM
│
├── repo/                           # 仓库智能
│   ├── index.py                    # RepoIndex 符号/依赖索引
│   └── shadow.py                   # ShadowGit 影子仓库
│
├── mcp/                            # MCP 协议
│   └── client.py                   # MCPClient 子进程管理
│
├── history/                        # 对话历史
│   ├── compression.py              # ContextManager 3层压缩
│   └── session.py                  # 会话存续
│
├── tools/                          # 工具实现
│   ├── bash.py, read.py, write.py, edit.py, ...
│   └── agent.py                    # 子Agent 工具
│
└── orchestration/                  # DAG 编排层
    ├── orchestrator.py             # 顶层 Orchestrator
    ├── viz.py                      # 图可视化
    ├── storage.py                  # 持久化
    ├── observability.py            # 结构化日志
    │
    ├── dag/                        # "跑什么" — 图结构与状态
    │   ├── models.py               # TaskNode, ExecutionResult, ...
    │   ├── graph.py                # TaskGraph DAG
    │   ├── memory.py               # WorkingMemory + MemoryInjector
    │   └── recovery.py             # RecoveryManager 重试/回滚
    │
    ├── engine/                     # "怎么跑" — 执行引擎
    │   ├── scheduler.py            # Scheduler 依赖感知调度
    │   ├── planner.py              # Planner (Static + LLM)
    │   ├── executor.py             # Executor 包装 ReAct 循环
    │   └── verifier.py             # Verifier 验证层
    │
    └── context/                    # "跑的时候看什么" — 上下文编排
        ├── orchestrator.py         # ContextOrchestrator 主入口
        ├── models.py               # ContextFragment, TokenBudget, ...
        ├── layers.py               # 5 个上下文层
        ├── pipeline.py             # 分阶段流水线
        ├── ranker.py               # 多信号相关性排序
        ├── retriever.py            # 图感知仓库检索
        └── policies.py             # 7 种执行状态策略

```

## 当库用

```python
from corecoder import Agent, LLM

llm = LLM(model="kimi-k2.5", api_key="your-key", base_url="https://api.moonshot.ai/v1")
agent = Agent(llm=llm)
response = agent.chat("找出项目里所有 TODO 注释并列出来")
```

## 加自定义工具（约 20 行）

```python
from corecoder.tools.base import Tool

class HttpTool(Tool):
    name = "http"
    description = "请求一个 URL。"
    parameters = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}

    def execute(self, url: str) -> str:
        import urllib.request
        return urllib.request.urlopen(url).read().decode()[:5000]
```

## 命令

```
/model           查看当前模型
/model <名称>    切换模型
/compact         压缩上下文（对标 Claude Code 的 /compact）
/tokens          查看 token 用量 + 费用估算
/diff            查看本次会话修改的文件
/save            保存会话
/sessions        列出已保存的会话
/reset           清空历史
quit             退出
```

保存的会话 ID 会先安全化再作为文件名，恢复数据始终留在 `~/.corecoder/sessions` 目录内。

## 对比

|  | Claude Code | Claw-Code | Aider | CoreCoder |
|---|---|---|---|---|
| 代码量 | 51万行（闭源） | 10万+行 | 5万+行 | **~1,400 行** |
| 模型 | 仅 Anthropic | 多模型 | 多模型 | **任意 OpenAI 兼容** |
| 能通读吗？ | 不能 | 很难 | 有点费劲 | **一个下午** |
| 适合 | 直接用 | 直接用 | 直接用 | **先看懂，再造自己的** |

## 源码导读

我还写了 [7 篇 Claude Code 架构深度导读](article/)：Agent 循环、工具系统、上下文压缩、流式执行、多 Agent、隐藏功能。想知道 CoreCoder 为什么这样设计，从那里开始。

## FAQ

**CoreCoder 支持 Skill / Subagent / MCP 吗？**

不支持，这是刻意的。CoreCoder 只保留可运行的最小核心 —— agent 循环、工具、流式、压缩。Skill、Subagent、MCP、hook、plugin 都是 Claude Code 在上层加的特性；如果 CoreCoder 也全都做了，就不再是一个可读的教学产物。上面的架构导读系列讲了 Claude Code 里这些系统是怎么工作的，你可以照着自己加。

如果你只是想要 Skill，配方很简单：启动时扫 `~/.claude/skills/*.md`，把标题列进 system prompt，让 agent 按名字请求某个 skill，再把那个文件的内容 inline 进对话就行了。

## License

MIT。Fork，然后拿去造更好的东西，如果能标注此出处就更好了。

---

作者 **[何宇峰](https://github.com/he-yufeng)** · Agentic AI Researcher @ Moonshot AI (Kimi)

[Claude Code 源码分析（知乎 17 万阅读，6000收藏）](https://zhuanlan.zhihu.com/p/1898797658343862272)
