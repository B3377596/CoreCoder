# CoreCoder 面试 Q&A

---

## 一、项目概述

### Q1: 一句话介绍 CoreCoder 是什么？

CoreCoder 是一个**极简、可读的 AI 编程智能体**，用约 2000 行 Python 从 Claude Code 源码中提炼出 7 个核心架构模式，定位是 "AI coding agent 的 nanoGPT"——一个让你看懂、魔改、自己造轮子的蓝图，而非生产工具。

### Q2: 项目的核心设计哲学是什么？

1. **less is more**：每个模块只保留承重墙，删掉所有装饰
2. **State-Centric 而非 Chat-History-Centric**：运行时认知走 SessionState 字段，不往聊天记录里注水
3. **Ephemeral context 与 Persistent history 严格分离**：仓库摘要、工作记忆、执行约束每次 turn 重新组装，绝不写入持久化历史
4. **零嵌入向量的符号化检索**：不需要 embedding 模型，用 AST + 符号图 + 启发式摘要做代码库理解

### Q3: 支持哪些 LLM？

任何兼容 OpenAI API 的模型都可以，包括 Kimi K2.5、GPT-5、DeepSeek、Qwen、Claude Opus 等，还通过 LiteLLM 支持 100+ 非 OpenAI 提供商（Bedrock、Vertex、Cohere 等）。

---

## 二、架构设计

### Q4: 项目的整体架构是怎样的？7 个核心模式分别是什么？

```
入口: cli.py (CLI解析 + REPL/单次/Plan三种模式)
  └─ Agent (core.py) — 核心 ReAct 循环
       ├─ LLM (client.py) — OpenAI + LiteLLM 客户端
       ├─ Tools (8个工具) — bash/read/write/edit/glob/grep/repo_info/agent
       ├─ Runtime (state.py + assembler.py) — 状态中心运行时 + 动态消息组装
       ├─ Context (compression.py) — 3层上下文压缩
       ├─ ShadowGit (shadow.py) — 影子Git做checkpoint/undo/diff
       ├─ RepoIndex (index.py) — AST符号提取 + 依赖分析
       ├─ Retrieval (retrieval/*) — 符号化仓库检索(无embedding)
       └─ Workflow (workflow/* + dag/*) — DAG任务编排(Plan模式)
```

7 个核心模式：
1. **Search-and-Replace 编辑**（精确单次匹配 + diff）
2. **并行工具执行**（asyncio.gather）
3. **3 层上下文压缩**（snip → summarize → collapse）
4. **子智能体隔离上下文**
5. **危险命令拦截**
6. **会话持久化**
7. **动态系统提示词构建**

### Q5: 为什么要用 State-Centric 架构而不是传统的 Chat-History-Centric？

传统的做法是把所有上下文（仓库结构、已完成步骤、约束条件等）直接塞进聊天历史里。这导致三个问题：

1. **压缩会损坏上下文**：当你对聊天历史做摘要时，仓库结构信息会被压缩成一团意义不明的文字
2. **撤消操作困难**：你分不清哪些是"真实对话"哪些是"注入的上下文"
3. **上下文污染**：LLM 难以区分"已发生的事情"和"环境信息"

State-Centric 的做法：
- `persistent_history` 只存真实对话（user/assistant/tool 消息）
- 仓库摘要、工作记忆、执行约束等存在 `SessionState` 的命名字段里
- 每次 turn 调用 `assembler` 重新组装瞬态前缀：`[system] + [assistant(mem)] + [assistant(repo)] + [assistant(run)] + persistent_history`
- 瞬态前缀绝不写入 persistent_history

### Q6: SessionState 的设计分了哪几个层级？各自的生命周期是什么？

```python
@dataclass
class SessionState:
    # session-long: 整个会话期间存活
    persistent_history     # 只有真实的 user/assistant/tool 对话
    repo_summary           # 仓库认知，惰性刷新

    # task-long: 每个任务重置
    active_files, active_symbols  # 当前任务涉及的文件/符号
    current_task, current_goal    # 当前任务描述
    completed_steps, important_decisions  # 可压缩的工作记忆
    failures, constraints          # 失败记录和约束

    # execution-long: 每次执行重置
    allowed_actions, forbidden_actions  # 边界控制
    stop_conditions, downstream_tasks   # 终止条件
```

这样分层的好处是：reset 时只需清空对应层级，不会丢掉整个对话历史。

### Q7: 消息组装器 (assembler) 是如何工作的？

`build_runtime_messages(state, system_prompt)` 每次调用时从零构建消息列表：

```
Layer 1: [system]              — 稳定的系统提示词（工具列表、规则）
Layer 2: [assistant(memory)]   — [WORKING MEMORY] 已完成步骤 + 关键决策
Layer 3: [assistant(repo)]     — [REPOSITORY CONTEXT] 仓库结构 + 活跃文件
Layer 4: [assistant(runtime)]  — [EXECUTION CONSTRAINTS] 允许/禁止/停止条件
Layer 5: current_turn          — 当前轮次的用户消息（可选，CLI单次模式用）
Layer 6: persistent_history    — 真实对话历史
```

关键设计：Layer 2-4 的内容**不会**出现在 persistent_history 中。这保证了压缩只触真实对话，checkpoint/undo 只操作对话边界。

---

## 三、Agent 循环与工具执行

### Q8: ReAct 循环的核心流程是什么？

`Agent.chat()` 的核心循环：

```
1. 将用户消息追加到 persistent_history
2. 用 ShadowGit 做快照（checkpoint）
3. 计算瞬态前缀 token 开销
4. 检查是否需要压缩（3层策略）
5. 进入 max_rounds=50 的循环：
   a. 构建消息列表（system + 瞬态前缀 + persistent_history）
   b. 通过 SSE 流式调用 LLM
   c. 如果 LLM 返回纯文本 → 结束，返回给用户
   d. 如果 LLM 返回工具调用 → 并行执行，结果追加到 persistent_history
   e. 回到步骤 a
6. 重建仓库索引（文件可能已被修改）
```

### Q9: SSE 流式执行有什么特别之处？

`_execute_turn_sse()` 实现了 **"边流式输出边执行工具"**：

1. 打开 LLM 的 SSE 流
2. 每当一个工具调用的 JSON 参数完整到达 → 立即 `asyncio.create_task` 启动后台执行
3. 流继续产出后续 token（可能是更多工具调用或文本）
4. 流结束后，`await` 所有后台任务的结果
5. 结果按顺序写回 persistent_history

这样工具执行和 LLM 输出是**重叠**的，减少了总延迟。比"等所有工具调用收齐了再一起执行"更高效。

### Q10: 工具执行的并行策略是怎样的？

```
_call_tools_parallel() 使用 asyncio.gather() 并发执行所有工具调用
```

但实际上，SSE 模式下用的是 fire-and-continue（收到一个启动一个），效果更优。

`_invoke()` 做了异步/同步兼容处理：
- 如果 `tool.execute` 是协程 → 直接 await
- 如果是同步函数 → `asyncio.to_thread` 放到线程池避免阻塞事件循环

---

## 四、工具系统

### Q11: 工具系统的设计是怎样的？

所有工具继承 `Tool` 抽象基类，需要提供 3 个字段：

```python
class Tool:
    name: str           # 工具名
    description: str    # 描述
    parameters: dict    # JSON Schema 格式的参数定义

    def execute(self, **kwargs) -> str:  # 执行并返回字符串结果
```

8 个内置工具：`bash`、`read_file`、`write_file`、`edit_file`、`glob`、`grep`、`repo_info`、`agent`。

工具注册在 `tools/__init__.py` 的 `ALL_TOOLS` 列表中，`get_tool(name)` 按名称查找。

### Q12: edit_file 工具为什么是 Claude Code 的核心创新？

传统做法是让 LLM 输出整个文件内容（whole-file rewrite），但有几个问题：
- 大文件浪费大量 token
- 容易出现格式偏差
- 不安全：可能会改到不相关的部分

edit_file 的做法是 **"精确子串匹配替换"**：

1. LLM 指定 `file_path`、`old_string`（要替换的原文）和 `new_string`（新内容）
2. 工具在文件中搜索 `old_string`
3. `count == 0` → 报错："没找到这段文字"
4. `count > 1` → 报错："这段文字出现了 N 次，请包含更多上下文使其唯一"
5. `count == 1` → 替换，返回 unified diff

**唯一性约束**（must appear exactly once）是这个设计的精髓：它强制 LLM 提供足够的上下文来确保替换位置正确，从根本上杜绝了歧义编辑。

### Q13: bash 工具有哪些安全措施？

9 条正则模式拦截危险命令：

| 模式 | 拦截内容 |
|------|---------|
| `rm -r* /~/\$HOME` | 递归删除家目录/根目录 |
| `rm -rf` | 强制递归删除 |
| `mkfs` | 格式化文件系统 |
| `dd ... of=/dev/` | 裸磁盘写入 |
| `> /dev/sd[a-z]` | 覆盖块设备 |
| `chmod -R 777 /` | 根目录全权限 |
| `:(){ :\|:& };:` | Fork 炸弹 |
| `curl \| bash` | 管道执行远程脚本 |
| `wget \| bash` | 同上 |

另外还有 120 秒超时、输出截断（长输出保留头 6000 字符 + 尾 3000 字符）、cd 命令跟踪（维护全局 `_cwd` 变量，支持跨命令的目录状态）。

### Q14: sub-agent（子智能体）工具是怎么实现的？

`agent.py`（约 50 行）的关键：

```python
class AgentTool(Tool):
    def execute(self, description, prompt, ...):
        # 创建一个隔离的 Agent 实例
        sub = Agent(llm=parent.llm, tools=parent.tools, ...)
        # 独立运行，有自己的上下文窗口
        result = await sub.chat(prompt)
        return result
```

子智能体的意义：当父智能体的上下文窗口已经被大量历史消息占满时，可以 spawn 一个"干净的"子智能体专注于某个子任务，完成后只返回结果摘要。

---

## 五、上下文压缩

### Q15: 3 层上下文压缩分别是什么？触发条件和策略是什么？

| 层级 | 触发阈值 | 策略 |
|------|---------|------|
| **Layer 1 (snip)** | 50% max_tokens | 将超过 1500 字符的工具输出截断为首尾各 3 行 |
| **Layer 2 (summarize)** | 70% max_tokens | LLM 摘要旧对话，保留最近 8 条消息完整 |
| **Layer 3 (collapse)** | 90% max_tokens | 紧急模式：只保留最后 4 条消息 + 硬摘要 |

关键细节：
- 压缩只触 `persistent_history`，瞬态前缀通过 `estimate_ephemeral_tokens()` 计入阈值判断但本身不参与压缩
- Layer 2 的摘要通过 LLM 生成，保留文件路径、关键决策、错误信息
- 如果 LLM 摘要失败，fallback 到 `_extract_key_info()` 做正则提取

### Q16: 为什么瞬态上下文不计入 persistent_history？

如果仓库摘要（可能几千 token）被写入 persistent_history，当压缩触发时：
- 仓库结构会被"摘要"成无意义的碎片
- agent 将失去对代码库结构的认知
- 或者在摘要中仓库信息占据主导，挤掉真正的对话信息

保持分离=压缩和仓库认知互不干扰。

### Q17: token 计数是怎么做的？

优先使用 `tiktoken` 的 `cl100k_base` 编码器（覆盖 GPT-4 / Claude / 大部分模型），精确计算 token 数。如果 tiktoken 未安装，fallback 到 `len(text) // 3` 的粗略估算。

---

## 六、影子 Git 与撤消

### Q18: ShadowGit 是什么？为什么不用用户的 .git？

ShadowGit 在 `~/.corecoder/shadow/<project-hash>/` 维护一个**独立的 git 仓库**，核心技巧是通过环境变量分离 git 目录和工作树：

```python
env["GIT_DIR"] = self.shadow_dir         # git 对象存这里
env["GIT_WORK_TREE"] = self.work_tree    # 操作的是真实项目文件
```

这样的好处：
- 不会污染用户的 git 历史
- 用户的 `.gitignore` 不受影响
- 可以独立 checkpoint/undo/diff
- 用户的 git hooks 不会被触发

### Q19: Checkpoint 和 Undo 的流程是怎样的？

**Checkpoint**（每次用户 turn 前）：
1. `shadow.snapshot(message)` → `git add -A` + `git commit --allow-empty`
2. 记录三元组：`(当前 persistent_history 长度, 描述, git commit hash)`

**Undo**（用户输入 `/undo`）：
1. 检查是否有 checkpoint 且确实有文件变更
2. pop 最后一个 checkpoint
3. `git reset --hard <commit>` 恢复文件
4. `persistent_history = persistent_history[:target_len]` 恢复对话
5. 如果没有更多 checkpoint → 清空全部历史

### Q20: ShadowGit 如何处理嵌套 git 仓库？

当 agent 在子目录执行 `git init` 创建了嵌套仓库时，`git add -A` 会报错。ShadowGit 的做法是：

1. 捕获 "does not have a commit checked out" 错误
2. 从错误信息中正则提取嵌套仓库路径
3. 自动将该路径追加到 `.gitignore`
4. 重试 `git add -A`

---

## 七、仓库索引与检索

### Q21: RepoIndex 是什么？怎么工作的？

`RepoIndex` 在 `<project>/.corecoder/` 下构建结构化的代码库知识：

1. **符号提取**：AST 解析提取类、函数、方法、签名
2. **依赖分析**：解析 `pyproject.toml`/`requirements.txt`/`package.json` 的外部依赖 + 文件间的 import 关系
3. **框架检测**：自动识别 FastAPI、Flask、Django、SQLAlchemy、React、Next.js 等
4. **仓库摘要**：生成包含入口点、依赖、关键符号的 Markdown 文档
5. **增量更新**：基于文件 mtime 的变化检测，有改动才重建

### Q22: 为什么要用符号化检索而不是 Embedding？

1. **零模型依赖**：不需要 embedding 模型（省显存、省部署）
2. **确定性**：符号匹配是精确的，不会出现语义检索"搜到不相关的文件"的问题
3. **可解释**：你清楚地知道为什么某个文件被检索到（因为包含某符号/被某模块导入/在依赖图中距离近）

代价是：无法理解自然语言语义（比如搜"认证逻辑"找不到名为 `login_handler` 的函数），需要依赖符号名称的精确或模糊匹配。

### Q23: 检索排序器 (StructuredRanker) 的多因素评分包括哪些维度？

1. **编辑距离**：符号名/文件名与查询词的相似度
2. **所有权分数**：查询涉及的符号是否定义在该文件中
3. **依赖深度**：文件在依赖图中的位置（离入口点近的更相关）
4. **摘要匹配**：文件用途描述与任务意图的匹配度
5. **最近修改时间**：最近修改的文件更可能相关
6. **任务对齐**：文件类型（测试/配置/源码）与任务类型的匹配

---

## 八、Plan 模式与 DAG 编排

### Q24: Plan 模式的完整流水线是什么？

```
用户输入 Goal
  → LLMPlanner: LLM 将目标分解为 TaskGraph (DAG)
    → Orchestrator: 协调整个流水线
      → Scheduler: 拓扑排序，找到就绪任务
        → MemoryInjector: 为每个任务构建 WorkingMemory
          → Executor: 对每个任务调用 Agent.chat()
            → Verifier: 检查 patch、文件是否存在
              → RecoveryManager: 失败时决定 retry/skip/replan/abort
      → (如果所有任务完成) → 返回结果
      → (如果有不可恢复失败) → Replan 或 Abort
```

### Q25: TaskGraph 如何保证无环？

在每次 `add_edge()` 时做**在线环检测**——添加边之前临时添加边，从 target 做前向 DFS 看能不能回到 source。如果能回来，说明这条边会形成环，拒绝添加。代价是 O(V+E) 每次加边，对于任务图（通常几十个节点）完全够用。

### Q26: RecoveryManager 的恢复策略是什么？

失败时的决策树：
1. **Retry**（指数退避）：临时错误（网络超时、工具执行失败）
2. **Skip**：该任务是可选的，或依赖它的任务可以容忍它失败
3. **Replan**：多个任务连续失败，说明计划本身有问题，让 LLM 重新规划
4. **Abort**：不可恢复的错误，终止整个流水线

重试策略带指数退避：1s → 2s → 4s → 8s → ...（最大 60s）

---

## 九、LLM 集成

### Q27: 如何支持多种 LLM 提供商？

两种客户端：

1. **OpenAI 客户端**：标准的 `openai` Python SDK，通过 `OPENAI_BASE_URL` 环境变量切换不同的兼容 API（Kimi、DeepSeek、OpenRouter 等）
2. **LiteLLM 客户端**：提供 100+ 提供商支持（Bedrock、Vertex、Cohere 等），自动翻译提供商特定的 API

两种客户端都实现了相同的接口（`chat()` 和 `chat_sse()`），Agent 不关心底层是哪个提供商。

### Q28: LLMResponse 和 SSEEvent 的数据结构是什么？

```python
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class LLMResponse:
    content: str | None
    reasoning_content: str | None
    tool_calls: list[ToolCall] | None
    prompt_tokens: int
    completion_tokens: int

@dataclass
class SSEEvent:
    type: str  # "text" | "reasoning" | "tool_call" | "done" | "error"
    token: str | None
    tool_call: ToolCall | None
    usage: dict | None
    error: str | None
```

`reasoning_content` 字段用于支持 DeepSeek-R1 等推理模型的思考过程。

---

## 十、工程实践

### Q29: 为什么选择 Python 而不是 TypeScript（Claude Code 的原生语言）？

1. **可读性优先**：Python 对 AI/AI Agent 研究者更友好，学习门槛更低
2. **生态优势**：OpenAI SDK、AST 解析（ast 标准库）、asyncio 都原生支持
3. **"蓝图"定位**：不适合生产部署，但适合快速理解和魔改

### Q30: 代码中有哪些值得学习的 Python 技巧？

1. **`from __future__ import annotations`**：延迟注解求值，解决前向引用问题
2. **`if TYPE_CHECKING:`**：条件导入，避免循环依赖
3. **lazy import**：`__init__.py` 中不立即导入所有模块，减少启动时间
4. **`inspect.iscoroutinefunction()`**：运行时检测函数是同步还是异步，做统一的 `_invoke` 调度
5. **`asyncio.to_thread()`**：把同步函数 safely 放到线程池执行，不阻塞事件循环
6. **dataclass + field(default_factory=list)**：避免可变默认值陷阱

### Q31: 项目的测试策略是什么？

8 个测试文件覆盖核心模块，使用 pytest。测试侧重：
- 工具执行（edit_file 的唯一性检查、bash 的安全拦截）
- 仓库索引（符号提取、依赖解析）
- DAG 图（环检测、拓扑排序）
- 上下文压缩（各层触发条件）

---

## 十一、深度思考题

### Q32: 如果让你改进 edit_file 工具，你会怎么做？

可能的思路：
- **模糊匹配**：当 old_string 不完全匹配时，用 difflib 找到最相似的片段并提示用户
- **多文件批量编辑**：允许一次请求编辑多个文件（原子性：要么全成功，要么全不应用）
- **编辑预览**：在实际写入前返回 diff 让用户确认
- **语义匹配**：不是精确字符串匹配，而是 AST 级别的代码块匹配（比如"把函数 foo 改成 bar"）

### Q33: State-Centric 架构的潜在问题是什么？

1. **状态一致性**：如果 assembler 和 compression 的字段不同步，LLM 会看到过时信息
2. **字段爆炸**：随着功能增多，SessionState 字段会越来越多，`apply_state_updates` 需要维护白名单
3. **序列化开销**：每次 turn 都要重新构建瞬态前缀，对大仓库可能有性能影响（不过有 retrieval_cache 缓解）
4. **学习曲线**：新开发者需要理解"哪些东西该放哪个层"，不像 chat-history-centric 那样直观

### Q34: 3 层压缩策略有什么局限性？

1. **摘要质量依赖 LLM**：Layer 2 的摘要效果取决于 LLM 本身的能力，弱模型可能丢关键信息
2. **阈值固定**：50%/70%/90% 是写死的，不同模型上下文窗口差异大（32K vs 256K）
3. **无增量压缩**：每次压缩都是全量重建摘要，没有缓存机制
4. **硬截断丢失信息**：Layer 1 的 snip 只保留首尾行，中间的代码/错误信息可能恰好是关键信息

### Q35: 如果要把 CoreCoder 从"蓝图"升级到"可生产使用"，需要做哪些事？

1. **安全性加固**：沙箱化 bash 执行（Docker 容器而非直接 shell）
2. **Hook 系统**：类似 Claude Code 的 hooks 机制，允许用户在执行前/后插入自定义逻辑
3. **权限系统**：细粒度的工具权限控制（允许读但禁止写某些目录）
4. **多模态支持**：图片输入（粘贴截图让 agent 改 UI）
5. **缓存优化**：LLM 响应的 prompt caching（Anthropic 风格），大幅降低成本
6. **更好的错误恢复**：网络断连自动重连、长任务断点续传
7. **IDE 集成**：VS Code / JetBrains 插件

### Q36: CoreCoder 和 Claude Code / Cursor / Copilot 的本质区别是什么？

| 维度 | CoreCoder | Claude Code / Cursor |
|------|-----------|---------------------|
| 定位 | 教学蓝图 | 生产工具 |
| 代码量 | ~2000 行 | 数万行 |
| 语言 | Python | TypeScript |
| 可靠性 | 实验级别 | 企业级别 |
| 安全性 | 基础正则 | 多层沙箱 |
| UI | 终端 REPL | IDE 集成 / 终端 |
| 模型 | 任何 OpenAI 兼容 | 深度绑定 Claude |

CoreCoder 的价值不在于"替代"它们，而在于"让人理解"它们是怎么工作的。

---

## 十二、快速突击题

- **Q: Agent 最多执行多少轮工具调用？** A: 50 轮（`max_rounds=50`）
- **Q: 工具输出超过多少字符会被截断？** A: 15000 字符（bash 工具），保留头 6000 + 尾 3000
- **Q: 瞬态前缀估算的作用是什么？** A: 让压缩阈值准确反映总上下文大小，而不是只看 persistent_history
- **Q: edit_file 的 old_string 必须满足什么条件？** A: 在目标文件中出现**恰好一次**
- **Q: ShadowGit 存储在哪个目录？** A: `~/.corecoder/shadow/<project-md5-hash>/`
- **Q: agent 工具传入的 _parent_agent 和 _repo_index 是怎么注入的？** A: Agent.__init__ 中遍历 self.tools，用 isinstance 检查并注入
- **Q: 什么时候会触发 Layer 3 (hard collapse)？** A: 总 token 超过 max_tokens 的 90%
- **Q: persistent_history 里只存什么？** A: 只存真实对话——user 消息、assistant 文本回复、assistant tool_call 消息、tool result 消息
