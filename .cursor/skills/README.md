# 项目 Skills

本目录是 OneMix 的 **项目级 Agent Skill** 存放处。写在这里的 skill 会随仓库共享，任何用 Cursor 打开本项目的人都能用。

个人全局 skill 请放在 `~/.cursor/skills/`，不要写进本仓库。

## 目录约定

每个 skill 独立成一个目录，目录名即 skill 名（小写字母、数字、连字符，最长 64）：

```text
.cursor/skills/
├── README.md                 本说明
└── skill-name/
    ├── SKILL.md              必填：frontmatter + 正文指令
    ├── reference.md          选填：详细参考（按需再读）
    ├── examples.md           选填：示例
    └── scripts/              选填：可执行辅助脚本
```

`SKILL.md` 最小结构：

```markdown
---
name: skill-name
description: 第三人称说明做什么、何时使用。包含触发关键词。
---

# Skill Name

## Instructions
给 Agent 的步骤说明。只写模型本身不知道的项目约定。
```

- `name`：与目录名一致。
- `description`：写清 **做什么** 和 **何时用**，Agent 靠它决定要不要加载。
- 正文尽量短于 500 行；细节放到同级 `reference.md` / `examples.md`，从 `SKILL.md` 只链一层。
- 路径一律用正斜杠：`scripts/helper.py`。
- 默认不要加 `disable-model-invocation`；只有「必须被点名才加载」时才设为 `true`。

## 什么时候新建 skill

适合固化成 skill 的，是本项目反复出现、且通用模型不知道的流程，例如：

- 改提示词模板 / 槽位规划时要遵守的字段和兜底格式
- 生图策略（即梦 / 万相）联调与回归步骤
- 前后端接口字段对齐（`SlotRow` ↔ `SlotJobIn`）
- Docker 发布检查清单

一次性问答、纯代码风格约束，不要做成 skill：风格写 `.cursor/rules/`，笔记写 `.cursor/lgx/`（本地，不入库）。

## 不要放这里

| 内容 | 正确位置 |
| --- | --- |
| Cursor 内置 skill | `~/.cursor/skills-cursor/`（系统管理，勿改） |
| 个人跨项目 skill | `~/.cursor/skills/` |
| 始终生效的编码约定 | `.cursor/rules/*.mdc` |
| 仅自己看的项目笔记 | `.cursor/lgx/`（已被 gitignore） |
