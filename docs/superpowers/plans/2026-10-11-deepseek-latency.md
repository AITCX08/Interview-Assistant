# 识别延迟与 DeepSeek 集成实施计划

> **供 Goal 执行：** 使用 `superpowers:executing-plans`，一个主代理串行执行，不委派实现和审查。

**目标：** 在现有桌面项目实现有界识别与可选自动 DeepSeek 流式建议，完成离线验证。

**架构：** 固定音频/识别/AI 工作线程，有界队列与任务世代隔离。Tkinter 主线程处理配置及显示，Key 仅存内存。

**技术栈：** Python 3.10.20、NumPy 2.2.6、Tkinter、FunASR 1.3.1、标准库 urllib/unittest。

**实施计划：** `docs/superpowers/plans/2026-10-11-deepseek-latency.md`

**Goal 提示词：** `docs/superpowers/plans/2026-10-11-deepseek-latency-goal-prompt.md`

## 全局约束和执行契约

- 保留 `.local/deepseek-checkpoint/preexisting.patch` 中的已有修改。
- 不上传真实音频/文本、不调用真实 DeepSeek、不安装依赖、不结束用户进程。
- 测试使用唯一 RunId、隐藏进程、独立 stdout/stderr/退出码文件；仅输出计数和最多十行日志。
- 按任务顺序推进，只在观察到通过结果后完成检查点；恢复时核对 PID、结果、Git 与本文件。

## 任务 1：有界识别链路

**前置任务：** 无。**检查点：** 分段、限流、旧结果与设备状态测试通过。

**文件：** `audio_capture.py`、`transcriber.py`、`tests/test_pipeline.py`、`scripts/run_checks.ps1`。

**接口：** `process_chunk(audio, callback)` 保持兼容；增加 `close()` 和指标/错误回调。

- [ ] 编写分段/音频队列/过载/失效测试，后台运行并确认现有实现失败。
- [ ] 以 NumPy 块缓冲、有界队列、固定 worker、世代编号实现最小修复。
- [ ] 聚焦测试通过，检查差异和安全边界，提交该任务卡。

## 任务 2：DeepSeek 流式客户端

**前置任务：** 任务 1 提交。**检查点：** SSE、异常、取消和排队替换测试通过。

**文件：** `ai_client.py`、`tests/test_ai_client.py`。

**接口：** `DeepSeekSettings`、`DeepSeekClient.stream()`、`AnswerService.submit()/cancel()/close()`，事件包含请求编号。

- [ ] 编写模拟 HTTP 流与单个待处理任务的失败测试。
- [ ] 实现固定官方端点、非思考流、输出上限、超时、关闭响应和取消隔离。
- [ ] 模拟测试通过，检查没有正文/Key 日志，提交该任务卡。

## 任务 3：GUI 与本地验证

**前置任务：** 任务 2 提交。**检查点：** 完整本地测试、GUI 模拟和性能报告通过。

**文件：** `gui.py`、`main.py`、`tests/test_gui.py`、`scripts/simulate_latency.py`、`README.md`、`.github/workflows/source-check.yml`、`docs/performance.md`。

**接口：** 主线程抓取配置；语音事件通过 GUI 调度自动请求；UI按请求编号隔离输出。

- [ ] 编写自动触发、Key 遮罩和旧事件不写回的失败测试。
- [ ] 增加配置/自动触发/回答区与延迟显示，修复停止和模型重载边界。
- [ ] 完整测试及合成音频/模拟 AI 的性能验证通过，记录局限；审查并提交。

## 完成条件

所有测试通过；GUI 可进行无 Key 的本地模拟；提供本机报告，不把模拟值当真实云服务或真实语音准确率。
