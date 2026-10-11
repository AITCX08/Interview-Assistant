# 显式扩展接口

核心使用 MIT 许可证，现有免费功能不依赖扩展、授权或商业库。核心不扫描插件目录，不自动导入第三方代码。

`extension_api.CORE_API_VERSION = 1`。单独安装的可信启动器可向 `main.InterviewAssistant(extension_factory=...)` 传入工厂，再通过 `gui.mount_extension(title, factory)` 创建页签。页签工厂接收 Tk parent 和不可变 `ExtensionContext`，可读取当前配置、预览简历和转写，通过 `context.post(callback)` 在 UI 主线程调度结果。扩展对象的 `close()` 在应用关闭时调用。

读取资料不代表获得网络发送许可；扩展必须提供独立明确的发送选择、取消和资料清空行为。仅传入可信代码：扩展与核心在同一 Python 进程，不构成安全沙箱。

`DeepSeekClient.stream_messages(settings, messages, cancel, on_delta)` 提供通用 SSE 传输。消息须为 system/user/assistant 的 role/content 字典，最多 40 条、合计 40000 字符；复用禁止重定向、总时限、取消、响应上限和脱敏错误处理。调用方负责提示、上下文内容、结果校验及用户发送许可。基础 `stream()` 的默认提示和简历选择保持原行为。

`local_data.EncryptedLocalStore(path)` 为所有扩展提供可选 Windows DPAPI JSON 存储，限制 512 KiB，原子替换，失败不改存明文，损坏文件保留供用户处理。`export_text(parent, text, markdown=False)` 仅在用户选择路径后导出指定文字，可能包含个人资料。两种接口均不自动收集 Key。

未来商业扩展可以使用私有策略与独立许可证；不能把已发布 MIT 核心的权利追溯改为商业授权。签名私钥、私人资料和商业实现不应进入本公开仓库。
