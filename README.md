```shell
conda create -n chatapi python=3.11
```

## 本地开发安装

在仓库根目录下：

```shell
pip install -e .
```

## 从私有 Git 服务器安装（其它项目 / 远程服务器）

```shell
pip install git+https://git.epsiotapi.com/EpsIotaPi/ChatAPI.git
```

私有仓库如需鉴权，把 token 拼进 URL（注意 token 会出现在 shell 历史/日志中，敏感环境建议改用环境变量或 `.netrc`）：

```shell
pip install git+https://<username>:<token>@git.epsiotapi.com/EpsIotaPi/ChatAPI.git
```

升级到某个具体版本（打 tag 后）：

```shell
pip install git+https://git.epsiotapi.com/EpsIotaPi/ChatAPI.git@v0.1.0
```

不指定 `@<ref>` 默认拉取默认分支最新代码；每次发布新功能后打一个 tag（并同步更新 `pyproject.toml` 的 `version`），下游项目就可以按需锁定版本或滚动升级到最新。

## 环境变量

按使用到的供应商设置对应的 API key：

- `OPENAI_KEY`
- `GOOGLE_KEY`
- `DEEPSEEK_KEY`

`PROMPTS_DIR`（可选）：`PromptManager` 默认从包内 `chatapi/prompts/prompts.json` 加载 prompt 模板；设置了 `PROMPTS_DIR` 后会改为从 `$PROMPTS_DIR/prompts.json` 加载，方便下游项目提供自己的一套 prompt 而不用改代码。
