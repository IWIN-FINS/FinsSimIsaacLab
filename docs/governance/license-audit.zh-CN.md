# FinsSimIsaacLab 许可证与授权链审计

审计日期：2026-09-21。本文是面向开源发布的工程审计，不构成法律意见；在发版、
签署商业合同或公开二进制/数据集前，应由实际权利人或律师完成最终确认。

## 结论摘要

1. 已确认由 FinsSim 自主编写的 Python 源码、脚本和文档可采用 Apache-2.0，前提是
   IWIN-FINS 对相应贡献拥有足以授权的著作权。不能在未完成文件级溯源前，把所有
   `source/` 或 `runtime/` 代码都作出这一主张。
2. `third_party/isaac-auv-env` 的固定上游提交和从其中复制的 WarpAUV USD 资产为
   **BSD-3-Clause**，不是此前声明的 MIT；本仓保留其原始 LICENSE，且
   `THIRD_PARTY_NOTICES.md` 已更正。
3. Isaac Lab 为 BSD-3-Clause，但 Isaac Sim/Omniverse Kit 的已安装组件另有 NVIDIA
   条款。本仓不应打包或再授权这些外部安装内容。
4. FinsROV 的 FBX 与生成 USD 来自内部 Unity 资产链，历史 WarpAUV `.pt` 来自上游
   训练产物。两者都缺少足以支持公开再分发的独立权利证明，因此目前必须作为
   **发布阻断项**，除非补齐权利链或从公开发行物移除。

## 推荐授权链

```text
IWIN-FINS 原创代码与文档
  -> Apache-2.0（根 LICENSE）
  -> 可选的书面商业协议（不撤回 Apache 已授予权利）

WarpAUV 上游代码与 USD 资产（固定 commit 7c5ebe7）
  -> BSD-3-Clause（保留上游 LICENSE、版权、免责声明）
  -> 不并入 Apache-2.0 的所有权主张

Isaac Lab / Isaac Sim / Python 依赖
  -> 独立安装、独立许可证
  -> 本仓不授予其任何再授权

FinsROV FBX/USD 与历史 .pt checkpoint
  -> 权利链未完成确认
  -> 在确认前排除出公开 release 与 Apache grant
```

## 逐项资源盘点

| 资源 | 位置 | 当前结论 | 发布动作 |
| --- | --- | --- | --- |
| 已确认 FinsSim 自主编写的 Python、physics 与脚本 | 对应的 `runtime/`、`source/`、`scripts/`、`tests/` 文件 | 可采用 Apache-2.0，但需确认贡献者授权/雇佣关系。 | 保留 LICENSE、NOTICE、贡献记录。 |
| WarpAUV 参考源码 | `third_party/isaac-auv-env/` | BSD-3-Clause；本地 LICENSE 与锁定上游 commit 一致。 | 保留完整 LICENSE 和 NOTICE；不得标为 MIT 或 Apache 自有代码。 |
| WarpAUV Lab 3 适配代码 | `assets/warpauv/`、`tasks/direct/warpauv/` 及相关 compatibility 代码 | 工程记录称其为上游适配，但目前没有文件级来源 header。 | 在完成逐文件比对/clean-room 证明前，按 BSD-3-Clause 派生内容管理，并补充来源与修改声明。 |
| WarpAUV USD/mesh | `assets/warpauv/data/` | 从同一上游适配导入；暂按 BSD-3-Clause 管理。 | 保留源、commit、许可证；若发现 USD 内部另有引用或声明，应补充。 |
| FinsROV FBX、USD、物理 hull | `assets/finsrov/` | 从内部 Unity `FinsROV_Fossen` 导入；manifest 有路径/哈希，但没有外部素材权利证明。 | 公开发布前取得版权主体与所有嵌入素材的书面确认；否则移出公开仓库/发行包。 |
| 历史 WarpAUV checkpoint | `runtime/legacy_checkpoints/*.pt` | 模型权重没有显式发布许可记录。 | 默认不公开再分发；取得上游明确许可或从发行内容排除。 |
| Isaac Lab | 外部依赖 | BSD-3-Clause；依赖/资产按所安装发行版本另行处理。 | 不 vendor；用户按上游安装与许可证操作。 |
| Isaac Sim / Omniverse Kit | 外部依赖 | NVIDIA 专有/附加条款。 | 不提交、镜像、打包或以 Apache-2.0 再授权。 |
| PyPI / CUDA 依赖 | lockfile、外部环境 | 许可证随具体包与二进制而不同。 | 发行容器或 wheel bundle 前生成 SBOM 和依赖 notices。 |

## 证据与上游来源

- 锁定的 WarpAUV commit：[`7c5ebe7`](https://github.com/warplab/isaac-auv-env/tree/7c5ebe7f7a08acd2570b5fba328e92b7f59f6794)；其
  [`LICENSE`](https://github.com/warplab/isaac-auv-env/blob/7c5ebe7f7a08acd2570b5fba328e92b7f59f6794/LICENSE)
  是 BSD-3-Clause。
- [Isaac Lab 官方仓库](https://github.com/isaac-sim/IsaacLab) 声明框架为
  BSD-3-Clause，并指出完整 PhysX、RTX、ROS 与 importer 工作流需要适用独立条款的
  Isaac Sim。
- FinsROV `asset_manifest.json` 记录了 Unity 源 prefab/profile、FBX 哈希和转换入口；
  它是技术追溯证据，而非著作权转让或第三方素材授权。
- `third_party/isaac-auv-env/README.md` 引用的 `imgs/qual-overview.png` 未包含在本地
  快照；这是上游参考文档的完整性问题，不影响 BSD-3-Clause 文本，但公开发布前应
  删除该失效图链、补入可再分发图片，或保留其上游原始目录结构。

## 公开发布前清单

- [ ] 确认 IWIN-FINS 的完整法定主体、签约代表及其对 FinsSim 贡献的授权来源。
- [ ] 对每位外部贡献者、承包方和导入代码完成贡献者许可或权利转让记录。
- [ ] 完成 WarpAUV Lab 3 port 的逐文件溯源：复制/改编文件保留 BSD-3-Clause header，
      仅独立创作文件采用 Apache-2.0 header。
- [ ] 对 FinsROV render mesh、physics mesh、材质、纹理及来源资产完成逐项权利确认；
      无法确认则从公开仓库与 release 包移除。
- [ ] 获得历史 `.pt` checkpoint 的明确发布许可，或将其从公共 Git 历史和发行物移除。
- [ ] 保留 WarpAUV BSD-3-Clause LICENSE、copyright 与免责声明，并核查 USD 内嵌引用。
- [ ] 发布 Docker、wheel、SDK 或二进制时，针对实际运行的 Isaac Sim/Isaac Lab/PyPI
      依赖生成 SBOM 与第三方 notices。
- [ ] 发布前再次审阅 `THIRD_PARTY_NOTICES.md`、`NOTICE` 和本审计，确认没有将
      第三方内容误标为 Apache-2.0。
