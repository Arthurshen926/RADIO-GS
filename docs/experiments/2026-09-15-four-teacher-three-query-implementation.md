# 四教师与三类 query：实现、实跑和未解决问题

当前候选已实现四教师辅助监督、三类实际元素召回监督，并完成联合训练、去 RADIO 对照和原生 query 冷启动。它仍是 figurines 的源视角开发实验，不能称为完整任务精度已恢复，也不能替代历史有效主线。

## 当前实现

冻结 MoGe 几何提供 274,805 个高斯及固定前向遮挡贡献。持久地图只有一个 64 维可学习元素特征表；所有 query 都产生同一公式的元素后验，再用固定几何贡献渲染到 2D。语义优化器不包含位置、尺度、旋转或 opacity。

- RADIO：原始 1280 维 backbone 特征重建及原生两两余弦关系。
- DINO：原生 **DINOv2-B/14** 的 768 维特征重建及关系；不是 DINOv3，也不是 RADIO adaptor 输出。
- SAM：原生 SAM3 源视角区域伪标签，以及有几何证据处的内部连续性、轮廓特征差异和已知 membership 边界监督。没有把 SAM 称为稠密语义向量教师。
- SigLIP：原生 SigLIP2 的 1536 维区域身份重建；原生文本/图像编码使用同一语义空间，并显式共享可学习 query 投影。

辅助头仅用于训练，不保存在冷推理场中。`--teachers dino,sam,siglip` 不加载 RADIO 教师文件，且安装文件访问审计钩子，禁止读取 RADIO 训练特征。`--teachers ''` 已完成三步真实训练，证明辅助头全部关闭仍可执行；这不是无辅助教师的精度实验。离线 SAM 伪标签和选择的 query encoder 仍有各自依赖，不能把“关闭辅助损失”说成这些依赖也消失。

三个任务的主损失均包括 3D 元素 membership 与跨视角渲染 membership 的 BCE + Dice。文本取同一伪类别区域并集；图像使用源区域 crop；prompt 使用源 RGB 的正负点 DINO 特征与已注册世界坐标。图像目前是 masked crop，普通未分割 crop 的鲁棒性尚未验证。prompt 是一个正点和一个负点的已注册协议，尚不是任意数量、未注册自由交互。

正负区域重叠时保持未知；无观测和未匹配区域不自动标为负例。负例需要跨视角互相匹配的第二个区域、双视角不重叠及不同外观证据；这仍是伪对应，不保证真实物理实例身份。

## 本次修复的具体问题

1. 本地 Transformers 将 SigLIP 文本输出头构造成 1152→1152，但原始 checkpoint 是 1152→1536。按权重实际形状重建输出头并严格加载全部文本权重，避免随机投影混入。
2. 正提示点原先监督最大贡献高斯；124 个区域中有 9 个所选高斯的区域 membership 小于 0.7，说明单个载体跨边界。改为监督源像素的完整渲染混合响应，避免把像素标签强写成单高斯标签。
3. 文本和图像原先各学一个投影，破坏有限监督下的原生 SigLIP 跨模态共享。现在由显式配置共享权重；其他不一致的空间仍独立适配。
4. SAM 的 membership 边界损失只作用于已知区域，不能覆盖未知侧轮廓。新增结构轮廓损失：轮廓表示特征不连续性，不代表外侧一定是检索负例。
5. RADIO 源 RGB、SAM 源 RGB、各教师文件、几何贡献表和完整几何张量绑定都进行了核对。8 个 RADIO 源图哈希与几何源图一致。

## 实跑范围与结果

新责任表分辨率 96×128；训练源帧 11/21/31/38/49/69，留出 query 与目标帧 88/96。两组没有交叉训练 episode。原始 DINO/RADIO 特征仍是 46×62，插值不增加原生特征细节。几何本身使用的 mapping 图像不等于语义训练集；这不是几何也完全未见过的场景测试。

124 个 SAM 区域生成 76 个训练任务（文本 12、图像 32、prompt 32）和 22 个留出任务（6/8/8）；98 个任务中 81 个存在显式负区域。每组训练 180 步，固定种子与任务顺序，公共辅助头按名称独立初始化。

以下为 **已知支持区域上的伪标签 IoU**，阈值固定 0.5，不是正式 benchmark IoU：

| 配置 | 文本 | 图像 | Prompt |
|---|---:|---:|---:|
| 四教师，独立文本/图像投影 | 0.2816 | 0.3324 | 0.3159 |
| 四教师，共享 SigLIP 投影 | **0.4291** | **0.3428** | **0.3372** |
| 去 RADIO，共享 SigLIP 投影 | 0.4362 | 0.3165 | 0.3135 |

共享投影四教师的训练 IoU 为 0.9920 / 0.9840 / 0.9387，仍有明显泛化差距。上述对照是用于修改方法的 Development Evidence；已经查看这些留出视角，不能再把它们称作最终盲测。没有进行显著性检验或多种子结论。

评估的已知像素只占约 16%–17%。训练文本只有 water/toy/remote/camera/wall 五个伪类别，留出文本是 apple/camera/fruit；留出目标高置信元素平均仅约 32.6% 被训练正例覆盖。固定词表还会给出错误类别。因此不能据此声称完整开放词汇语义监督已经解决，也不能把未知区域中的高响应统计成可靠的 false positive 或忽略整个地图的泄漏问题。

## 冷推理与验证

原生冷启动脚本加载 checkpoint、冻结几何、原生模型和外部 query 输入。审计钩子禁止读取四种教师训练文件、episodes、queries 和 RADIO 特征目录。文本从字符串编码、图像从实际 crop 编码、prompt 从源 RGB 与已注册点重新编码；输入 fixture 中的参考向量仅作编码一致性对照，不用于生成推理 query。

- 每类输出覆盖全部 274,805 行，并单独返回 92,249 个有训练观测支持的有效行；未知行分数不可直接当有效检索。
- checkpoint 绑定完整几何、query encoder 权重文件和预处理身份；几何训练前后不变。
- 冷重载相同编码的后验误差为 0。实际原生重编码允许 fp16/batch 差异，详见独立冷报告；它是执行一致性验证，不是任务精度评测。
- 回归测试 **468 passed，4 warnings**。覆盖未知梯度、冲突区域、文本并集、教师关闭、关系基变换、轮廓有效性、共享投影和原有几何/渲染/v4 合约。

## 入口与复现顺序

1. `radio_gs.scripts.build_frozen_geometry_authority`：用已封存几何报告创建新贡献表和源图清单。
2. `radio_gs.scripts.prepare_four_teacher_retrieval --root ROOT`：生成四类源教师与 query 数据；`--without-radio` 跳过 RADIO 资产。SAM 源清单仅复用原生 2D mask，不复用旧 3D 标签。
3. `radio_gs.scripts.compile_three_query_episodes --root ROOT --geometry GEOMETRY`：在新高斯域建立跨视角监督。
4. `radio_gs.scripts.prepare_native_three_query_inputs`：显式传入 `--root`、`--geometry`、`--sam-source-index`、`--siglip-root`、`--dino-checkpoint`、`--dino-repo`，封存原生 encoder 身份和实际冷 query 输入。
5. `radio_gs.scripts.train_three_query_field --root ROOT --geometry GEOMETRY --name RUN --steps 180 --share-siglip-adapter`。去 RADIO 添加 `--teachers dino,sam,siglip`。关闭教师辅助不改变伪标签来源。
6. `scripts/audits/cold_three_query_native.py`：显式传入 fixture、场文件及其 SHA256，进行原生冷编码。

运行环境为 `/root/miniconda3/envs/cybersim_agent/bin/python`，设置 `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=.`。脚本拒绝覆盖完成的产物。

本次产物根目录：`/mnt/pool/sqy/results/RADIO-GS/output/four_teacher_three_query_v2_20260915/figurines`。当前候选为 `all_four_shared` / `without_radio_shared`，较早的 `all_four`、`*_relations`、`*_final` 是保留的开发对照，不是按名字选择的最终结果。

仓库证据：

- `paper/artifacts/four_teacher_three_query_20260915.json`
- `paper/artifacts/three_query_supervision_coverage_20260915.json`
- `paper/artifacts/cold_three_query_native_shared_20260915.json`
- `paper/artifacts/three_query_shared_heldout_20260915.png`

## 下一阶段的真实缺口

优先扩充源视角和可靠跨视角区域身份，提高每个目标的正负元素覆盖；用可靠的语义描述、软类别分布与人工抽查替代仅有少数词的 top-1 伪标签。训练/开发/最终评估应分别封存，不能用当前留出集继续调参后报最终成绩。

随后在相同几何和同一评估合约下评估更多场景、完整 query 集、全图响应泄漏、边界精度与覆盖率，报告多种子及教师消融。未知区域如何获得可验证标签是评估工作的一部分，不能通过将其一律视为负例掩盖。几何细节和覆盖率缺口仍按独立几何验收处理；特征训练不负责修复它们。

历史有效主线继续作为性能对照。本次完成的是四教师、三 query 的可执行开发闭环；完整 benchmark 恢复、可靠语义标签覆盖和生产级 query 协议仍未完成。
