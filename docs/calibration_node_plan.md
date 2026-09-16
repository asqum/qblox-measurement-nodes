# Calibration Node 建置企劃書

最後更新:2026-09-15。這份文件是「持續更新」的活文件——每次新增/修改 node,請同步更新下面的狀態表與缺口清單,不要另開一份新文件。

## 1. 目的

對齊使用者在 Notion 上規劃的完整 calibration node 清單,與目前 `src/qblox_lab/experiments/` 的實際建置進度,並把「independent / joint flux」這個目前還沒有任何 node 真正做出來的維度講清楚——這是 Notion 上幾乎每一列都打了 tag、但程式碼裡完全沒有對應開關的最大缺口。之後每次要建新 node,先回來看這份文件的「狀態表」跟「缺口清單」,再決定要接著做哪一個。

## 2. 現況總覽(以程式碼實際狀態為準,2026-08-28 確認)

```
src/qblox_lab/experiments/
  cal01_time_of_flight.py
  cal02_resonator_spectroscopy_full_bandwidth.py
  cal03_resonator_spectroscopy.py
  cal04_resonator_punchout.py
  cal05_resonator_flux_spectroscopy.py
  cal06_qubit_spectroscopy_full_bandwidth.py
  cal07_qubit_spectroscopy_intrinsic_width.py
  cal08_qubit_flux_spectroscopy.py (2026-09-15 起改用 VoltageOffset flux pulse,見下方異動說明)
  cal09_rabi.py                (mode="power"/"time" 二選一,同一份檔案)
  cal10_ramsey_vs_flux.py      (新增,2026-09-15,見下方新增說明)
  cal11_energy_relaxation.py   (T1Analysis)
  cal12_ramsey.py
  cal13_iq_blob.py
  cal14_readout_frequency_optimization.py
  cal15_readout_power_optimization.py
  cal16_power_rabi_state.py    (新增,2026-09-15,見下方新增說明)
  dev_active_reset.py          (Phase 0 驗證用,未進 calNN 序列)
  dev_joint_flux_pulse_check.py(Phase 0 驗證用,未進 calNN 序列)
  dev_snz_flux_pulse_check.py  (Phase 0 驗證用,未進 calNN 序列,2026-08-28 新增,見下方說明)
```

**下一個空號:`cal17`。** 指派新編號前務必先 `ls src/qblox_lab/experiments/` 再確認一次,不要照這份文件的數字直接套(之前發生過中途插入 node 而全部往後移一位的狀況,這次(2026-09-15)又發生了一次)。

**修正(2026-08-28):`cal08_qubit_flux_spectroscopy.py` 的缺口 2 依賴是誤判。** §3.3 缺口清單原本把「Qubit flux spectroscopy」列為需要先完成缺口 1、2(coupler flux config 路徑驗證)才能動工,但實際確認後這個 node 只需要被測 qubit自己的 flux line——機制完全比照 `cal05_resonator_flux_spectroscopy.py` 的 independent DC-ramp(`resolve_flux_offset_parameter` + `SetParameter`),不會解析、也不會碰到任何 coupler port。已跟使用者確認範圍:被測 qubit 掃自己的 flux,其餘 qubit(來自 `flux_config` 的所有項目,扣掉被測的)在量測前用 `apply_flux_config(hardware_agent, flux_config, qubits=<非被測清單>)` 統一 park 到 idle,量測過程中不會再被排程碰到。`qblox_scheduler.analysis.spectroscopy_analysis.QubitFluxSpectroscopyAnalysis` 是現成的分析類別(quadratic 模型抓 sweetspot / sweetspot_freq),跟 cal05 用的 `ResonatorFluxSpectroscopyAnalysis` 同一套 dataset 慣例(x0=Frequency, x1=Offset, y0=Magnitude, y1=Phase)。已用 `config/example/` 的 dummy hardware 跑過 `build_schedule()` → `simulated_data()` → `analysis()` → `plot()` 全流程驗證通過。**joint 模式仍然缺,`flux_mode` 參數也還沒加**——之後要幫這個 node 補 joint(例如同時掃 qubit + coupler)時,缺口 1、2 才是真正的前提,現在的 independent-only 版本不受影響。

**編號調整(2026-08-28,同日稍後):`Qubit flux spectroscopy` 從 `cal14` 改編為 `cal08`,原本的 `cal08`–`cal13` 依序往後推一位變成 `cal09`–`cal14`。** 使用者要求把它排在 `cal07`(qubit spectroscopy)後面、`cal08 Rabi` 前面,理由是它在量測順序上更接近既有的 resonator/qubit spectroscopy 系列,不是跟 readout optimization 系列排在一起。對應調整:`cal08_rabi.py`→`cal09_rabi.py`、`cal09_energy_relaxation.py`→`cal10_energy_relaxation.py`、`cal10_ramsey.py`→`cal11_ramsey.py`、`cal11_iq_blob.py`→`cal12_iq_blob.py`、`cal12_readout_frequency_optimization.py`→`cal13_readout_frequency_optimization.py`、`cal13_readout_power_optimization.py`→`cal14_readout_power_optimization.py`;對應的 `runNN_*.ipynb` 也全部同步改名跟改 import。改名時發現 `run14_qubit_flux_spectroscopy.ipynb` 曾被使用者在真實硬體(`ic_cluster_A_module19`)上實際執行過一次(q2 flux 掃描結果:sweet_spot=-0.193250 V,sweet_spot_frequency=3.321936014 GHz)——那份帶執行結果的版本已保留並改名為 `run08_qubit_flux_spectroscopy.ipynb`(只改了 import 那一行的模組名稱,輸出/結果原封不動),沒有被我先前建立的空白版本蓋掉。**下一個空號目前是 `cal15`(前面章節已同步更新),不再是 `cal14`。**

**重大異動(2026-09-15):`cal08_qubit_flux_spectroscopy.py` 的 flux 機制從 §3.1 的 independent `SetParameter` DC-ramp,整個換成 §3.2 的 `VoltageOffset` in-schedule flux pulse(使用者明確指示直接取代,而非新開一份 node)。**新機制內容:每個被測 qubit 各自把 flux 脈衝到「相對於外部 `apply_flux_config` 設定之 idle bias 的偏移量」(`flux_pulse_start`/`flux_pulse_stop`,不再是絕對電壓),並保持脈衝與 CW drive 橫跨整個頻率掃描(仿 `cal07b_CW_flux_qubit_spectroscopy.py` 的模式),讀出後歸零回 idle——機制直接從 `dev_joint_flux_pulse_check.py` 的 `JointFluxPulseDriveCheck`(Phase 0 原型)搬過來,分析管線也一併換成該原型的 PCA 旋轉 + Lorentzian 擬合 + branch-tracking + sigma-clipping(取代原本內建的 `QubitFluxSpectroscopyAnalysis`,因為在這個訊噪比下太脆弱)。多 qubit 支援維持,但實作方式改成每顆 qubit 各自的 flux×frequency 區塊在同一個 repetitions 迴圈內**依序**串接(不用 `ref_op`,理由同 §3.2 的 compiler bug),不是同時觸發——**因此嚴格來說這仍然是「一次一條 flux 線」的 independent 效果,只是底層换成 pulse 機制,並不構成 §3.2/§3.3 定義的真 joint(多元件同時觸發)**,狀態表(§4.3)的 joint 欄位不因此改成 ✅。

**已知風險,尚未解決:這次替換跳過了原本 §3.2 要求的驗證步驟**——原計畫是先跑 `dev_joint_flux_pulse_check.py` 對照舊版 `cal08`(SetParameter 機制)的 sweet spot 結果,確認一致後才把 pulse 機制併入正式 node;這次是使用者直接要求跳過比對、直接取代。`notebooks/run08_qubit_flux_spectroscopy.ipynb` 先前在真實硬體上用舊機制跑過的執行結果(q1 sweet_spot=0.184164 V,2026-09-10)已因此作廢,新機制目前**還沒有任何真實硬體驗證**,下次執行前建議先過一遍 `simulated_data()`。

**新增(2026-09-15):`cal10_ramsey_vs_flux.py` + `run10_ramsey_vs_flux.ipynb`,插入 `cal09`(Rabi)跟原本的 `cal10`(Energy relaxation)之間,原本 `cal10`–`cal14` 依序往後推一位變成 `cal11`–`cal15`(對應 `runNN_*.ipynb` 與其內部 import 同步改名,`cal14`/`cal15` 檔案裡對 `cal12_iq_blob.py` 的文字引用也已改成 `cal13_iq_blob.py`)。** 仿照使用者提供的 QM 參考檔 `/home/reny871224/QM/06a_Ramsey_vs_Flux_Calibration.py` 移植:單一(非正負交錯)artificial detuning 的 Ramsey 序列,搭配在 idle/自由演化窗格期間才開啟的 flux 脈衝(`VoltageOffset` 開 → `IdlePulse(delay)` 持續整個 idle 時間 → 關,語意上是「相對 idle 的偏移量」,同 `cal08_qubit_flux_spectroscopy.py` 現行機制,而非 CW 整個頻率掃描期間都開著),detuning 用 `cal12_ramsey.py` 已驗證過的 virtual Z(`ShiftClockPhase`)做,而非切換 drive clock 頻率。每個 flux 點各自對 Ramsey delay 掃描做一次 `RamseyAnalysis` 擬合取得該點的振盪頻率,再對「振盪頻率 vs flux」做二次擬合,一次解出 flux 偏壓修正量(拋物線頂點的 flux 位置)跟 qubit 頻率修正量(頂點處的振盪頻率相對於人工 detuning 的偏差)——`update_device()` 同時套用兩者。**比 QM 參考檔做了一個明確改進**:參考檔算 virtual Z 相位時只用 `delay`,忽略了 flux 脈衝前後的 settle buffer(參考檔自己也留了 TODO 承認這是近似);這裡改用 `delay + 2*pulse_settle_time`(脈衝實際跨越的完整時長)算相位,相位計算精確。多 qubit 一樣採 `cal08_qubit_flux_spectroscopy.py` 的序列觸發(不用 `ref_op`,理由同 §3.2 的 compiler bug),不是同時,所以也不是 §3.2/§3.3 定義的真 joint。用 `config/example/` dummy hardware 跑過 `build_schedule()`(thermal 與 active reset 都測了)→`compile()`→`simulated_data()`→`analysis()`→`plot()`→`update_device()`→`read_flux_biases()`/`save_flux_biases()`/`save_device_configuration()` 全流程,並確認 `simulated_data()` 在物理上合理的 curvature 下能準確還原模擬的 flux_offset/freq_offset/quad_term;同時發現一個真實限制並寫進 `analysis()` docstring:掃描範圍/curvature 太大導致 detuning 在掃描範圍內變號時,per-flux 的 Ramsey 擬合會把負頻率摺回正值,`>0` 過濾抓不到,會悄悄拉歪二次擬合——這也是為什麼此 node 設計上只適合窄範圍的局部微調(先用 `cal08_qubit_flux_spectroscopy.py` 找到大致 sweet spot,再用這個 node 微調),不是拿來做大範圍 flux spectroscopy。真實硬體尚未驗證。

**新增(2026-09-15):`cal16_power_rabi_state.py` + `run16_power_rabi_state.ipynb`,插在最後,不需要 renumber(`cal15` 之後的下一個空號本來就是 `cal16`)。** 移植自使用者提供的 QM 參考檔 `/home/reny871224/QM/09_Power_Rabi_State.py`:重複播放同一個單 qubit 旋轉(x180/x90/-x90/y90/-y90)`N` 次來放大振幅誤差,同時掃 `N` 跟振幅縮放係數(相對 `rxy.amp180` 的乘數,不是絕對值),用 `cal13_iq_blob.py` 校準過的 `measure.acq_rotation`/`acq_threshold` 把每個 single-shot 讀出判讀成 0/1,對 `repetitions` 個 shot 取平均得到每個 `(N, 振幅)` 點的 population。兩種模式:只掃到 `N=1`(`max_number_of_pulses` 設小)時退化成一般 power Rabi,對 population vs 振幅做餘弦擬合;`N` 有多個值時才是真正的 error amplification——對每個振幅把 population 沿 `N` 軸取平均,取平均值最大的振幅(振幅誤差會隨重複次數放大,只有校準正確的振幅在所有 `N` 都維持接近 1 的 population)。

**確認的 compiler 限制,刻意偏離參考檔的地方**:QM 用一個「重複次數本身是即時迴圈變數」的動態迴圈(`for_(count, 0, count<npi, count+1)`,`npi` 又是外層迴圈變數)。直接測試確認:目前安裝的 `qblox_scheduler` 的 `arange`/`linspace` 迴圈域建構子的 `stop` 參數只接受具體數值,傳入 Expression 會直接 `TypeError`——這代表「重複次數本身可即時變化」這件事目前這個 scheduler 版本做不到。改成在 Python 端(schedule 建構時)展開 `N` 這個維度(對每個 `N` 值各自建一段 amplitude 即時迴圈,`N` 次旋轉也是 Python for-loop 展開,不是即時迴圈),只有振幅軸維持即時迴圈——代價是編譯出來的 schedule 會包含 `sum(n_pi_values)` 個實際的旋轉操作,不是精簡的動態迴圈,所以 notebook 預設把 `max_number_of_pulses` 設得遠比參考檔的 200 保守(避免 schedule 過大)。另一個簡化:這個 scheduler 的裝置模型只有一個 `rxy.amp180` 參數,所有 `Rxy` 系列閘(`X90`/`Y90`...)編譯時都是從它內部算比例縮放出來的,所以不像參考檔那樣需要額外同步一份「x90 振幅」(`update_x90` 選項在這裡沒有對應需求)。

**開發過程中修正的一個真實 bug**:一開始照抄參考檔「`phi_fit - pi*(phi_fit>pi/2)` 再解 `factor=(pi-phi_fit)/(2*pi*f_fit)`」的相位代數,結果在單振盪擬合模式下算出的 pi 振幅偏差了近 100 倍(用 `simulated_data()` 驗證抓到的)——原因是 lmfit 擬合出的 `amplitude` 參數可能是負值(等效於相位多轉 π),此時該公式的象限判斷就不成立。改成數值解法:把擬合出的餘弦曲線在振幅掃描範圍內取細網格,直接用 `scipy.signal.find_peaks` 找第一個局部極大值當作 pi 振幅——不管擬合出的振幅/相位正負號慣例為何都穩健,用 `simulated_data()` 驗證準確還原模擬的 true_amplitude_factor(單振盪模式與 error-amplification 模式都各自測過,含 thermal/active reset 兩種 reset 都跑過 `build_schedule()`→`compile()`)。真實硬體尚未驗證,且必須先跑過 `cal13_iq_blob.py` 校準 `acq_rotation`/`acq_threshold`,否則 `analysis()` 會用未校準的預設值(0, 0)誤判每個 shot 的狀態。

**基礎設施更新(2026-08-28):硬體/元件/flux 三份設定檔已從 3-qubit 範例擴充成完整 10 qubit + 9 coupler 拓樸。**
- [`config/hw_config_AS_QRC.json`](../config/hw_config_AS_QRC.json):q1–q10、qw(witness,只有 drive+readout,無 flux)、cpl1_2–cpl9_10(只有 flux,無 mw/res)全部接線完成。Flux 模組假設為 `QCM`(4 個 real_output)、drive 模組假設為 `QCM_RF`(2 個 complex_output)、FeedLineA/B 假設為 `QRM_RF`——這是依 Qblox 慣例推斷,尚未跟使用者的實際模組型號逐一核對。
- [`config/dut_config.json`](../config/dut_config.json):q1–q10 + qw 共 11 個 `BasicTransmonElement`,acq_channel 0–10 全域唯一。**Coupler 沒有被加成 element**——查證確認 `qblox_scheduler` 的 `DeviceElement.__model_registry__` 只有 `BasicTransmonElement`/`BasicElectronicNVElement`/`BasicSpinElement`/`ChargeSensor` 四種,沒有 coupler 專用類型;CZ gate 的 `CompositeSquareEdge` 也證實是直接用兩個**qubit** element 的 flux port,不需要 coupler 被註冊成 element。Coupler 的存在完全只在 hw_config 的硬體接線層。
- [`config/flux_config.json`](../config/flux_config.json):q1–q10 idle bias 之外,新增 9 個 coupler entries(`cpl1_2`…`cpl9_10`),每個都帶顯式 `"port": "cplX_Y:fl"`。**同步小改了 [`src/qblox_lab/config/hardware.py`](../src/qblox_lab/config/hardware.py) 的 `apply_flux_config`**:當某個 key 在 `get_element()` 找不到對應 element 時(例如 coupler),改用該 entry 顯式指定的 `"port"` 欄位直接丟給 `resolve_flux_offset_parameter()`——這個函式本來就只查硬體接線圖、不管 `QuantumDevice` 有沒有註冊該 element,所以這個 fallback 是安全的,不影響既有 qubit 路徑。**這是缺口 2(coupler flux config 路徑驗證)的部分進度**——程式邏輯已經打通且用 dummy connection 驗證過 `.get()`/`.set()` 可正常呼叫,但**還沒在真實硬體上跑過** coupler 的 ramp,缺口 2 在硬體驗證前不算解決。
- 所有佔位數值(`clock_freqs`、`output_att`、`rxy`/`measure` 參數等)都是沿用原本 q0/q1-q3 範例的假資料,之後要照每個 qubit/coupler 實際校準結果覆蓋。

## 3. 核心缺口:independent / joint flux 目前完全沒有落地

這是這次企劃書要解決的主要問題。先把兩個模式的定義、現況、跟缺口講清楚,再套進下面的狀態表。

### 3.1 Independent 模式 —— 已存在,是目前所有 node 的唯一模式

機制:`apply_flux_config()`([`config/hardware.py:132`](../src/qblox_lab/config/hardware.py))呼叫 `resolve_flux_offset_parameter()`([`config/hardware.py:88`](../src/qblox_lab/config/hardware.py)),透過 QCoDeS 的 `outN_offset` DC 參數,在 shot 之間用 ramp 的方式,把**呼叫時明確列出的 qubit(s)**各自打到 `flux_config_AS_QRC.json` 裡各自預先配好的 idle 值。

現況:`cal01`–`cal07`、`cal09`–`cal14` 都是這個模式——每個 node 只會動到建構時傳入的 `qubit_names`,其餘 qubit / coupler 完全不會被這次實驗碰到,會維持前一次殘留的偏壓(不會自動 park)。`cal08_qubit_flux_spectroscopy.py` 是例外:它會額外把被測 qubit 以外、`flux_config` 裡列出的所有 qubit 主動 park 到 idle,而且從 2026-09-15 起,被測 qubit 自己也不再用這裡講的 SetParameter DC-ramp,改用下方 §3.2 的 `VoltageOffset` in-schedule pulse(見上方「重大異動」說明)——效果上仍是一次只動一條 flux 線,所以歸類上還是算 independent,但底層機制已經是 pulse 而非 DC-ramp,是目前唯一混用兩種機制的 node。

`cal05_resonator_flux_spectroscopy.py` 看起來像「多 qubit joint」,但確認過後其實不是——它是把**同一個掃描值**,用同一套 `SetParameter` DC-ramp 機制,同時推給列表中每個 qubit 各自的 flux line(`schedule.loop(...) as flux_offset: for qubit in self.qubits: schedule.add(SetParameter(...))`)。這是「多個 qubit 共用同一條掃描軸」,不是「多個元件可以各自獨立設定、且能在單一 shot 內快速切換」的真 joint。

### 3.2 Joint 模式 —— 規劃中,連 Phase 0 硬體驗證都還沒做完

需要的能力:在單一個 shot 的排程內,用 `VoltageOffset` pulse 把多個元件(qubit + coupler)同時或依序推到指定電壓,結束後再退回 0V 讓外部 DC 靜態偏壓重新掌控。原因是 QCoDeS 的 DC ramp 只能在 shot 與 shot 之間切換,做不到 shot 內快速切換(例如未來 CZ-gate chevron 需要的「進 interaction 點 → 做閘 → 退回 idle」)。

現況:目前唯一的原型是 `dev_joint_flux_pulse_check.py`(驗證用,未進 calNN 序列),但**只支援單一 qubit**的 VoltageOffset pulse-and-return(`schedule.add(VoltageOffset(flux_pulse_amplitude, 0, port=port))` → `IdlePulse` → 退回 `VoltageOffset(0.0, 0, port=port)`),還沒推廣到「多個 qubit / coupler 同時」的情境。換句話說,連 Phase 0 的硬體驗證都還沒做到真正的「joint」。

**新發現(2026-08-28,重要,會影響缺口 1 的做法):`qblox_scheduler`(目前安裝版本)有一個會直接擋死「多元件同時」設計的 compiler bug。** 在建 `dev_snz_flux_pulse_check.py`(見下方新原型說明)時,原本想用 `ref_op`/`ref_pt="start"` 把 coupler 的 flux pulse 跟兩個 qubit 的 drive pulse 對齊成真正同時開始,結果編譯階段一直丟 `ValueError: Node ... not found in schedulables`。用 dummy connection 反覆 dry-run 二分排除後確認:**只要 `ref_op` 關聯到的任一方是 baseband/real-valued pulse(`VoltageOffset` 或 `SuddenNetZeroPulse`,也就是任何 flux port 操作),reference-graph 驗證器就會壞掉**——跟是否在 `schedule.loop()` 裡面、anchor 有沒有被重複參照、幾個 sibling、哪個 qubit/port 都無關;純 IQ 系(drive/readout)彼此用 `ref_op` 完全沒事(`dev_joint_flux_pulse_check.py` 現有的單 qubit 版本之所以能用,是因為它本來就沒有用 `ref_op` 對齊任何東西,只是單純循序 `schedule.add(..., rel_time=None)`)。**這代表「同時觸發多個 flux 通道」這個 joint 模式的核心需求,目前這個 scheduler 版本用 `ref_op` 這條路走不通**,要嘛升級 `qblox_scheduler` 版本確認是否修掉這個 bug,要嘛改用手動算好時間的 `rel_time`/pulse 的 `t0` 參數自己排絕對時間,不能再假設 `ref_op` 可以套用在任何牽涉 flux port 的操作上。

**`dev_snz_flux_pulse_check.py`(新原型,2026-08-28 新增,未進 calNN 序列):** 驗證 `SuddenNetZeroPulse`(CZ gate 標準的 SNZ flux pulse 波形,Negirneac 2021)能不能在 coupler 的 flux port 上正確編譯/下發/回到 idle,鎖定 q1/q2 + cpl1_2。因為上述 bug,最終改成**循序式**(兩個 qubit 平行 reset+X90 → coupler SNZ pulse → 兩個 qubit 平行 measure),不是真正的「同時重疊」,但已經用 dummy connection 驗證過 `build_schedule()` → `compile()` → `plot_pulse_diagram()` 全部正常,**還沒上真實硬體跑**。只做機制驗證,不做 chevron 掃描/物理分析(沒有現成的 coupler flux ground truth 可比對)。

### 3.3 要補的缺口(照相依順序排列,前面沒做完後面不能開工)

1. **把 `dev_joint_flux_pulse_check.py` 從單 qubit 推廣成多元件同時 VoltageOffset pulse**,在真實硬體上驗證多通道同時觸發沒有時序 / crosstalk 問題。這是所有 joint node 的共同前提。**更新(2026-08-28):`ref_op` 對齊法在目前 `qblox_scheduler` 版本下對 flux pulse 完全不可用(見上方 §3.2 的 bug 說明),這條缺口現在多了一個前置阻擋——要嘛先確認升級 scheduler 版本能不能修掉這個 bug,要嘛改設計成手動算絕對時間(`rel_time`/`t0`)的排程方式,「同時」目前無法用 `ref_op` 這條路徑做到,`dev_snz_flux_pulse_check.py` 目前只能做循序版本。**
2. **Coupler flux config 的路徑驗證**——`flux_config_AS_QRC.json` 雖然已經列出 9 個 coupler 的 idle 值,但 `apply_flux_config` / `resolve_flux_offset_parameter` 解析 coupler port 的路徑目前沒有被任何現有 node 實際跑過(全部現有 node 都只用 qubit),需要單獨驗證。這是 `resonator_spectroscopy_coupler_flux` 動工前的前置需求。(修正,2026-08-28:`Qubit flux spectroscopy` 的 independent 版本不碰 coupler,不受這條阻擋,已用 `cal08_qubit_flux_spectroscopy.py` 建成——只有它未來的 joint 版本才會需要這條。)**進度更新(2026-08-28):`apply_flux_config`(`hardware.py`)已加上 coupler port fallback(見上方 §2 基礎設施更新),`config/flux_config.json` 也已補上 9 個 coupler entries,邏輯已用 dummy connection 驗證過,但真實硬體上的 ramp 還沒測過,缺口本身尚未關閉。**
3. **定義顯式的 `flux_mode` 參數**,跟 `cal13_iq_blob.py` 的 `multiplexed: bool` 同等級,例如 `flux_mode: Literal["independent", "joint"]`,讓同一份 node 檔案能透過參數在兩種模式間切換,而不是每個模式各寫一份檔案。要延續既有的「不建共用 module」慣例(每個檔案 copy-paste 自己的 helper),所以這個邏輯要在每個 node 檔案內各自複製一份,不集中成 shared util。
4. 上面三項都在真實硬體驗證通過後,才把 joint 模式陸續補進下面 Phase B 列出的既有 node。
5. **CZ gate(SNZ)參數的儲存位置——目前完全沒地方寫,需要先寫一個新的 Edge 類別。**(2026-08-28 查證)`qblox_scheduler` 的 `dut_config.json` 支援頂層 `"edges"` 區塊,透過 `QuantumDevice.add_edge()` 載入兩比特閘參數,概念上跟 `"elements"` 平行、一樣用 `edge_type` 欄位做 discriminated union 自動註冊(`Edge.__model_registry__`)。但內建只有 `CompositeSquareEdge`/`SpinEdge` 兩個 Edge 類別,參數都是方波格式(`square_amp`/`square_duration`/`parent_phase_correction`/`child_phase_correction`),完全沒有 SNZ 需要的 `amp_A`/`amp_B`/`net_zero_A_scale`/`t_pulse`/`t_phi`/`t_integral_correction`。`CompositeSquareEdge` 官方文件明講自己是「An example Edge implementation」,設計成範本讓人抄——要用 SNZ 就得寫一個新的 `CompositeSuddenNetZeroEdge`(仿 `composite_square_edge.py`,把 `cz` submodule 換成 SNZ 那 6 個欄位,`factory_func` 指向一個包 `SuddenNetZeroPulse` 的新 factory),寫好會自動註冊、不用改任何框架程式碼。這件事排在 `dev_snz_flux_pulse_check.py` 真實硬體驗證通過、確定要往正式 CZ chevron node 推進之後再做,現在只是先把路線圖記下來。

## 4. 完整 node 狀態表(對照 Notion 三個分類)

### 4.1 Qubit properties

| Notion 項目 | 對應程式 | independent | joint | 備註 |
|---|---|---|---|---|
| T1 & T1 Statistic | T1 已有 `cal11_energy_relaxation.py`;T1 Statistic 無 | ✅(T1 部分) | ❌ | T1 Statistic 卡在等使用者提供 Quantum Machine 參考腳本,尚未排期 |
| T2 echo | 無 | ❌ | ❌ | 全新,需要先找/寫 reference |
| IQ blob | `cal13_iq_blob.py`(已建立) | ✅ | ❌ | 已有 `multiplexed: bool`,但那是控制**讀取**是否同時進行,跟 flux 的 independent/joint 是兩件事,不要混用同一個參數 |
| CPMG | 無 | ❌ | ❌ | 同 T1 Statistic,等 QM 參考腳本 |
| All_XY | 無 | ❌ | ❌ | 全新 |
| Single qubit RB | 無(cal15 候選) | ❌ | ❌ | 全新,規格待定 |

### 4.2 Error amplification

| Notion 項目 | 對應程式 | independent | joint | 備註 |
|---|---|---|---|---|
| Ramsey | `cal12_ramsey.py`(已建立) | ✅ | ❌ | |
| Ramsey flux | `cal10_ramsey_vs_flux.py`(已建立,2026-09-15) | ✅(見上方新增說明) | ❌ | 用 flux 脈衝(相對 idle 偏移量)搭配 virtual-Z Ramsey 做窄範圍局部微調,同時解 flux 偏壓與 qubit 頻率修正,不是缺口 1 定義的多元件同時 joint |
| Power Rabi optimization | `cal16_power_rabi_state.py`(已建立,2026-09-15) | ✅ | ❌ | 移植自 QM `09_Power_Rabi_State.py`(節點名稱本身就叫「error amplification」),語意上比 `cal09_rabi.py` 的一般 power Rabi 更貼近「optimization」這個字——但這是推測,尚未跟使用者確認過命名對應,見第 6 節開放問題。跟下面「Power Rabi」是否為同一個 node 也需要確認 |
| Readout power optimization | `cal15_readout_power_optimization.py`(已建立) | ✅ | ❌ | |
| Readout frequency optimization | `cal14_readout_frequency_optimization.py`(已建立) | ✅ | ❌ | |
| Drag calibration | 無 | ❌ | ❌ | 全新 |
| AC stark shift | 無 | ❌ | ❌ | 全新 |

### 4.3 Resonator / Qubit spectroscopy

| Notion 項目 | 對應程式 | independent | joint | 備註 |
|---|---|---|---|---|
| resonator_spectroscopy | `cal03_resonator_spectroscopy.py`(已建立) | ✅ | ❌ | |
| resonator_spectroscopy_qubit_flux | `cal05_resonator_flux_spectroscopy.py`(已建立) | ✅(lock-step 多 qubit,見 3.1) | ❌ | 目前的多 qubit 掃描不是真 joint,補 joint 時要重新設計掃描機制,不能只加參數 |
| resonator_spectroscopy_coupler_flux | `cal05b_resonator_coupler_flux_spectroscopy.py`(已建立,2026-09-01) | ✅ | ❌ | 只掃 coupler 自己的 flux(`resolve_flux_offset_parameter`+`SetParameter`,同 cal05/cal08 機制),被測的兩個 qubit park 到各自 idle,不做 QM 參考檔的 qubit-flux crosstalk 補償(需要缺口 1 的 joint 機制)。分析沿用 `ResonatorFluxSpectroscopyAnalysis`(同 cal05),沒有搬 QM 參考檔那套 dip-detection 式 decouple-offset 分析。已用 `create_dummy_connections=True` + AS_QRC config(q1/q2 + `cpl1_2`)跑通 `build_schedule`→`hardware_agent.compile`→`simulated_data`→`analysis`→`plot` 全流程,確認缺口 2 的 coupler port 解析路徑可用；q4/q5 + `cpl4_5` 組合在 compile 階段會撞到既有的 NCO 頻率範圍限制(`cal05` 用 q4/q5 一樣會撞到,是既有 config 問題,非本節點新增),notebook 預設改用 q1/q2。真實硬體上的 coupler DC ramp 仍未驗證,缺口 2 尚未完全關閉。 |
| resonator_spectroscopy_vs_power | `cal04_resonator_punchout.py`(已建立,即 `run04`) | ✅ | ❌ | |
| Qubit_spectroscopy | `cal06`/`cal07`(已建立) | ✅ | ❌ | Notion 標「完成」,但那是指 independent 版本完成,joint 仍待做 |
| Qubit flux spectroscopy | `cal08_qubit_flux_spectroscopy.py`(已建立,2026-08-28;2026-09-15 flux 機制改為 VoltageOffset pulse) | ✅(pulse 機制,見 §2 異動說明) | ❌ | 只動被測 qubit 自己的 flux,其餘 qubit park 到 idle,不碰 coupler,所以不受缺口 2 阻擋。2026-09-15 起改用 `dev_joint_flux_pulse_check.py` 的 VoltageOffset pulse 機制取代 SetParameter DC-ramp,但多 qubit 仍是序列觸發、非同時,故仍歸類 independent,不算真 joint;真實硬體尚未驗證新機制 |
| Power Rabi | 疑似同 `cal09_rabi.py` | ✅ | ❌ | 見第 6 節開放問題 |

## 5. 建議施工順序

**Phase A(前置,阻擋所有 joint node)**
1. 推廣 `dev_joint_flux_pulse_check.py` 支援多元件同時 VoltageOffset,真實硬體驗證。
2. 驗證 coupler flux config 的 resolve 路徑。
3. 訂出 `flux_mode` 參數的命名與行為規範(可以先用一份新的 `dev_*` 驗證檔案試跑,通過後再變成規範)。

**Phase B(把 joint 補進已存在的 node,建議順序)**
`cal05`(resonator flux spectroscopy)→ `cal12`(Ramsey)→ `cal06`/`cal07`(Qubit spectroscopy)→ 其餘 independent-only 的 node,依實際需求決定是否真的需要 joint 版本(例如 readout optimization 的 joint 版本意義是什麼,建議先跟使用者確認,不要照單全收)。

**Phase C(全新 node)**
`cal16` 起,依使用者當下優先序決定:目前看起來 All_XY / Single qubit RB / Drag calibration / AC stark shift / T2 echo 都還沒有 reference 腳本,T1 Statistic 與 CPMG 明確卡在等 QM 範例。

## 6. 開放問題(需要使用者決定,先不要自行假設)

1. **「Power Rabi optimization」(Error amplification 分類)跟「Power Rabi」(spectroscopy 分類)是同一個 node 重複列出,還是兩個不同的實驗?**(2026-09-15 更新)目前暫時對應成:「Power Rabi」= `cal09_rabi.py` 的一般 power Rabi(單次擬合振盪抓 pi 振幅),「Power Rabi optimization」= `cal16_power_rabi_state.py` 的 error-amplification 版本(重複播放 N 次放大振幅誤差,更精確)——這個對應是根據 QM 參考檔命名推測的,還沒跟使用者確認過,需要跟使用者核對這個猜測是否正確。
2. **哪些 node 真的需要 joint 模式?** Notion 上幾乎每一列都打了 independent/joint tag,但像 readout frequency/power optimization 這種本質上是「找當下最佳讀取參數」的 node,joint 版本的物理意義(多 qubit 同時找最佳讀取點?)需要使用者確認,不要為了填 tag 而硬做。
3. **cal15 之後的建置順序**,目前是照 Notion 表格由上而下猜測優先序,實際要看哪個 node 對當前實驗最急迫。

## 7. 相關背景(既有架構決策,持續有效)

- 不建共用 module,所有 helper 都是每個 node 檔案各自 copy-paste 一份(`_drive_port_clock`/`_readout_port_clock`/`_apply_flux_point` 等)。
- `ConditionalReset`(active reset)目前只有 `dev_active_reset.py` 驗證過,尚未加進任何正式 node 的 `reset_type="active"`。
- `multiplexed: bool`(`cal13_iq_blob.py`)是讀取時序的開關(同時讀 vs. 依序讀),跟這份文件討論的 flux `independent`/`joint` 是完全不同的兩個維度,新增 node 時不要把兩者混成同一個參數。
