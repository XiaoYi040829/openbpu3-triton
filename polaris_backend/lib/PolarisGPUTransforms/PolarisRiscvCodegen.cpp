//===- PolarisRiscvCodegen.cpp - Triton IR -> 真 RV64 RV-SIMT 机器码 -*- C++ -*-===//
//
// 本 pass 发射 RV64 + RV-SIMT 机器码（编码经 GNU as 交叉验证），产出的 .bin
// 可直接由周期精确 RTL 模拟器（build/rtl/emu）执行，也可 DMA 进 QV19P 的
// DDR4 由核执行。
//
// 生成的代码遵循 OpenBPU-3 的 SIMT 编程范式：
//   csrw THREAD_NUM -> csrw THREAD_BASEID -> START_WARP -> TID
//   -> csrr WARP_ID/THREAD_BASEID 算全局 id -> 计算 -> SYN_WARP -> END_WARP
//
// 内存布局：代码从 0x80000000 起；第 i 个指针参数在 ARG_BASE + i*ARG_STRIDE
// （0x80002000 起）；只读常量池在 CONST_BASE（0x80020000）。host 把输入写进
// 参数区，kernel 结果写回对应指针参数指向的区域。
//
//===----------------------------------------------------------------------===//
#include "PolarisGPU/Passes.h"
#include "PolarisGPU/RiscvAsmWriter.h"
#include "PolarisGPU/RiscvEncoder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <cstdlib>   // getenv: 读取 POLARIS_* 调试开关
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace mlir;
using namespace polaris;
using namespace polaris::rv;

namespace {


// kernel 指针参数的裸机约定: 第 i 个指针 = ARG_BASE + i*ARG_STRIDE
// host 侧(openbpu3_backend.py)按同一约定把输入数据放进镜像、从此处取回结果。
// 参数区从 0x80002000 起，.text 上限 8 KB。不放到 0x80100000：那是 QV19P
// 板上镜像收尾做 cache 驱逐时遍历读取的 256 KB 区域。.text 超过 8 KB 的
// kernel（如 mha 双层全展开）需缩小 constexpr 规模。
static constexpr uint32_t ARG_BASE = 0x80002000u;
static constexpr uint32_t ARG_STRIDE = 0x1000u;

// 浮点常量池（只读）：host 侧预置 [0.0, 1.0, 0.5, 2^23]，kernel 只 flw 不写。
// 地址必须高于全部参数区（ARG_BASE + i*ARG_STRIDE，i < 6），否则参数数据会
// 覆盖常量。0x80020000 同时避开板上镜像收尾的 cache 驱逐区。
static constexpr uint32_t CONST_BASE = 0x80020000u;
// 保留给常量的浮点寄存器（不参与 SSA 值分配，mapFReg 的范围要避开）
enum : uint32_t { FC_ZERO = 28, FC_ONE = 29, FC_HALF = 30, FC_MAGIC = 31 };

// 可用作中间值的 RISC-V 通用寄存器（避开 x0/sp/gp/tp 和我们固定占用的）
// x5=t0 x6=t1 x7=t2 保留给 prologue；x28-x31 留作临时
// 中间值可用的整数寄存器。**必须与参数寄存器分离**：若共用同一池子，
// 寄存器不够时会覆盖已装载的参数基址，kernel 访存到错误地址。
// 参数固定用 s1(x9)/s2(x18)/s3(x19)/s4(x20)/s5(x21)/s6(x22)，最多 6 个指针参数。
static const uint32_t kParamRegs[] = {9, 18, 19, 20, 21, 22};
static constexpr int kNumParamRegs =
    (int)(sizeof(kParamRegs) / sizeof(kParamRegs[0]));

// 中间值池。**必须排除 emitOp 内部用作临时的 x15/x16/x17 与 x30/x31**：
// select / floor / remsi / cmpi 都会往这些寄存器写中间量，若它们同时被
// 分配给活跃的 SSA 值就会被冲掉。
static const uint32_t kAllocRegs[] = {
    23, 24, 25, 26, 27,  // s7..s11
    10, 11, 12, 13, 14,  // a0..a4   (a5=x15/a6=x16/a7=x17 留作 emitOp 临时)
    28, 29,              // t3,t4    (t5=x30/t6=x31 留作 emitOp 临时)
    5, 6, 7,             // t0,t1,t2 prologue 用完即弃
    1, 3, 4, 8,          // ra,gp,tp,s0 —— 裸机无调用/无全局指针，可自由使用
    2,                   // sp —— 本后端不用栈，也不生成 spill 代码
};
static constexpr int kNumAllocRegs =
    (int)(sizeof(kAllocRegs) / sizeof(kAllocRegs[0]));

// ---- 安全浮点减法：绕开 fsub.s 的硬件缺陷 ----
// **两操作数指数差恰为 1 且结果为负**时，fsub.s 丢失符号、只保留幅值：
//     0.6-1.0 -> +0.4（应 -0.4）   1.5-3.0 -> +1.5（应 -1.5）
// 指数差 0 或 >=2 都正确：
//     0.6-0.9 (差 0) -> -0.3  对      1.5-6.0 (差 2) -> -4.5  对
// 触发条件在对阶路径上，用 a=1.5, b=1.5*2^k 定向扫描可复现。窗口只有一格宽，
// 少量代表点的测试很容易漏掉；atan2 的 swap 分支在 |t|>=0.5 时会受影响。
// 因此 a-b 一律降级为 a+(-b)，取反用 fsgnjn.s（纯位操作）。
// 临时寄存器用 f7：emitOp 浮点临时区（f1-f7）末位，生命期仅跨一条指令。
//
// POLARIS_UNSAFE_FSUB=1 会退回裸 fsub.s。这不是给生产用的开关，而是让
// 浮点用例生成器能验证自身的检出能力：一个抓不到已知缺陷的生成器，
// 通过了也说明不了任何事。默认（未设该变量）始终走安全路径。
static bool unsafeFSub() {
  static const bool v = [] {
    const char *e = getenv("POLARIS_UNSAFE_FSUB");
    return e && e[0] == '1';
  }();
  return v;
}

// 同上，用于复核 fmax.s 的原始行为（见 arith.maxnumf 处说明）
static bool unsafeFMax() {
  static const bool v = [] {
    const char *e = getenv("POLARIS_UNSAFE_FMAX");
    return e && e[0] == '1';
  }();
  return v;
}

// ---- store 后插入的分隔 nop 数 ----
// 规避 WarpLSU 地址/数据错位一拍（详见 tt.store 处注释）。
// 上板时建议 2：错位是"读到上一条的 warp_addr"，而 LSU 进入 Init_IDLE
// 前还要过 moqEnqueue，留两拍余量。
// ⚠ 插 nop 只能降低出错概率，**不能保证正确**：故障是非确定性的，同一
// 配置重复运行结果不同。根治需修 RTL（WarpLSU.scala 的 warp_addr 同周期
// 读写问题）。
//
// POLARIS_STORE_PAD=N 可不重编就覆盖（codegen 在 Python 进程内读环境变量）。
static int storePadNops() {
  static const int v = [] {
    if (const char *e = getenv("POLARIS_STORE_PAD")) {
      int n = atoi(e);
      if (n >= 0 && n <= 64) return n;
    }
    // 默认 0：模拟器上不存在该缺陷，插 nop 只会改变指令布局。
    // 上板时显式设置 POLARIS_STORE_PAD=2。
    return 0;
  }();
  return v;
}

static bool storePadDisabled() {
  static const bool v = [] {
    const char *e = getenv("POLARIS_NO_STORE_PAD");
    return e && e[0] == '1';
  }();
  return v;
}

template <typename BufT>
static void emitFSub(BufT &buf, uint32_t rd, uint32_t a, uint32_t b) {
  if (unsafeFSub()) {
    buf.emit(FSUB_S(rd, a, b));  // 已知有缺陷，仅用于生成器自检
    return;
  }
  buf.emit(FSGNJN_S(7, b, b));   // f7 = -b
  buf.emit(FADD_S(rd, a, 7));    // rd = a + (-b)
}

static uint32_t mapParamReg(uint32_t idx) {
  return kParamRegs[idx % kNumParamRegs];
}

//===----------------------------------------------------------------------===//
// 线性扫描寄存器分配 (LSRA)
//
// 为什么必须做真正的分配：朴素做法给每个 SSA 值一个编号，再用
// `pool[id % poolSize]` 环绕映射。SSA 值超过池子大小时，仍然活跃的值
// 会被覆盖，产出的 kernel 静默算错，编译期没有任何提示。
//
// 线性扫描按活跃区间回收寄存器：真正的需求是"最大同时活跃数"而非
// "SSA 值总数"，对这类算术密集的直线 kernel 通常只有个位数，
// 因此现有池子完全够用，不必 spill。
//===----------------------------------------------------------------------===//
struct RvInterval {
  int start;      // def 序号
  int end;        // last-use 序号（无使用者时等于 start）
  void *key;      // SSA 值的 opaque 指针
};

// 峰值同时活跃数 = 真正需要的寄存器数。
// 注意与"SSA 值总数"区分：线性扫描会回收已死值的寄存器，所以一个含 36 个
// SSA 值的 kernel 峰值可能只有 6 个，20 个寄存器的池子完全够。
static int livePeak(const std::vector<RvInterval> &ivs) {
  // 扫描线：+1 在 start，-1 在 end 之后
  std::vector<std::pair<int, int>> ev;
  ev.reserve(ivs.size() * 2);
  for (const auto &iv : ivs) {
    ev.push_back({iv.start, +1});
    ev.push_back({iv.end + 1, -1});
  }
  std::sort(ev.begin(), ev.end());
  int cur = 0, peak = 0;
  for (const auto &e : ev) {
    cur += e.second;
    if (cur > peak) peak = cur;
  }
  return peak;
}

// 单池线性扫描。返回 false 表示池子耗尽（调用方应报错而不是静默产错码）。
static bool linearScanAssign(std::vector<RvInterval> ivs,
                             const std::vector<uint32_t> &pool,
                             std::map<void *, uint32_t> &out) {
  std::sort(ivs.begin(), ivs.end(),
            [](const RvInterval &a, const RvInterval &b) {
              return a.start < b.start;
            });
  // freeRegs 用尾部 pop，故按逆序压入以优先取小号（便于阅读产出的汇编）
  std::vector<uint32_t> freeRegs(pool.rbegin(), pool.rend());
  struct Act {
    int end;
    uint32_t reg;
  };
  std::vector<Act> active;
  bool ok = true;

  for (const auto &iv : ivs) {
    // 1) 回收 end < 当前 start 的活跃项
    std::vector<Act> keep;
    keep.reserve(active.size());
    for (const auto &a : active) {
      if (a.end < iv.start)
        freeRegs.push_back(a.reg);
      else
        keep.push_back(a);
    }
    active.swap(keep);

    // 2) 分配
    if (freeRegs.empty()) {
      ok = false;              // 池子耗尽，交由调用方处理
      out[iv.key] = pool.empty() ? 0u : pool.front();
      continue;
    }
    uint32_t r = freeRegs.back();
    freeRegs.pop_back();
    out[iv.key] = r;
    active.push_back({iv.end, r});
  }
  return ok;
}

// 浮点寄存器分区（供线性扫描使用）：
//   f0      不使用（该核 f0 作运算目的时读回为 NaN，见文件末的 f0 检查）
//   f1-f7   emitOp 内部临时
//   f8-f27  SSA 值分配池
//   f28-f31 常量池 FC_ZERO/FC_ONE/FC_HALF/FC_MAGIC（见 CONST_BASE 说明）

// 把一个 MLIR op 发射为真 RISC-V 指令。返回 false 表示该 op 不产码。
static bool emitOp(StringRef name, RvBuffer &buf, uint32_t rd, uint32_t rs1,
                   uint32_t rs2, int32_t immOrOff, bool hasImm,
                   bool isFloat = false) {
  // ---- 整数算术 ----
  if (name == "arith.addi") {
    buf.emit(hasImm ? ADDI(rd, rs1, immOrOff) : ADD(rd, rs1, rs2));
    return true;
  }
  if (name == "arith.subi") { buf.emit(SUB(rd, rs1, rs2)); return true; }
  if (name == "arith.muli") { buf.emit(MUL(rd, rs1, rs2)); return true; }
  // ---- 单精度浮点 (RVF.scala:48-52) ----
  if (name == "arith.addf") { buf.emit(FADD_S(rd, rs1, rs2)); return true; }

  if (name == "arith.subf") { emitFSub(buf, rd, rs1, rs2); return true; }
  if (name == "arith.mulf") { buf.emit(FMUL_S(rd, rs1, rs2)); return true; }
  if (name == "arith.divf") { buf.emit(FDIV_S(rd, rs1, rs2)); return true; }
  if (name == "math.sqrt")  { buf.emit(FSQRT_S(rd, rs1)); return true; }
  if (name == "arith.maxnumf" || name == "arith.maximumf") {
    // 该核 fmax.s 结果错误（max(2.0,3.0) 给 2.0），fmin.s 正常。
    // 故用恒等式绕开：max(a,b) = -min(-a,-b)
    //
    // POLARIS_UNSAFE_FMAX=1 退回裸 fmax.s。与 POLARIS_UNSAFE_FSUB 同性质：
    // 不是给生产用的开关，而是用于独立复核硬件 fmax.s 本身的行为 ——
    // 绕行生效时测到的是"绕法正确"，不是"硬件指令正确"，两者必须分开。
    if (unsafeFMax()) { buf.emit(FMAX_S(rd, rs1, rs2)); return true; }
    // 临时用 f1/f2：f28/f29 是常量池 FC_ZERO/FC_ONE，写它们会破坏后续
    // floor/exp 等依赖的常量。
    buf.emit(FSGNJN_S(1, rs1, rs1));    // f1 = -a
    buf.emit(FSGNJN_S(2, rs2, rs2));    // f2 = -b
    buf.emit(FMIN_S(1, 1, 2));          // f1 = min(-a,-b)
    buf.emit(FSGNJN_S(rd, 1, 1));       // rd = -min(-a,-b) = max(a,b)
    return true;
  }
  if (name == "arith.minnumf" || name == "arith.minimumf") {
    buf.emit(FMIN_S(rd, rs1, rs2)); return true;
  }
  // ---- math.exp / math.exp2：区间归约实现，见函数内说明 ----
  if (name == "math.exp" || name == "math.exp2") {
    // 区间归约：exp(x) = 2^k * exp(r)，|r| <= ln2/2；exp2 取 r = (x-k)*ln2。
    // 本核无超越函数指令，按本核缺陷设计：
    //  * k：t + 1.5*2^23 就近舍入后用 fmv.x.w 读出位模式，一条整数减法得 k。
    //  * fmul 下溢不归零：t < -125 直接置 0（FTZ），保证乘积恒为正规数。
    //  * fadd 遇 inf/NaN 返回 0：溢出 / NaN / 过小三种情形最后用整数
    //    按位合并覆盖，中间结果不影响输出。
    //  * 2^k 拆成两个因子相乘，各自指数都在正规范围；真值溢出时 fmul 得 inf。
    // 多项式为 Cephes expf；exp 用 ln2 高低两段（Cody-Waite）保精度。
    // 正常范围内误差不超过 1 ULP；结果低于 2^-125 时置 0。
    // 临时：f1-f7（不用 f0），整数 x15-x17、x30、x31。
    const bool e2 = name == "math.exp2";
    const uint32_t XA = 15, XB = 16, XC = 17, XK = 30, XT = 31;
    auto LC = [&](uint32_t fr, uint32_t bits) {
      std::vector<uint32_t> v; LI32(v, XT, bits); buf.emitAll(v);
      buf.emit(FMV_W_X(fr, XT));
    };
    auto LIX = [&](uint32_t xr, uint32_t bits) {
      std::vector<uint32_t> v; LI32(v, xr, bits); buf.emitAll(v);
    };
    if (e2) buf.emit(FSGNJ_S(1, rs1, rs1));                 // t = x
    else { LC(2, 0x3fb8aa3b); buf.emit(FMUL_S(1, rs1, 2)); } // t = x*log2e
    buf.emit(FEQ_S(XA, rs1, rs1));                          // XA = !isnan(x)
    LC(2, 0x43000000); buf.emit(FLT_S(XB, 2, 1));           // XB = t > 128
    LC(2, 0xc2fa0000); buf.emit(FLT_S(XC, 1, 2));           // XC = t < -125
    LC(2, 0x4b400000);                                      // 1.5*2^23
    buf.emit(FADD_S(3, 1, 2));
    buf.emit(FMV_X_W(XK, 3));
    LIX(XT, 0x4b400000);
    buf.emit(SUB(XK, XK, XT));                              // k（整数）
    emitFSub(buf, 3, 3, 2);                                 // kf = round(t)
    if (e2) {
      emitFSub(buf, 6, 1, 3);                               // t - kf
      LC(4, 0x3f317218); buf.emit(FMUL_S(6, 6, 4));         // r = (t-kf)*ln2
    } else {
      LC(4, 0x3f317200); buf.emit(FMUL_S(5, 3, 4));         // kf*ln2_hi
      emitFSub(buf, 6, rs1, 5);
      LC(4, 0x35bfbe8e); buf.emit(FMUL_S(5, 3, 4));         // kf*ln2_lo
      emitFSub(buf, 6, 6, 5);                               // r
    }
    static const uint32_t P[] = {0x39506967, 0x3ab743ce, 0x3c088908,
                                 0x3d2aa9c1, 0x3e2aaaaa, 0x3f000000};
    LC(1, P[0]);
    for (int i = 1; i < 6; ++i) {
      buf.emit(FMUL_S(1, 1, 6)); LC(4, P[i]); buf.emit(FADD_S(1, 1, 4));
    }
    buf.emit(FMUL_S(2, 6, 6));                              // r^2
    buf.emit(FMUL_S(1, 1, 2));
    buf.emit(FADD_S(1, 1, 6));
    buf.emit(FADD_S(1, 1, FC_ONE));                         // exp(r)
    buf.emit(I(0x401, XK, 0x5, XT, OPC_OPIMM));             // k1 = k >> 1 (srai)
    buf.emit(SUB(XK, XK, XT));                              // k2 = k - k1
    buf.emit(ADDI(XT, XT, 127)); buf.emit(SLLI(XT, XT, 23)); buf.emit(FMV_W_X(2, XT));
    buf.emit(ADDI(XK, XK, 127)); buf.emit(SLLI(XK, XK, 23)); buf.emit(FMV_W_X(3, XK));
    buf.emit(FMUL_S(1, 1, 2)); buf.emit(FMUL_S(1, 1, 3));   // * 2^k1 * 2^k2
    buf.emit(FMV_X_W(XK, 1));
    buf.emit(ADDI(XC, XC, -1)); buf.emit(AND(XK, XK, XC));  // t<-125 -> 0
    buf.emit(ADDI(XT, XB, -1)); buf.emit(AND(XK, XK, XT));  // t>128  -> inf
    buf.emit(SUB(XB, X0, XB)); LIX(XT, 0x7f800000);
    buf.emit(AND(XB, XB, XT)); buf.emit(OR(XK, XK, XB));
    buf.emit(SUB(XT, X0, XA)); buf.emit(AND(XK, XK, XT));   // NaN -> qNaN
    buf.emit(ADDI(XA, XA, -1)); LIX(XT, 0x7fc00000);
    buf.emit(AND(XA, XA, XT)); buf.emit(OR(XK, XK, XA));
    buf.emit(FMV_W_X(rd, XK));
    return true;
  }
  // ---- 常量 ----
  // 浮点常量: RV 没有浮点立即数, 需先把 bit 模式装进整数寄存器再 fmv.w.x
  if (name == "arith.constant" && isFloat) {
    std::vector<uint32_t> tmp;
    LI32(tmp, 31, (uint32_t)immOrOff);   // x31 作临时, 装 f32 的 bit 模式
    buf.emitAll(tmp);
    buf.emit(FMV_W_X(rd, 31));
    return true;
  }
  if (name == "arith.constant") {
    std::vector<uint32_t> tmp;
    LI32(tmp, rd, (uint32_t)immOrOff);
    buf.emitAll(tmp);
    return true;
  }
  // ---- 访存: 地址在 rs1(operand0), store 的值在 rs2(operand1) ----
  // PolarisGPU 的 SIMT 是"一条指令自动作用于 SIMT_Width 条 lane"
  // (ALU.scala:222 VecInit(0 until SIMT_Width)), 故 tensor<8xf32> 天然对应
  // 一条标量指令 —— 各 lane 用自己的地址访存, 不需要循环展开。
  if (name == "tt.load") {
    buf.emit(isFloat ? FLW(rd, rs1, immOrOff) : LW(rd, rs1, immOrOff));
    return true;
  }
  if (name == "tt.store") {
    buf.emit(isFloat ? FSW(rs2, rs1, immOrOff) : SW(rs2, rs1, immOrOff));
    // ---- store 后插分隔 nop：规避 WarpLSU 的地址/数据错位一拍缺陷 ----
    // WarpLSU.scala:240-252 在 Init_IDLE 里对寄存器 warp_addr 写入后
    // **同周期**就读它送进 qscInput，取到的是上一条 store 的地址，
    // 而数据取的是本条的 wdata —— 地址与数据错位一拍。
    // 症状是"某个 store 写入了另一个 store 该写的数据"（数值本身对，
    // 只是落错地址），且随指令布局变化。模拟器测不出来：其读回通路
    // SIMD_WBU 在 LSU 之前。
    //
    // 拉开相邻 store 的发射间隔可让错位的那一拍落到 nop 上。
    // 写 x0 无副作用、不占寄存器。nop 数由 storePadNops() 给出（默认 0），
    // POLARIS_NO_STORE_PAD=1 强制关闭。
    if (!storePadDisabled()) {
      for (int k = 0, n = storePadNops(); k < n; ++k) buf.emit(ADDI(0, 0, 0));
    }
    return true;
  }
  // ---- 整数/浮点互转 ----
  // i32/u32 <-> f32 在遍历阶段处理（见"i32 <-> f32 转换"）。其余位宽
  // 到达这里时不支持，返回 false 让 codegen 计入 num_skipped 并编译失败。
  if (name == "arith.sitofp" || name == "arith.fptosi") return false;

  // ---- 浮点取反: fsgnjn.s rd,rs,rs 即 fneg.s ----
  if (name == "arith.negf") { buf.emit(FSGNJN_S(rd, rs1, rs1)); return true; }
  // ---- 浮点绝对值: fsgnjx.s rd,rs,rs 即 fabs.s (RVF.scala:55) ----
  if (name == "math.absf") { buf.emit(FSGNJX_S(rd, rs1, rs1)); return true; }
  // ---- 整数位运算 (掩码算术用) ----
  if (name == "arith.andi") { buf.emit(AND(rd, rs1, rs2)); return true; }
  // ---- 取余（i32 已在遍历阶段用 W 指令处理，这里处理其余位宽）----
  // 无符号 x % 2^k == x & (2^k-1)；有符号取余对负数不能用位与，一律用 rem。
  if (name == "arith.remsi" || name == "arith.remui") {
    if (name == "arith.remui" && hasImm && immOrOff > 0 &&
        (immOrOff & (immOrOff - 1)) == 0) {
      // 模数是 2 的幂：x & (mod-1)
      std::vector<uint32_t> tmp;
      LI32(tmp, 15, (uint32_t)(immOrOff - 1));
      buf.emitAll(tmp);
      buf.emit(AND(rd, rs1, 15));
      return true;
    }
    buf.emit(R(0x01, rs2, rs1, (name == "arith.remsi") ? 0x6 : 0x7, rd, OPC_OP));
    return true;
  }
  if (name == "arith.ori")  { buf.emit(OR(rd, rs1, rs2)); return true; }
  if (name == "arith.xori") { buf.emit(XOR(rd, rs1, rs2)); return true; }
  // ---- 整数比较: immOrOff 传 predicate (0=eq 1=ne 2=slt 3=sle 4=sgt 5=sge) ----
  if (name == "arith.cmpi") {
    switch (immOrOff) {
    case 0:   // eq  -> (a^b)==0 : xor + sltiu rd,rd,1
      buf.emit(XOR(rd, rs1, rs2));
      buf.emit(I(1, rd, 0x3, rd, OPC_OPIMM));      // sltiu rd,rd,1
      return true;
    case 1:   // ne  -> (a^b)!=0 : xor + sltu x0,rd
      buf.emit(XOR(rd, rs1, rs2));
      buf.emit(R(0x00, rd, X0, 0x3, rd, OPC_OP));  // sltu rd,x0,rd
      return true;
    case 2: buf.emit(SLT(rd, rs1, rs2)); return true;              // slt
    case 4: buf.emit(SLT(rd, rs2, rs1)); return true;              // sgt
    case 3:   // sle a<=b  ->  !(b<a)
      buf.emit(SLT(rd, rs2, rs1));
      buf.emit(I(1, rd, 0x4, rd, OPC_OPIMM));      // xori rd,rd,1
      return true;
    case 5:   // sge a>=b  ->  !(a<b)
      buf.emit(SLT(rd, rs1, rs2));
      buf.emit(I(1, rd, 0x4, rd, OPC_OPIMM));
      return true;
    default: buf.emit(SLT(rd, rs1, rs2)); return true;
    }
  }
  // ---- math.floor: RV 无 floor 指令。用 fcvt 往返 + 修正实现:
  //      t = (float)(int)x;  floor = (x < t) ? t-1 : t
  //      x 为负且有小数时 fcvt.w.s 向零取整会偏大, 故需减 1 修正。
  if (name == "math.floor") {
    // r = copysign((|x| + 2^23) - 2^23, x)；floor = r - [x < r]
    // 必须对 |x| 取整：x 为负时 x+2^23 落在 [2^22,2^23)，ulp=0.5，得不到
    // 整数；|x|+2^23 落在 [2^23,2^24)，ulp=1，就近舍入即得整数。
    // |x| >= 2^23 或 inf/NaN 时 x 无需取整，按位合并原样返回（同时避开
    // fadd.s 遇 inf/NaN 得 0 的缺陷）。结果符号随 x，floor(-0.0) = -0.0。
    // 取掩码 m 之后不能再调用 emitFSub：它以 f7 作临时，会覆盖 m。
    // 常量来自只读常量池（FC_MAGIC=2^23, FC_ONE=1.0, 见 CONST_BASE）。
    const uint32_t FR = 5, FD = 6, FM = 7, IT0 = 15, IT1 = 16, IT2 = 17;
    buf.emit(FSGNJX_S(FR, rs1, rs1));       // |x|
    buf.emit(FADD_S(FR, FR, FC_MAGIC));
    emitFSub(buf, FR, FR, FC_MAGIC);        // round(|x|)
    buf.emit(FSGNJ_S(FR, FR, rs1));         // r = copysign(round(|x|), x)
    buf.emit(FLT_S(IT0, rs1, FR));          // cond = (x < r)
    {
      std::vector<uint32_t> cp;
      LI32(cp, IT1, CONST_BASE);
      buf.emitAll(cp);
      buf.emit(SLLI(IT0, IT0, 2));
      buf.emit(ADD(IT1, IT1, IT0));
      buf.emit(FLW(FM, IT1, 0));            // m = cond ? 1.0 : 0.0
    }
    buf.emit(FSGNJN_S(FM, FM, FM));         // -m
    buf.emit(FADD_S(FD, FR, FM));           // FD = r - m
    buf.emit(FSGNJ_S(FD, FD, rs1));         // 符号跟随 x
    // small = |x| < 2^23（inf/NaN/大数为假）；按位合并：small ? FD : x
    buf.emit(FSGNJX_S(FM, rs1, rs1));       // |x|
    buf.emit(FLT_S(IT0, FM, FC_MAGIC));     // small
    buf.emit(FMV_X_W(IT1, FD));
    buf.emit(FMV_X_W(IT2, rs1));
    buf.emit(SUB(IT0, X0, IT0));            // mask = -small
    buf.emit(AND(IT1, IT1, IT0));
    buf.emit(I(0xFFF, IT0, 0x4, IT0, OPC_OPIMM));  // ~mask（xori -1）
    buf.emit(AND(IT2, IT2, IT0));
    buf.emit(OR(IT1, IT1, IT2));
    buf.emit(FMV_W_X(rd, IT1));
    return true;
  }
  // ---- 浮点比较: 结果是整数 0/1, 写整数寄存器 (RVF.scala:61-63) ----
  // immOrOff 复用为 predicate 编码: 0=olt 1=ole 2=oeq 3=ogt 4=oge
  if (name == "arith.cmpf") {
    switch (immOrOff) {
    case 0: buf.emit(FLT_S(rd, rs1, rs2)); return true;   // rs1 <  rs2
    case 1: buf.emit(FLE_S(rd, rs1, rs2)); return true;   // rs1 <= rs2
    case 2: buf.emit(FEQ_S(rd, rs1, rs2)); return true;   // rs1 == rs2
    case 3: buf.emit(FLT_S(rd, rs2, rs1)); return true;   // rs1 >  rs2
    case 4: buf.emit(FLE_S(rd, rs2, rs1)); return true;   // rs1 >= rs2
    default: buf.emit(FLT_S(rd, rs1, rs2)); return true;
    }
  }
  // ---- SIMT: program_id 由 WARP_ID CSR 读取 (SIMD_CSR.scala:119) ----
  if (name == "tt.get_program_id") {
    buf.emit(CSRR(rd, CSR_WARP_ID));
    return true;
  }
  if (name == "tt.get_num_programs") {
    buf.emit(CSRR(rd, CSR_THREAD_NUM));
    return true;
  }
  // ---- barrier -> SYN_WARP (SIMD_CSR.scala:793; 仅 SM 内有效) ----
  if (name == "gpu.barrier") { buf.emit(SYN_WARP(0)); return true; }
  // ---- tt.make_range -> TID: 各 lane 得到自己的序号 ----
  // 这是 tensor 语义落到 SIMT 的关键: tensor<8xi32> 的 [0,1,..7] 正是 8 条 lane
  // 各自的 TID 值 (ALU.scala:246 aluRes(i):=i)。start 非 0 时再加偏移。
  if (name == "tt.make_range") {
    buf.emit(TID(rd));
    if (hasImm && immOrOff != 0) buf.emit(ADDI(rd, rd, immOrOff));
    return true;
  }
  // splat/broadcast/expand_dims: SIMT 下标量寄存器对各 lane 天然可见,
  // 只需 mv。
  //
  // tt.expand_dims 在 SIMT 模型下是纯类型变换（tensor<8xi32> ->
  // tensor<8x1xi32>），不改变任何 lane 里的实际数值 —— 每个 lane 依旧
  // 持有自己那一个元素。真正把 2D 索引变成地址的是后续的 tt.broadcast
  // 与 tt.addptr。所以这里与 broadcast 同样处理即可。
  if (name == "tt.splat" || name == "tt.broadcast" ||
      name == "tt.expand_dims" || name == "tt.reshape" ||
      name == "tt.trans") {
    if (isFloat) buf.emit(FSGNJ_S(rd, rs1, rs1));   // fmv.s
    else buf.emit(ADDI(rd, rs1, 0));
    return true;
  }
  // ---- 指针算术: tt.addptr 的索引是元素数, 需按元素宽度左移 ----
  // Triton 语义: addptr(ptr, idx) = ptr + idx*sizeof(elem)，4 字节元素先 <<2。
  if (name == "tt.addptr") {
    if (immOrOff > 0) {           // immOrOff 复用为元素字节宽度
      int sh = (immOrOff == 8) ? 3 : (immOrOff == 4) ? 2
               : (immOrOff == 2) ? 1 : 0;
      if (sh) {
        // 用 x30(t5) 作临时。不能用 x31: 浮点常量装载已占它做 bit 模式暂存,
        // 两者混用会互相破坏。
        buf.emit(SLLI(30, rs2, sh));   // t5 = idx * elemsize
        buf.emit(ADD(rd, rs1, 30));
        return true;
      }
    }
    buf.emit(ADD(rd, rs1, rs2));
    return true;
  }
  return false;
}

struct PolarisRiscvCodegenPass
    : public PassWrapper<PolarisRiscvCodegenPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PolarisRiscvCodegenPass)

  StringRef getArgument() const final { return "polaris-riscv-codegen"; }
  StringRef getDescription() const final {
    return "Lower Triton IR to real RV64 + RV-SIMT machine code";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // ---- 先把 scf.for 完全展开 ----
    // 本后端不发分支/跳转，没有回边；不展开时循环体只会执行一次。
    // 因此循环边界必须是 tl.constexpr（编译期常量），在这里完全展开。
    //
    // 不用 mlir::loopUnrollByFactor：该 API 在本 LLVM 版本（10dc3a8e）会生成
    // 类型不一致的归纳变量运算（"'arith.muli' op requires the same type for
    // all operands and results"）。手写克隆 body 可精确控制类型。
    // 迭代式：每轮只找一个最内层循环展开，然后重新扫描。
    //
    // 不能一次性收集所有 ForOp 再逆序展开：展开外层时会把内层循环整体克隆
    // 若干份，这些克隆体不在预先收集的列表里；而列表里的原始内层句柄在
    // 外层 erase() 后已经失效，继续使用会触发 "dyn_cast on a non-existent
    // value" 断言。
    for (int round = 0; round < 256; ++round) {
      scf::ForOp target;
      mod.walk([&](scf::ForOp f) {
        if (target) return;
        // 只挑最内层：body 里不再含 ForOp
        bool hasInner = false;
        f.getBody()->walk([&](scf::ForOp) { hasInner = true; });
        if (!hasInner) target = f;
      });
      if (!target) break;                       // 没有可展开的了

      scf::ForOp f = target;
      auto cstOf = [](Operation *o, int64_t &v) -> bool {
        if (!o) return false;
        if (auto c = dyn_cast<arith::ConstantOp>(o))
          if (auto ia = dyn_cast<IntegerAttr>(c.getValue())) {
            v = ia.getInt();
            return true;
          }
        return false;
      };
      int64_t lo = 0, hi = 0, st = 0;
      bool boundsOk = cstOf(f.getLowerBound().getDefiningOp(), lo) &&
                      cstOf(f.getUpperBound().getDefiningOp(), hi) &&
                      cstOf(f.getStep().getDefiningOp(), st);
      int64_t trip = (boundsOk && st > 0 && hi > lo)
                         ? (hi - lo + st - 1) / st : 0;
      // 展开不了就必须跳出整个循环 —— 用 continue 会让下一轮重新选中
      // 同一个 ForOp，形成死循环。
      // 迭代次数上限 512：kernel 代码须放在参数区（0x80002000）之前，
      // 展开后过大的 kernel 会在末尾的代码长度检查处报错。
      if (trip < 1 || trip > 512) break;

      OpBuilder b(f);
      Location loc = f.getLoc();
      SmallVector<Value> carried(f.getInitArgs().begin(),
                                 f.getInitArgs().end());
      Block &body = f.getRegion().front();
      auto yield = dyn_cast<scf::YieldOp>(body.getTerminator());
      if (!yield) break;

      for (int64_t k = 0; k < trip; ++k) {
        IRMapping map;
        // 归纳变量替换为常量，类型跟原归纳变量一致（避免 arith 类型冲突）
        Value iv = b.create<arith::ConstantOp>(
            loc, b.getIntegerAttr(f.getInductionVar().getType(),
                                  lo + k * st));
        map.map(f.getInductionVar(), iv);
        for (auto [ra, cv] : llvm::zip(f.getRegionIterArgs(), carried))
          map.map(ra, cv);
        for (Operation &o : body.without_terminator())
          b.clone(o, map);
        SmallVector<Value> next;
        for (Value v : yield.getOperands())
          next.push_back(map.lookupOrDefault(v));
        carried = next;
      }
      // 循环结果替换为最后一次迭代的 carried 值，然后删掉原循环
      if (f.getNumResults() == (int64_t)carried.size())
        for (auto [res, v] : llvm::zip(f.getResults(), carried))
          res.replaceAllUsesWith(v);
      f.erase();
    }

    RvBuffer buf;
    int64_t nFunc = 0, nMapped = 0, nSkipped = 0, totPtrArgs = 0;
    // CTA 8 维特征向量的原始计数 (供 SNN 调度器)
    int64_t featMem = 0, featFp = 0, featInt = 0, featTc = 0;
    int64_t featSel = 0, featBar = 0, sharedBytes = 0;
    // select 展开占用的指令数：由编译器在生成时直接计数，导出为元数据
    int64_t selCount = 0, selInsts = 0;
    std::map<std::string, int> skippedNames;   // 被跳过的 op 名 -> 次数
    // 其中会影响数值正确性的那些（排除纯诊断类）。非空则编译失败，
    // 见下方 emitOp 分支处的说明。
    std::map<std::string, int> fatalSkipped;
    // 张量元素总数必须等于 SIMT 宽度 8：一个值占一个寄存器、每条 lane 一个元素，
    // tt.make_range 降级为单条 TID。其他形状会少算或越界写，记下违规形状，
    // 编译末尾统一报错。
    std::map<std::string, int> badShapes;
    // 其余"降级不了、降级了会静默算错"的构造：键为人读得懂的描述，
    // 编译末尾统一报错（与 badShapes / fatalSkipped 同一策略）。
    std::map<std::string, int> unsupported;
    // 读改写检测：当前 SoC 把同一组 CTA 同时派发给两个 SM（NutShell.scala
    // 顶层硬编码派发，mhartid/smId 恒为 0，软件无法区分 SM），对同一缓冲区的
    // 读改写会被执行两次。对同一指针参数既读又写的 kernel 拒绝。
    std::set<unsigned> loadArgs, storeArgs;
    std::map<std::string, int> badReduce;   // 跨 lane 归约（归约维长度 > 1）
    std::map<std::string, int> badMem;      // 非 32 位元素访存、带掩码访存
    int64_t maxRegUsed = 0;
    bool regExhausted = false;
    int64_t regPressureIntOut = 0, regPressureFltOut = 0;
    int64_t regPeakIntOut = 0, regPeakFltOut = 0;

    mod.walk([&](mlir::triton::FuncOp fn) {
      ++nFunc;

      // ---- SIMT prologue ----
      // 线程数 = warp 宽度 8 (NutCore.scala:96 SIMT_Width)
      std::vector<uint32_t> pro;
      LI32(pro, T0, 8);
      buf.emitAll(pro);
      buf.emit(CSRW(CSR_THREAD_NUM, T0));     // csrw 0xD, t0
      buf.emit(CSRW(CSR_THREAD_BASEID, X0));  // csrw 0xE, x0
      buf.emit(START_WARP());                 // 0x00C0007B
      buf.emit(TID(T1));                      // t1 = lane id

      // ---- 常量池：预载 0.0 / 1.0 / 0.5 / 2^23 到固定浮点寄存器 ----
      // floor、exp 等多个降级共用这些常量。host 把它们预置在镜像的只读
      // 常量区（CONST_BASE，由 run_triton_kernel.py 填充），prologue 用 flw
      // 一次载入 f28-f31，之后各处直接引用，不必在每个使用点重新物化。
      {
        std::vector<uint32_t> cb;
        LI32(cb, 28, CONST_BASE);
        buf.emitAll(cb);
        buf.emit(FLW(FC_ZERO, 28, 0));   // f28 = 0.0
        buf.emit(FLW(FC_ONE, 28, 4));    // f29 = 1.0
        buf.emit(FLW(FC_HALF, 28, 8));   // f30 = 0.5
        buf.emit(FLW(FC_MAGIC, 28, 12)); // f31 = 2^23 (floor 用)
      }

      // 各指针参数由下面的"参数裸机约定"统一装载（t2 会被参数分配覆盖）。

      // ---- 顺序发射函数体 ----
      // 逻辑寄存器按 SSA 值出现顺序分配, 再映射到真实 RISC-V 寄存器
      // 整数值与浮点值分别在两个寄存器堆里编号, 避免 x/f 混用导致编码错误
      // 判断某 SSA 值该放浮点寄存器堆还是整数寄存器堆。
      // 关键: 指针必须归整数堆。tensor<8x!tt.ptr<f32>> 的元素是"指向 f32 的指针",
      // 若只看最内层元素类型会误判为浮点, 把地址搬进浮点寄存器。
      auto valueIsFloat = [](Value v) -> bool {
        Type t = v.getType();
        if (auto rt = dyn_cast<RankedTensorType>(t)) t = rt.getElementType();
        if (isa<mlir::triton::PointerType>(t)) return false;  // 指针走整数堆
        return isa<FloatType>(t);
      };
      // 固定绑定表：指针参数占用 kParamRegs 中的专属寄存器，
      // 不进入中间值分配池，故不会被中间值覆盖。
      std::map<void *, uint32_t> pinnedReg;

      // ---- 预扫描：计算每个 SSA 值的活跃区间，再做线性扫描分配 ----
      // 必须先分配再发射：发射时才知道用哪个物理寄存器。
      std::map<void *, uint32_t> allocInt, allocFlt;
      bool regAllocOK = true;
      int regPressureInt = 0, regPressureFlt = 0;
      int regPeakInt = 0, regPeakFlt = 0;
      {
        // 给函数体内的 op 编号（walk 顺序与后面发射顺序一致）
        std::map<Operation *, int> opIdx;
        int nOps = 0;
        fn.walk([&](Operation *o) { opIdx[o] = nOps++; });

        // def 位置与 last-use 位置
        std::map<void *, int> defAt, lastUse;
        std::map<void *, bool> isFlt;
        auto note = [&](Value v, int idx, bool isDef) {
          void *k = v.getAsOpaquePointer();
          isFlt[k] = valueIsFloat(v);
          if (isDef) {
            if (!defAt.count(k)) defAt[k] = idx;
          }
          // 无论 def 还是 use 都更新 last-use，保证区间覆盖到最后一次引用
          auto it = lastUse.find(k);
          if (it == lastUse.end() || it->second < idx) lastUse[k] = idx;
        };
        fn.walk([&](Operation *o) {
          int idx = opIdx[o];
          for (Value res : o->getResults()) note(res, idx, /*isDef=*/true);
          for (Value opd : o->getOperands()) note(opd, idx, /*isDef=*/false);
        });

        // 上面按"操作数出现"统计 last-use 会漏掉使用者不在 opIdx 里的情况
        // （嵌套区域、展开残留等），漏算会让区间提前结束、寄存器被提前回收
        // 给别的值，部分 store 落到错误位置。
        // 这里直接顺 getUsers() 把区间补到最后一个使用者，再兜底拉到函数尾。
        fn.walk([&](Operation *o) {
          for (Value res : o->getResults()) {
            void *k = res.getAsOpaquePointer();
            for (Operation *u : res.getUsers()) {
              auto it = opIdx.find(u);
              int uidx = (it != opIdx.end()) ? it->second : nOps;
              if (!lastUse.count(k) || lastUse[k] < uidx) lastUse[k] = uidx;
            }
          }
        });

        // 函数参数在入口处定义（idx = -1），非指针参数才参与分配
        for (auto arg : fn.getArguments()) {
          if (isa<mlir::triton::PointerType>(arg.getType())) continue;
          void *k = arg.getAsOpaquePointer();
          isFlt[k] = valueIsFloat(arg);
          defAt[k] = -1;
          if (!lastUse.count(k)) lastUse[k] = -1;
        }

        std::vector<RvInterval> ivInt, ivFlt;
        for (auto &kv : defAt) {
          void *k = kv.first;
          // 指针参数已 pin，不进池
          int s = kv.second;
          int e = lastUse.count(k) ? lastUse[k] : s;
          RvInterval iv{s, e, k};
          if (isFlt[k])
            ivFlt.push_back(iv);
          else
            ivInt.push_back(iv);
        }
        regPressureInt = (int)ivInt.size();
        regPressureFlt = (int)ivFlt.size();
        // 峰值同时活跃数才是真正的寄存器需求（线性扫描会复用已死值的寄存器）
        regPeakInt = livePeak(ivInt);
        regPeakFlt = livePeak(ivFlt);

        std::vector<uint32_t> poolInt(kAllocRegs,
                                      kAllocRegs + kNumAllocRegs);
        // 浮点池：f8..f27（f0 不用，f1-f7 留作 emitOp 内部临时，
        // f28-f31 是常量池 FC_ZERO/FC_ONE/FC_HALF/FC_MAGIC）
        std::vector<uint32_t> poolFlt;
        for (uint32_t r = 8; r <= 27; ++r) poolFlt.push_back(r);

        bool okI = linearScanAssign(ivInt, poolInt, allocInt);
        bool okF = linearScanAssign(ivFlt, poolFlt, allocFlt);
        regAllocOK = okI && okF;

        // POLARIS_DUMP_REGALLOC=1: 打印整数堆的区间与分配结果，并自检
        // 是否存在"区间重叠却同寄存器"的冲突。寄存器错配的症状是数值
        // 静默变错（不是崩溃），靠读汇编逐条比对代价太高，需要机器自检。
        if (const char *dbg = getenv("POLARIS_DUMP_REGALLOC")) {
          if (dbg[0] == '1') {
            llvm::errs() << "[regalloc] int intervals (start,end)->reg\n";
            for (auto &iv : ivInt) {
              Value vv = Value::getFromOpaquePointer(iv.key);
              llvm::errs() << "  [" << iv.start << "," << iv.end << "] -> x"
                           << allocInt[iv.key] << "  ";
              if (Operation *d = vv.getDefiningOp())
                llvm::errs() << d->getName().getStringRef();
              else
                llvm::errs() << "<blockarg>";
              llvm::errs() << "\n";
            }
            llvm::errs() << "[regalloc] flt intervals (start,end)->reg\n";
            for (auto &iv : ivFlt) {
              Value vv = Value::getFromOpaquePointer(iv.key);
              llvm::errs() << "  [" << iv.start << "," << iv.end << "] -> f"
                           << allocFlt[iv.key] << "  ";
              if (Operation *d = vv.getDefiningOp())
                llvm::errs() << d->getName().getStringRef();
              else
                llvm::errs() << "<blockarg>";
              llvm::errs() << "\n";
            }
            auto scan = [&](const char *tag, std::vector<RvInterval> &ivs,
                            std::map<void *, uint32_t> &al, char pfx) {
              for (size_t i = 0; i < ivs.size(); ++i)
                for (size_t j = i + 1; j < ivs.size(); ++j) {
                  const auto &A = ivs[i], &B = ivs[j];
                  if (al[A.key] != al[B.key]) continue;
                  if (A.start <= B.end && B.start <= A.end)
                    llvm::errs() << "[regalloc] CONFLICT " << tag << " " << pfx
                                 << al[A.key] << ": [" << A.start << ","
                                 << A.end << "] vs [" << B.start << ","
                                 << B.end << "]\n";
                }
            };
            scan("int", ivInt, allocInt, 'x');
            scan("flt", ivFlt, allocFlt, 'f');
          }
        }
      }

      auto regOf = [&](Value v) -> uint32_t {
        void *k = v.getAsOpaquePointer();
        auto pit = pinnedReg.find(k);
        if (pit != pinnedReg.end()) return pit->second;
        bool flt = valueIsFloat(v);
        auto &tbl = flt ? allocFlt : allocInt;
        auto it = tbl.find(k);
        if (it != tbl.end()) return it->second;
        // 兜底：理论上预扫描已覆盖所有值；若走到这里说明有遗漏，
        // 用池首元素而不是静默环绕，便于暴露问题。
        return flt ? 8u : kAllocRegs[0];
      };
      // ---- 函数参数的裸机调用约定 ----
      // OpenBPU-3 无 OS/无 ABI, kernel 指针参数不能靠寄存器传入。
      // 约定: 第 i 个指针参数 = ARG_BASE + i*0x1000, host 侧按同一约定把
      // 数据写进镜像。这里在 prologue 里把这些常量地址装进对应寄存器。
      // (不做这一步, 参数寄存器为随机值, 访存会落到非法地址)
      uint32_t nPtrArgs = 0;
      {
        for (auto arg : fn.getArguments()) {
          if (isa<mlir::triton::PointerType>(arg.getType())) {
            // 指针参数占用**独立**的参数寄存器（kParamRegs），
            // 与中间值池子（kAllocRegs）不重叠，避免被覆盖。
            uint32_t r = mapParamReg(nPtrArgs);
            pinnedReg[arg.getAsOpaquePointer()] = r;
            std::vector<uint32_t> tmp;
            LI32(tmp, r, ARG_BASE + nPtrArgs * ARG_STRIDE);
            buf.emitAll(tmp);
            ++nPtrArgs;
          } else {
            (void)regOf(arg);
          }
        }
      }

      fn.walk([&](Operation *op) {
        StringRef nm = op->getName().getStringRef();
        for (Value res : op->getResults()) {
          if (auto rt = dyn_cast<RankedTensorType>(res.getType())) {
            int64_t n = 1;
            for (auto d : rt.getShape()) n *= d;
            if (n != 8) {
              std::string ts;
              llvm::raw_string_ostream os(ts);
              rt.print(os);
              ++badShapes[os.str()];
            }
          }
        }
        // ---- 无法正确降级的构造：一律拒绝 ----
        // tt.trans 降级为原样传递；只有至多一维长度 > 1 时这才等价于转置。
        if (auto tr = dyn_cast<mlir::triton::TransOp>(op)) {
          int nontriv = 0;
          if (auto in = dyn_cast<RankedTensorType>(tr->getOperand(0).getType()))
            for (auto d : in.getShape()) nontriv += d > 1 ? 1 : 0;
          if (nontriv > 1) ++unsupported["tt.trans that reorders lanes"];
        }
        // 只有 axis=0 映射到硬件（WARP_ID）；axis=1/2 没有对应的硬件来源，
        // 读 WARP_ID 会得到错误的值，故拒绝。
        if (auto p = dyn_cast<mlir::triton::GetProgramIdOp>(op))
          if (p.getAxisAsInt() != 0) ++unsupported["tl.program_id(axis != 0)"];
        if (auto p = dyn_cast<mlir::triton::GetNumProgramsOp>(op))
          if (p.getAxisAsInt() != 0) ++unsupported["tl.num_programs(axis != 0)"];
        // 记录 load / store 的地址追溯到哪个指针参数，末尾检查读改写。
        if (nm == "tt.load" || nm == "tt.store") {
          std::set<Operation *> seen;
          std::function<void(Value)> up = [&](Value v) {
            if (auto ba = dyn_cast<BlockArgument>(v)) {
              Type t = ba.getType();
              if (auto rt = dyn_cast<RankedTensorType>(t)) t = rt.getElementType();
              if (isa<mlir::triton::PointerType>(t) &&
                  isa<mlir::triton::FuncOp>(ba.getOwner()->getParentOp()))
                (nm == "tt.load" ? loadArgs : storeArgs).insert(ba.getArgNumber());
              return;
            }
            Operation *d = v.getDefiningOp();
            if (!d || !seen.insert(d).second) return;
            for (Value o : d->getOperands()) up(o);
          };
          if (op->getNumOperands() > 0) up(op->getOperand(0));
        }
        uint32_t rd = 0, rs1 = 0, rs2 = 0;
        int32_t imm = 0;
        bool hasImm = false;
        bool isFloat = false;

        if (op->getNumOperands() > 0) rs1 = regOf(op->getOperand(0));
        if (op->getNumOperands() > 1) rs2 = regOf(op->getOperand(1));
        if (op->getNumResults() > 0) rd = regOf(op->getResult(0));

        // 取整型常量属性作为立即数
        if (auto attr = op->getAttrOfType<IntegerAttr>("value")) {
          imm = (int32_t)attr.getInt();
          hasImm = true;
        }
        // 浮点常量: 取其 f32 bit 模式(而非数值), 供 LI32+FMV_W_X 装载
        if (auto fattr = op->getAttrOfType<FloatAttr>("value")) {
          float f = (float)fattr.getValueAsDouble();
          uint32_t bits;
          std::memcpy(&bits, &f, sizeof(bits));
          imm = (int32_t)bits;
          hasImm = true;
        }
        // tensor 常量: flagtree 的 canonicalizer 会把 splat 折叠成
        // arith.constant dense<...> : tensor<8xf32>, 属性类型是
        // DenseElementsAttr 而非 FloatAttr，需单独取出 splat 值。
        if (auto dattr = op->getAttrOfType<DenseElementsAttr>("value")) {
          Type et = dattr.getElementType();
          // 只有 splat（所有元素相同）能用一条标量常量表示；非 splat 常量需要
          // 按 lane 分别装载，尚未实现，故拒绝（否则 8 条 lane 都会得到首元素）。
          if (!dattr.isSplat()) {
            std::string ts;
            llvm::raw_string_ostream os(ts);
            dattr.getType().print(os);
            ++unsupported["non-splat constant " + os.str()];
          }
          if (isa<FloatType>(et)) {
            APFloat af = *dattr.value_begin<APFloat>();
            bool losesInfo = false;
            af.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven,
                       &losesInfo);
            imm = (int32_t)(uint32_t)af.bitcastToAPInt().getZExtValue();
            hasImm = true;
            isFloat = true;
          } else if (isa<IntegerType>(et)) {
            imm = (int32_t)(*dattr.value_begin<APInt>()).getSExtValue();
            hasImm = true;
          }
        }
        // splat/constant 的 tensor<...xf32> 也算浮点
        if (nm == "arith.constant" && op->getNumResults() > 0) {
          Type t = op->getResult(0).getType();
          if (auto rt = dyn_cast<RankedTensorType>(t)) t = rt.getElementType();
          if (isa<FloatType>(t)) isFloat = true;
        }
        // splat/broadcast: 只有元素为浮点"且不是指针"时才走浮点搬运
        if ((nm == "tt.splat" || nm == "tt.broadcast") &&
            op->getNumResults() > 0) {
          Type t = op->getResult(0).getType();
          if (auto rt = dyn_cast<RankedTensorType>(t)) t = rt.getElementType();
          if (!isa<mlir::triton::PointerType>(t) && isa<FloatType>(t))
            isFloat = true;
        }
        // make_range 的 start 作为立即数基
        if (nm == "tt.make_range") {
          if (auto s = op->getAttrOfType<IntegerAttr>("start")) {
            imm = (int32_t)s.getInt();
            hasImm = true;
          }
        }
        // arith.remsi/remui: 若除数是常量且为 2 的幂, 把它经 imm 传给 emitOp,
        // 让其用 "x & (mod-1)" 代替除法(查表索引的模数都是 2 的幂)。
        if (nm == "arith.remsi" || nm == "arith.remui") {
          if (op->getNumOperands() > 1) {
            if (auto def = op->getOperand(1).getDefiningOp()) {
              int64_t mv = 0;
              bool got = false;
              if (auto ia = def->getAttrOfType<IntegerAttr>("value")) {
                mv = ia.getInt(); got = true;
              } else if (auto da =
                             def->getAttrOfType<DenseElementsAttr>("value")) {
                if (isa<IntegerType>(da.getElementType())) {
                  mv = (*da.value_begin<APInt>()).getSExtValue(); got = true;
                }
              }
              if (got && mv > 0) { imm = (int32_t)mv; hasImm = true; }
            }
          }
        }
        // arith.cmpi 的 predicate: MLIR CmpIPredicate
        //   0=eq 1=ne 2=slt 3=sle 4=sgt 5=sge 6=ult 7=ule 8=ugt 9=uge
        // 映射成 emitOp 认识的 0=eq 1=ne 2=slt 3=sle 4=sgt 5=sge
        if (nm == "arith.cmpi") {
          if (auto pa = op->getAttrOfType<IntegerAttr>("predicate")) {
            int pv = (int)pa.getInt();
            switch (pv) {
            case 0: imm = 0; break;               // eq
            case 1: imm = 1; break;               // ne
            case 2: case 6: imm = 2; break;       // slt / ult
            case 3: case 7: imm = 3; break;       // sle / ule
            case 4: case 8: imm = 4; break;       // sgt / ugt
            case 5: case 9: imm = 5; break;       // sge / uge
            default: imm = 2; break;
            }
            hasImm = true;
          }
        }
        // arith.cmpf 的 predicate 映射成 emitOp 内部编码
        // (0=olt 1=ole 2=oeq 3=ogt 4=oge)。实际发射由下方"16 个 MLIR 谓词
        // 全部显式映射"一段完成，这里只为 emitOp 的 cmpf 分支提供回退值。
        if (nm == "arith.cmpf") {
          if (auto pa = op->getAttrOfType<IntegerAttr>("predicate")) {
            switch ((int)pa.getInt()) {
            case 1: imm = 2; break;   // OEQ
            case 2: imm = 3; break;   // OGT
            case 3: imm = 4; break;   // OGE
            case 4: imm = 0; break;   // OLT
            case 5: imm = 1; break;   // OLE
            default: imm = 0; break;
            }
            hasImm = true;
          }
        }
        // tt.addptr: 把指向元素的字节宽度经 imm 传给 emitOp, 供其左移缩放。
        // Triton 的 addptr 索引是"元素个数"而非字节, 不缩放会算错地址。
        if (nm == "tt.addptr" && op->getNumOperands() > 0) {
          Type pt = op->getOperand(0).getType();
          if (auto rt = dyn_cast<RankedTensorType>(pt)) pt = rt.getElementType();
          if (auto p = dyn_cast<mlir::triton::PointerType>(pt)) {
            Type et = p.getPointeeType();
            unsigned bits = 32;
            if (auto it = dyn_cast<IntegerType>(et)) bits = it.getWidth();
            else if (auto ft = dyn_cast<FloatType>(et)) bits = ft.getWidth();
            imm = (int32_t)(bits / 8);
            hasImm = true;
          }
        }

        // 判断该访存/运算的元素类型是否为浮点
        auto elemIsFloat = [](Type t) -> bool {
          if (auto rt = dyn_cast<RankedTensorType>(t)) t = rt.getElementType();
          if (auto pt = dyn_cast<mlir::triton::PointerType>(t))
            t = pt.getPointeeType();
          if (auto rt2 = dyn_cast<RankedTensorType>(t)) t = rt2.getElementType();
          return isa<FloatType>(t);
        };
        // 访存约束：元素必须为 32 位（降级恒用 LW/SW/FLW/FSW，不看位宽），
        // 且暂不支持掩码。两者都会静默访问错误的字节或 lane，故拒绝。
        if ((nm == "tt.load" && op->getNumResults() > 0) ||
            (nm == "tt.store" && op->getNumOperands() > 1)) {
          Type vt = nm == "tt.load" ? op->getResult(0).getType()
                                    : op->getOperand(1).getType();
          if (auto rt = dyn_cast<RankedTensorType>(vt)) vt = rt.getElementType();
          unsigned bw = vt.isIntOrFloat() ? vt.getIntOrFloatBitWidth() : 0;
          if (bw != 32) {
            std::string ts;
            llvm::raw_string_ostream os(ts);
            vt.print(os);
            ++badMem[nm.str() + " of " + os.str()];
          }
          unsigned maskIdx = nm == "tt.load" ? 1 : 2;
          if (op->getNumOperands() > maskIdx) ++badMem[nm.str() + " with mask"];
        }
        if (nm == "tt.load" && op->getNumResults() > 0)
          isFloat = elemIsFloat(op->getResult(0).getType());
        else if (nm == "tt.store" && op->getNumOperands() > 1)
          isFloat = elemIsFloat(op->getOperand(1).getType());
        // 其余 op 按结果类型兜底标记浮点。必须排除 addptr/splat/broadcast:
        // 它们的结果是指针 tensor, 误判成浮点会让地址进浮点寄存器堆。
        else if (op->getNumResults() > 0 && !isFloat &&
                 nm != "tt.addptr" && nm != "tt.splat" &&
                 nm != "tt.broadcast" && nm != "tt.make_range") {
          Type t = op->getResult(0).getType();
          if (auto rt = dyn_cast<RankedTensorType>(t)) t = rt.getElementType();
          if (!isa<mlir::triton::PointerType>(t))
            isFloat = elemIsFloat(op->getResult(0).getType());
        }

        // ---- tt.reduce: 跨 lane 归约 ----
        // SIMT 下 tensor<8xf32> 的 8 个元素分布在 8 条 lane 上，归约需要
        // lane 间通信，而本核没有 lane 间 shuffle 指令。本后端只支持被归约
        // 维长度为 1 的情形（降级为恒等传递，rd = rs1），其余情形在下面
        // 检查后拒绝。kernel 侧需要跨 lane 归约时，应改写为显式循环。
        if (nm == "tt.reduce" && op->getNumOperands() >= 1 &&
            op->getNumResults() >= 1) {
          // 恒等传递只在被归约的那一维长度为 1 时成立（每条 lane 本就只有
          // 自己那一个元素）。归约维长度 > 1 即为跨 lane 归约，本核无 lane 间
          // 通路，恒等传递会只得到本 lane 的值，故拒绝。
          {
            bool okAxis = false;
            if (auto in = dyn_cast<RankedTensorType>(op->getOperand(0).getType()))
              if (auto ax = op->getAttrOfType<IntegerAttr>("axis")) {
                int64_t a = ax.getInt();
                if (a >= 0 && a < (int64_t)in.getRank() && in.getShape()[a] == 1)
                  okAxis = true;
              }
            if (!okAxis) {
              std::string ts;
              llvm::raw_string_ostream os(ts);
              op->getOperand(0).getType().print(os);
              ++badReduce[os.str()];
            }
          }
          uint32_t rsr = regOf(op->getOperand(0));
          uint32_t rdr = regOf(op->getResult(0));
          bool fred = false;
          {
            Type t = op->getResult(0).getType();
            if (auto rtn = dyn_cast<RankedTensorType>(t))
              t = rtn.getElementType();
            fred = isa<FloatType>(t);
          }
          if (fred) buf.emit(FSGNJ_S(rdr, rsr, rsr));   // fmv.s rd, rs
          else buf.emit(ADDI(rdr, rsr, 0));
          ++nMapped;
          return;
        }
        // tt.reduce.return 是 region 的终结符，本身不产码（值已在 reduce
        // 那步处理）。显式识别以免计入 num_skipped 造成误报。
        if (nm == "tt.reduce.return") { ++nMapped; return; }

        // arith.select 有 3 个操作数(cond, tval, fval), 通用框架只取前 2 个,
        // 故在此单独发射。用掩码算术实现无分支选择:
        //   mask = -cond            (cond 为 0/1 -> 全0/全1)
        //   out  = (t & mask) | (f & ~mask)
        // 浮点值需先搬到整数寄存器做位运算, 再搬回 —— SIMT 下这样比条件分支
        // 更稳(避免 warp 内 lane 分歧, 该核 SYNWARP 仅 SM 内有效)。
        if (nm == "arith.select" && op->getNumOperands() == 3) {
          ++featSel;   // 分支/选择密度(CTA 特征向量第 5 维)
          const size_t selStart = buf.size();
          uint32_t rc = regOf(op->getOperand(0));   // cond (整数堆)
          uint32_t rt = regOf(op->getOperand(1));   // true 值
          uint32_t rf = regOf(op->getOperand(2));   // false 值
          bool fsel = false;
          {
            Type t = op->getResult(0).getType();
            if (auto rtn = dyn_cast<RankedTensorType>(t)) t = rtn.getElementType();
            fsel = isa<FloatType>(t) &&
                   !isa<mlir::triton::PointerType>(t);
          }
          // 临时寄存器：整数 t0/t1/t2 语义已被 prologue 占用，这里用
          // x15..x17（a5..a7）。不使用浮点临时：f5/f6/f7 是 floor 与 emitFSub
          // 的临时区，select 的常见前驱恰好是 cmpf/abs/floor，共用会互相覆盖。
          const uint32_t IT0 = 15, IT1 = 16;
          if (fsel) {
            // 浮点 select：按位合并，逐位精确（含 -0、inf、NaN）。
            //   fmv.x.w 取位模式 -> 整数掩码 and/or -> fmv.w.x
            // 不经过浮点运算，fadd/fsub 的缺陷影响不到它；不使用浮点临时，
            // 也不涉及 f0（f0 的缺陷见文件末的 f0 检查）。
            const uint32_t IT2 = 17;             // a7，emitOp 临时
            buf.emit(FMV_X_W(IT0, rt));          // t 的位模式
            buf.emit(FMV_X_W(IT1, rf));          // f 的位模式
            buf.emit(SUB(IT2, X0, rc));          // mask = -cond（全 1 / 全 0）
            buf.emit(AND(IT0, IT0, IT2));
            buf.emit(ADDI(IT2, rc, -1));         // ~mask = cond - 1
            buf.emit(AND(IT1, IT1, IT2));
            buf.emit(OR(IT0, IT0, IT1));
            buf.emit(FMV_W_X(rd, IT0));          // out
          } else {
            // 整数 select：掩码算术即可
            buf.emit(SUB(IT0, X0, rc));          // mask = -cond
            buf.emit(ADDI(IT1, X0, -1));
            buf.emit(XOR(IT1, IT0, IT1));        // ~mask
            buf.emit(AND(IT0, rt, IT0));
            buf.emit(AND(IT1, rf, IT1));
            buf.emit(OR(rd, IT0, IT1));
          }
          ++selCount;
          selInsts += (int64_t)(buf.size() - selStart);
          ++nMapped;
          return;
        }


        // ---- 统计 CTA 8 维特征向量的原始计数 ----
        if (nm == "tt.load" || nm == "tt.store" || nm == "tt.atomic_rmw")
          ++featMem;
        else if (nm.starts_with("arith.") &&
                 (nm.ends_with("f") || nm.contains("numf")))
          ++featFp;
        else if (nm.starts_with("arith.") && nm.ends_with("i"))
          ++featInt;
        if (nm == "arith.cmpf") ++featSel;
        if (nm == "tt.dot") ++featTc;
        if (nm == "gpu.barrier") ++featBar;
        // tl.shared 的分配: 当前 RTL 的 SM 内无 shared memory 实体,
        // 只有调度器记账。这里累计请求字节数, 由 CTA 包的 sharedMemSize 携带,
        // 调度器据此做 availSharedMem 检查(sche.scala:208)。
        if (nm.contains("local_alloc") || nm.contains("alloc_tensor") ||
            nm.contains("shared")) {
          if (op->getNumResults() > 0) {
            Type t = op->getResult(0).getType();
            if (auto rt = dyn_cast<RankedTensorType>(t)) {
              unsigned bits = 32;
              Type et = rt.getElementType();
              if (auto it = dyn_cast<IntegerType>(et)) bits = it.getWidth();
              else if (auto ft = dyn_cast<FloatType>(et)) bits = ft.getWidth();
              int64_t n = 1;
              for (auto d : rt.getShape()) n *= d;
              sharedBytes += n * (bits / 8);
            }
          }
        }

        // ---- i32 整数语义 ----
        // 约定：i32 值在 64 位寄存器中一律保持符号扩展（与 LW 一致）。
        // 由此：加/减/乘/取余用 W 指令（ADDW/SUBW/MULW/REMW/REMUW）以保证回绕；
        // bit31 为 1 的常量在 LI32 后补 sext.w；SLT/SLTU 直接给出 32 位
        // 有符号/无符号比较。有符号 x % 2^k 用移位序列，不用除法。
        {
          auto intWidth = [](Type t) -> unsigned {
            if (auto rt = dyn_cast<RankedTensorType>(t)) t = rt.getElementType();
            if (auto it = dyn_cast<IntegerType>(t)) return it.getWidth();
            return 0;
          };
          unsigned rw = op->getNumResults() > 0 ? intWidth(op->getResult(0).getType()) : 0;
          unsigned ow = op->getNumOperands() > 0 ? intWidth(op->getOperand(0).getType()) : 0;
          auto W32 = [](uint32_t f7, uint32_t b, uint32_t a, uint32_t f3, uint32_t d) {
            return R(f7, b, a, f3, d, OPC_OP32);
          };
          bool done = false;
          if (nm == "arith.constant" && !isFloat && hasImm && rw > 0 && rw <= 32) {
            int32_t v = rw == 1 ? (imm & 1) : imm;            // i1 true 按 1 装载
            std::vector<uint32_t> tmp;
            LI32(tmp, rd, (uint32_t)v);
            buf.emitAll(tmp);
            if (v < 0) buf.emit(I(0, rd, 0x0, rd, OPC_OPIMM32));  // sext.w
            done = true;
          } else if (rw == 32 && nm == "arith.addi" && !hasImm) {
            buf.emit(W32(0x00, rs2, rs1, 0x0, rd)); done = true;   // addw
          } else if (rw == 32 && nm == "arith.subi") {
            buf.emit(W32(0x20, rs2, rs1, 0x0, rd)); done = true;   // subw
          } else if (rw == 32 && nm == "arith.muli") {
            buf.emit(W32(0x01, rs2, rs1, 0x0, rd)); done = true;   // mulw
          } else if (rw == 32 && nm == "arith.remsi" && hasImm && imm > 0 &&
                     (imm & (imm - 1)) == 0) {
            // 有符号 x % 2^k（C 语义，余数符号随被除数），无分支、不用除法：
            //   bias = (x >> 31) >>> (32-k)      x<0 时为 2^k-1，否则为 0
            //   r    = x - ((x + bias) & -2^k)
            // 全部为单周期 ALU 指令；rd 可能与 rs1 同寄存器，故先算入临时。
            const uint32_t T0 = 15, T1 = 16, T2 = 17;          // a5 a6 a7
            int k = 0;
            while ((1 << k) < imm) ++k;
            if (k == 0) {
              buf.emit(ADDI(rd, X0, 0));                       // x % 1 = 0
            } else {
              buf.emit(I(0x400 | 31, rs1, 0x5, T0, OPC_OPIMM32));   // sraiw
              buf.emit(I(32 - k, T0, 0x5, T0, OPC_OPIMM32));        // srliw
              buf.emit(W32(0x00, T0, rs1, 0x0, T1));                // addw
              if (imm <= 2048) {
                buf.emit(I((uint32_t)(-imm), T1, 0x7, T1, OPC_OPIMM)); // andi
              } else {
                std::vector<uint32_t> v;
                LI32(v, T2, ~(uint32_t)(imm - 1));
                buf.emitAll(v);
                buf.emit(AND(T1, T1, T2));
              }
              buf.emit(W32(0x20, T1, rs1, 0x0, rd));                // subw
            }
            done = true;
          } else if (rw == 32 && nm == "arith.remsi") {
            buf.emit(W32(0x01, rs2, rs1, 0x6, rd)); done = true;   // remw
          } else if (rw == 32 && nm == "arith.remui" &&
                     !(hasImm && imm > 0 && (imm & (imm - 1)) == 0)) {
            buf.emit(W32(0x01, rs2, rs1, 0x7, rd)); done = true;   // remuw
          } else if (nm == "arith.cmpi" && ow > 0) {
            if (auto pa = op->getAttrOfType<IntegerAttr>("predicate")) {
              int pv = (int)pa.getInt();
              if (pv >= 6 && pv <= 9) {                            // ult ule ugt uge
                bool swap = (pv == 7 || pv == 8);                  // ule/ugt 以 b<a 表示
                buf.emit(SLTU(rd, swap ? rs2 : rs1, swap ? rs1 : rs2));
                if (pv == 7 || pv == 9)                            // ule = !(b<a), uge = !(a<b)
                  buf.emit(I(1, rd, 0x4, rd, OPC_OPIMM));          // xori rd,rd,1
                done = true;
              }
            }
          }
          // ---- arith.cmpf：16 个 MLIR 谓词全部显式映射 ----
          // Triton 对 != 生成无序谓词 une，必须正确处理 NaN。
          // 无序谓词 = 对应有序谓词取反（NaN 时有序谓词为假，取反为真）。
          if (!done && nm == "arith.cmpf") {
            if (auto pa = op->getAttrOfType<IntegerAttr>("predicate")) {
              const uint32_t T = 15;                       // a5，emitOp 临时
              auto notr = [&]() { buf.emit(I(1, rd, 0x4, rd, OPC_OPIMM)); };
              auto ordered = [&](uint32_t d) {             // d = (a==a)&(b==b)
                buf.emit(FEQ_S(d, rs1, rs1));
                buf.emit(FEQ_S(T, rs2, rs2));
                buf.emit(AND(d, d, T));
              };
              switch ((int)pa.getInt()) {
              case 0:  buf.emit(ADDI(rd, X0, 0)); break;                  // false
              case 1:  buf.emit(FEQ_S(rd, rs1, rs2)); break;               // oeq
              case 2:  buf.emit(FLT_S(rd, rs2, rs1)); break;               // ogt
              case 3:  buf.emit(FLE_S(rd, rs2, rs1)); break;               // oge
              case 4:  buf.emit(FLT_S(rd, rs1, rs2)); break;               // olt
              case 5:  buf.emit(FLE_S(rd, rs1, rs2)); break;               // ole
              case 6:  buf.emit(FLT_S(rd, rs1, rs2));                      // one
                       buf.emit(FLT_S(T, rs2, rs1));
                       buf.emit(OR(rd, rd, T)); break;
              case 7:  ordered(rd); break;                                 // ord
              case 8:  buf.emit(FLT_S(rd, rs1, rs2));                      // ueq = !one
                       buf.emit(FLT_S(T, rs2, rs1));
                       buf.emit(OR(rd, rd, T)); notr(); break;
              case 9:  buf.emit(FLE_S(rd, rs1, rs2)); notr(); break;       // ugt = !ole
              case 10: buf.emit(FLT_S(rd, rs1, rs2)); notr(); break;       // uge = !olt
              case 11: buf.emit(FLE_S(rd, rs2, rs1)); notr(); break;       // ult = !oge
              case 12: buf.emit(FLT_S(rd, rs2, rs1)); notr(); break;       // ule = !ogt
              case 13: buf.emit(FEQ_S(rd, rs1, rs2)); notr(); break;       // une = !oeq
              case 14: ordered(rd); notr(); break;                         // uno
              case 15: buf.emit(ADDI(rd, X0, 1)); break;                   // true
              default: ++unsupported["arith.cmpf with unknown predicate"]; break;
              }
              done = true;
            }
          }
          // ---- i32 <-> f32 转换 ----
          // fcvt.w.s / fcvt.s.w 在本核上符合规范。只支持 i32/u32 与 f32 之间，
          // 其余位宽拒绝。
          //   fptosi/fptoui: rtz 截断（Triton 语义），RV64 上结果符号扩展到 64 位
          //   sitofp/uitofp: 显式 RNE，不依赖 frm CSR
          if (!done && (nm == "arith.fptosi" || nm == "arith.fptoui" ||
                        nm == "arith.sitofp" || nm == "arith.uitofp")) {
            auto elt = [](Type t) {
              if (auto rt = dyn_cast<RankedTensorType>(t)) t = rt.getElementType();
              return t;
            };
            Type it = elt(nm == "arith.fptosi" || nm == "arith.fptoui"
                              ? op->getResult(0).getType() : op->getOperand(0).getType());
            Type ft = elt(nm == "arith.fptosi" || nm == "arith.fptoui"
                              ? op->getOperand(0).getType() : op->getResult(0).getType());
            if (it.isInteger(32) && ft.isF32()) {
              uint32_t uns = (nm == "arith.fptoui" || nm == "arith.uitofp") ? 1 : 0;
              if (nm == "arith.fptosi" || nm == "arith.fptoui")
                buf.emit(R(0x60, uns, rs1, 0x1, rd, OPC_FP));   // fcvt.w[u].s rtz
              else
                buf.emit(R(0x68, uns, rs1, 0x0, rd, OPC_FP));   // fcvt.s.w[u] rne
            } else {
              std::string s;
              llvm::raw_string_ostream os(s);
              op->getOperand(0).getType().print(os);
              ++unsupported[nm.str() + " from " + os.str()];
            }
            done = true;
          }
          // ---- arith.maximumf / minimumf：任一输入为 NaN 时结果为 NaN ----
          // maxnumf/minnumf 的 fmin.s 序列会忽略 NaN，不能直接复用。先算
          // ordered 标志（feq.s x,x 仅对 NaN 为假），再算 num 版本结果，
          // 最后按位合并：ordered ? 结果 : 0x7fc00000。标志必须先算，
          // 因为 rd 可能与某个输入共用寄存器。
          if (!done && (nm == "arith.maximumf" || nm == "arith.minimumf")) {
            const uint32_t A = 15, Bt = 16, Q = 17;          // a5 a6 a7
            buf.emit(FEQ_S(A, rs1, rs1));
            buf.emit(FEQ_S(Bt, rs2, rs2));
            buf.emit(AND(A, A, Bt));                         // ordered
            emitOp(nm == "arith.maximumf" ? "arith.maxnumf" : "arith.minnumf",
                   buf, rd, rs1, rs2, 0, false, true);
            buf.emit(FMV_X_W(Bt, rd));
            buf.emit(SUB(A, X0, A));                         // mask
            buf.emit(AND(Bt, Bt, A));
            buf.emit(I(0xFFF, A, 0x4, A, OPC_OPIMM));        // ~mask
            buf.emit(LUI(Q, 0x7fc00));                       // canonical NaN
            buf.emit(AND(Q, Q, A));
            buf.emit(OR(Bt, Bt, Q));
            buf.emit(FMV_W_X(rd, Bt));
            done = true;
          }
          if (done) { ++nMapped; return; }
        }

        if (emitOp(nm, buf, rd, rs1, rs2, imm, hasImm, isFloat))
          ++nMapped;
        else if (nm != "tt.func" && nm != "tt.return" && nm != "builtin.module") {
          ++nSkipped;
          skippedNames[nm.str()]++;
          // 区分"漏算"与"无害"。被跳过的 op 若参与数据流，产出的二进制
          // 就是错的，所以必须编译失败，不能只记进 metadata。
          //
          // 典型例子：循环边界为运行时变量时，scf.for / scf.yield 无法
          // 展开，跳过它们会让循环体执行次数错误。
          //
          // 例外：纯诊断 / 无数值副作用的 op 只计数，不致命。
          static const char *kHarmless[] = {
              "tt.assert", "tt.print", "tt.elementwise_inline_asm",
              "ttg.local_dealloc", "builtin.unrealized_conversion_cast",
          };
          bool harmless = false;
          for (const char *h : kHarmless)
            if (nm == h) { harmless = true; break; }
          if (!harmless) fatalSkipped[nm.str()]++;
        }
      });
      // CTA 特征向量第 8 维用真实寄存器压力（线性扫描前的活跃值总数）
      regPressureIntOut = regPressureInt;
      regPressureFltOut = regPressureFlt;
      if (regPeakInt > regPeakIntOut) regPeakIntOut = regPeakInt;
      if (regPeakFlt > regPeakFltOut) regPeakFltOut = regPeakFlt;
      if ((int64_t)(regPressureInt + regPressureFlt) > maxRegUsed)
        maxRegUsed = (int64_t)(regPressureInt + regPressureFlt);
      // 池子耗尽 -> 导出为属性；编译末尾据此报错。
      if (!regAllocOK) regExhausted = true;

      totPtrArgs = (int64_t)nPtrArgs;

      // ---- SIMT epilogue ----
      buf.emit(SYN_WARP(0));   // 收敛
      buf.emit(END_WARP());    // 0x00E0007B
    });

    OpBuilder b(mod.getContext());
    mod->setAttr("polaris.riscv.num_func", b.getI64IntegerAttr(nFunc));
    mod->setAttr("polaris.riscv.num_inst", b.getI64IntegerAttr((int64_t)buf.size()));
    mod->setAttr("polaris.riscv.num_mapped", b.getI64IntegerAttr(nMapped));
    mod->setAttr("polaris.riscv.num_skipped", b.getI64IntegerAttr(nSkipped));
    mod->setAttr("polaris.riscv.select_count", b.getI64IntegerAttr(selCount));
    mod->setAttr("polaris.riscv.select_insts", b.getI64IntegerAttr(selInsts));
    mod->setAttr("polaris.riscv.reg_exhausted",
                 b.getI64IntegerAttr(regExhausted ? 1 : 0));
    mod->setAttr("polaris.riscv.reg_pressure_int",
                 b.getI64IntegerAttr(regPressureIntOut));
    mod->setAttr("polaris.riscv.reg_pressure_flt",
                 b.getI64IntegerAttr(regPressureFltOut));
    // 峰值同时活跃数：真正需要的寄存器数。与 reg_pressure_*（SSA 值总数）
    // 区分开，否则看图会误以为溢出（如 36 个 SSA 值 > 20 个寄存器池），
    // 而线性扫描复用后峰值可能只有个位数。
    mod->setAttr("polaris.riscv.reg_peak_int",
                 b.getI64IntegerAttr(regPeakIntOut));
    mod->setAttr("polaris.riscv.reg_peak_flt",
                 b.getI64IntegerAttr(regPeakFltOut));
    // 把被跳过的 op 名字也导出。被跳过的 op 若参与数据流，产出的二进制就是
    // 错的，所以下面据此让编译失败；名单用于定位是哪个 op。
    if (!skippedNames.empty()) {
      std::string s;
      for (auto &kv : skippedNames) {
        if (!s.empty()) s += ",";
        s += kv.first + "x" + std::to_string(kv.second);
      }
      mod->setAttr("polaris.riscv.skipped_ops", b.getStringAttr(s));
    }
    // 有算子被漏掉且会影响数值时，直接让编译失败，而不是产出错的二进制。
    // 设 POLARIS_ALLOW_SKIPPED=1 可退回旧行为（仅用于排查，不要用于生产）。
    if (!fatalSkipped.empty()) {
      const char *allow = getenv("POLARIS_ALLOW_SKIPPED");
      if (!(allow && allow[0] == '1')) {
        std::string s;
        for (auto &kv : fatalSkipped) {
          if (!s.empty()) s += ", ";
          s += kv.first + " x" + std::to_string(kv.second);
        }
        mod.emitError()
            << "polaris backend: " << fatalSkipped.size()
            << " operation kind(s) could not be lowered and were skipped: "
            << s
            << ". The emitted binary would silently compute a wrong result, "
               "so compilation is aborted. Common cause: a loop whose bound "
               "is a runtime value (scf.for/scf.yield) -- the bound must be a "
               "tl.constexpr because loops are fully unrolled. "
               "Set POLARIS_ALLOW_SKIPPED=1 to override (debugging only).";
        return signalPassFailure();
      }
    }
    {
      const char *allow = getenv("POLARIS_ALLOW_SKIPPED");
      bool override = allow && allow[0] == '1';
      auto join = [](const std::map<std::string, int> &m) {
        std::string s;
        for (auto &kv : m) {
          if (!s.empty()) s += ", ";
          s += kv.first + " x" + std::to_string(kv.second);
        }
        return s;
      };
      for (unsigned a : loadArgs)
        if (storeArgs.count(a))
          ++unsupported["load and store through the same pointer argument #" +
                        std::to_string(a) +
                        " (both SMs run every CTA, so read-modify-write is "
                        "applied twice)"];
      if (!unsupported.empty() && !override) {
        mod.emitError()
            << "polaris backend: construct not supported: " << join(unsupported)
            << ". Lowering it would silently compute a wrong result, so "
               "compilation is aborted. "
               "Set POLARIS_ALLOW_SKIPPED=1 to override (debugging only).";
        return signalPassFailure();
      }
      if (!badReduce.empty() && !override) {
        mod.emitError()
            << "polaris backend: cross-lane reduction not supported: "
            << join(badReduce)
            << ". The core has no lane-to-lane path, so tl.sum/tl.max/tl.min "
               "over a dimension longer than 1 would silently return each "
               "lane's own value. Reduce with an explicit loop instead. "
               "Set POLARIS_ALLOW_SKIPPED=1 to override (debugging only).";
        return signalPassFailure();
      }
      if (!badMem.empty() && !override) {
        mod.emitError()
            << "polaris backend: memory access not supported: " << join(badMem)
            << ". Only unmasked loads/stores of 32-bit elements (i32/f32) are "
               "lowered; other widths would access the wrong bytes and masks "
               "would be ignored, silently corrupting memory. "
               "Set POLARIS_ALLOW_SKIPPED=1 to override (debugging only).";
        return signalPassFailure();
      }
    }
    if (!badShapes.empty()) {
      const char *allow = getenv("POLARIS_ALLOW_SKIPPED");
      if (!(allow && allow[0] == '1')) {
        std::string s;
        for (auto &kv : badShapes) {
          if (!s.empty()) s += ", ";
          s += kv.first + " x" + std::to_string(kv.second);
        }
        mod.emitError()
            << "polaris backend: tensor shape not supported: " << s
            << ". Every tensor must hold exactly 8 elements (one per SIMT "
               "lane); larger blocks are not split and smaller ones are not "
               "masked, so the binary would silently skip or overwrite "
               "elements. Use tl.arange(0, 8) and launch more programs. "
               "Set POLARIS_ALLOW_SKIPPED=1 to override (debugging only).";
        return signalPassFailure();
      }
    }
    // 寄存器池耗尽必须编译失败：本后端不生成 spill 代码，若继续发射，
    // 多个活跃值会共用同一寄存器，产出的二进制静默算错。
    if (regExhausted) {
      const char *allow = getenv("POLARIS_ALLOW_SKIPPED");
      if (!(allow && allow[0] == '1')) {
        mod.emitError()
            << "polaris backend: register pool exhausted (peak live int="
            << regPeakIntOut << "/" << kNumAllocRegs << ", fp="
            << regPeakFltOut << "/20). The backend does not implement register "
               "spilling, so the kernel cannot be compiled. Reduce live values, e.g. "
               "split the kernel or unroll fewer iterations. "
               "Set POLARIS_ALLOW_SKIPPED=1 to override (debugging only).";
        return signalPassFailure();
      }
    }
    // 导出参数布局供 host 侧对齐: 第 i 个指针参数 = arg_base + i*arg_stride
    mod->setAttr("polaris.riscv.num_ptr_args", b.getI64IntegerAttr(totPtrArgs));
    mod->setAttr("polaris.riscv.arg_base", b.getI64IntegerAttr(ARG_BASE));
    mod->setAttr("polaris.riscv.arg_stride", b.getI64IntegerAttr(ARG_STRIDE));

    // ---- CTA 元数据: 8 维程序特征向量, 供 SNN 动态 Warp 调度器选调度策略 ----
    // 硬件接口: sche.scala:73 `snnspecvec = Vec(snn_input_size, UInt(8.W))`,
    // snn_input_size=8 (SNNSche.scala:11), 由 CTA 包携带
    // (NutShell.scala:248 cta.io.hostCTAs.bits.snnspecvec)。
    // 每维取值 0-255, 这里从 kernel 的静态特征归一化得到:
    //   [0] 访存密度   [1] 浮点密度   [2] 整数密度   [3] TensorCore 使用
    //   [4] 分支/选择密度  [5] barrier 密度  [6] 指令总量档位  [7] 寄存器压力
    {
      int64_t tot = (int64_t)buf.size() > 0 ? (int64_t)buf.size() : 1;
      auto q = [&](int64_t cnt) -> int64_t {   // 归一化到 0-255
        int64_t v = cnt * 255 / tot;
        return v > 255 ? 255 : (v < 0 ? 0 : v);
      };
      std::vector<int32_t> vec(8, 0);
      vec[0] = (int32_t)q(featMem);
      vec[1] = (int32_t)q(featFp);
      vec[2] = (int32_t)q(featInt);
      vec[3] = (int32_t)q(featTc);
      vec[4] = (int32_t)q(featSel);
      vec[5] = (int32_t)q(featBar);
      // 指令量档位: <64 -> 低, <256 -> 中, 否则高
      vec[6] = (int32_t)(buf.size() < 64 ? 64 : (buf.size() < 256 ? 160 : 255));
      // 寄存器压力: 用不同 SSA 值数占可分配寄存器数的比例
      vec[7] = (int32_t)(maxRegUsed * 255 / (kNumAllocRegs > 0 ? kNumAllocRegs : 1));
      if (vec[7] > 255) vec[7] = 255;
      mod->setAttr("polaris.cta.snnspecvec", b.getDenseI32ArrayAttr(vec));
      mod->setAttr("polaris.cta.snn_input_size", b.getI64IntegerAttr(8));
      // 共享内存需求: 当前 RTL 的 SM 内无 shared memory 实体(仅调度器记账
      // availSharedMem/sharedMemSize), tl.shared 无物理载体, 故报 0。
      // 若 kernel 用了 tt.gpu 的 shared 分配, 这里会记录字节数供调度器检查。
      mod->setAttr("polaris.cta.shared_mem_bytes",
                   b.getI64IntegerAttr(sharedBytes));
    }

    // ---- 硬性约束：指针参数个数与代码长度 ----
    // 指针参数固定装入 kParamRegs（6 个）；第 7 个起 mapParamReg 会回绕，
    // 两个指针共用一个寄存器，访存静默落到错误的数组上。
    if (totPtrArgs > kNumParamRegs) {
      mod.emitError() << "polaris backend: too many pointer arguments: "
                      << totPtrArgs << " (at most " << kNumParamRegs
                      << " are supported). Pack several arrays into one buffer "
                         "and pass offsets as tl.constexpr.";
      return signalPassFailure();
    }
    // 代码从 0x80000000 起，参数区从 ARG_BASE 起；host 还会在 kernel 末尾
    // 追加约 30 条回读指令，故 kernel 本体上限为参数区之前留出 512 字节。
    {
      const size_t limitBytes = (ARG_BASE - 0x80000000u) - 512;
      const size_t codeBytes = buf.size() * 4;
      if (codeBytes > limitBytes) {
        mod.emitError() << "polaris backend: kernel too large: " << codeBytes
                        << " bytes of code exceeds the " << limitBytes
                        << "-byte limit before the argument area at 0x"
                        << llvm::utohexstr(ARG_BASE)
                        << ". Reduce unrolling (smaller tl.constexpr sizes) "
                           "or split the kernel.";
        return signalPassFailure();
      }
    }

    // ---- 硬性约束：生成代码不得读写 f0 ----
    // 32 个浮点寄存器中只有 f0 在"作为浮点运算目的、隔一条指令再读"时
    // 变成规范 NaN（0xffc00000），插 nop 无效。各降级都不使用 f0，这里再做
    // 一道检查：任何浮点字段指向 f0 就拒绝编译（视为后端 bug）。
    // 只检查确实是浮点寄存器的字段：fmv.x.w / fcvt.w.s / 比较的 rd、
    // fmv.w.x / fcvt.s.w 的 rs1 是整数寄存器，为 0 时是 x0，不算违规。
    {
      auto usesF0 = [](uint32_t w) -> bool {
        uint32_t opc = w & 0x7f, rd = (w >> 7) & 31, rs1 = (w >> 15) & 31,
                 rs2 = (w >> 20) & 31, rs3 = (w >> 27) & 31, f5 = w >> 27;
        switch (opc) {
        case 0x07: return rd == 0;                              // flw
        case 0x27: return rs2 == 0;                             // fsw
        case 0x43: case 0x47: case 0x4b: case 0x4f:             // fmadd 族
          return rd == 0 || rs1 == 0 || rs2 == 0 || rs3 == 0;
        case 0x53: {                                            // OP-FP
          bool rdF = true, rs1F = true, rs2F = true;
          switch (f5) {
          case 0x0b: rs2F = false; break;                       // fsqrt
          case 0x14: rdF = false; break;                        // feq/flt/fle
          case 0x18: rdF = false; rs2F = false; break;          // fcvt.w.s
          case 0x1a: rs1F = false; rs2F = false; break;         // fcvt.s.w
          case 0x1c: rdF = false; rs2F = false; break;          // fmv.x.w/fclass
          case 0x1e: rs1F = false; rs2F = false; break;         // fmv.w.x
          default: break;
          }
          return (rdF && rd == 0) || (rs1F && rs1 == 0) || (rs2F && rs2 == 0);
        }
        default: return false;
        }
      };
      int nF0 = 0;
      for (uint32_t x : buf.words()) nF0 += usesF0(x) ? 1 : 0;
      if (nF0 > 0) {
        mod.emitError()
            << "polaris backend: internal error: " << nF0
            << " emitted floating-point instruction(s) use register f0, "
               "which loses its value on this core when it is the "
               "destination of an FP operation and is read later. This is a "
               "backend bug (f0 must never be allocated); please report it.";
        return signalPassFailure();
      }
    }

    std::vector<int32_t> w;
    for (uint32_t x : buf.words()) w.push_back(static_cast<int32_t>(x));
    mod->setAttr("polaris.riscv.words", b.getDenseI32ArrayAttr(w));

    // ---- CodeGen 输出 .S 汇编(含 RV-SIMT 扩展指令) ----
    // 自定义 warp 控制与 TID 用 .insn 表达, 标准 GNU as 即可汇编回与
    // 直出二进制逐字节相同的 bin（命令见 RiscvAsmWriter.h）。
    {
      std::string kname = "polaris_kernel";
      mod.walk([&](mlir::triton::FuncOp fn) {
        if (kname == "polaris_kernel") kname = fn.getName().str();
      });
      std::string asmText = rv::toAsm(buf.words(), kname);
      mod->setAttr("polaris.riscv.asm", b.getStringAttr(asmText));
      mod->setAttr("polaris.riscv.kernel_name", b.getStringAttr(kname));
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace polaris {

std::unique_ptr<mlir::Pass> createPolarisRiscvCodegenPass() {
  return std::make_unique<PolarisRiscvCodegenPass>();
}

} // namespace polaris
} // namespace triton
} // namespace mlir
