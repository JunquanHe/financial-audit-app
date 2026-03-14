import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import json
import time

# --- 配置部分 ---
st.set_page_config(
    page_title="智能企业风险评估与审计助手",
    page_icon="🛡️",
    layout="wide"
)

# 尝试从环境变量获取默认 Key，如果没有则为空
DEFAULT_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# --- 模拟数据生成器 ---
def get_mock_financial_data(company_name):
    """模拟获取公司财务数据"""
    # 设置随机种子，保证同一次会话中数据不变
    if 'seed' not in st.session_state:
        st.session_state.seed = np.random.randint(0, 1000)
    
    np.random.seed(st.session_state.seed + hash(company_name) % 100)
    
    data = {
        "指标": ["营收增长率 (%)", "净利润率 (%)", "资产负债率 (%)", "流动比率", "经营性现金流 (百万)", "应收账款周转天数"],
        "数值": [
            round(np.random.uniform(-5, 25), 2),
            round(np.random.uniform(5, 30), 2),
            round(np.random.uniform(30, 85), 2),
            round(np.random.uniform(0.8, 2.5), 2),
            round(np.random.uniform(-200, 800), 2),
            round(np.random.uniform(30, 120), 2)
        ],
        "行业平均": [10.0, 15.0, 50.0, 1.5, 200.0, 60.0],
    }
    df = pd.DataFrame(data)
    
    # 简单逻辑标记状态
    def check_status(row):
        if row['指标'] == '资产负债率 (%)' and row['数值'] > 70:
            return "⚠️ 高风险"
        if row['指标'] == '流动比率' and row['数值'] < 1.0:
            return "⚠️ 高风险"
        if row['指标'] == '营收增长率 (%)' and row['数值'] < 0:
            return "⚠️ 负增长"
        return "✅ 正常"

    df['状态'] = df.apply(check_status, axis=1)
    return df

# --- AI 交互核心逻辑 ---
def init_llm_chain(api_key):
    try:
        from langchain_community.llms import Tongyi
        # 使用 qwen-turbo 或 qwen-max，取决于你的配额
        llm = Tongyi(model="qwen-turbo", dashscope_api_key=api_key)
        return llm
    except Exception as e:
        st.error(f"❌ 模型初始化失败: {e}")
        return None

def analyze_risk_and_audit(llm, company_name, df_fin):
    """生成风险评分和审计重点"""
    data_str = df_fin.to_string(index=False)
    
    prompt_text = f"""
    你是一位资深的首席审计官。
    公司名称：{company_name}
    财务数据：
    {data_str}
    
    任务：
    1. 给出一个 0-100 的风险评分（0为无风险，100为极高风险）。
    2. 列出 3-5 个具体的审计重点领域。
    3. **必须且仅**返回标准的 JSON 格式，不要包含 markdown 代码块标记（如 ```json），格式如下：
    {{
        "score": 整数,
        "reason": "简短的评分理由",
        "audit_focus": ["重点1", "重点2", "重点3"]
    }}
    """
    try:
        response = llm.invoke(prompt_text)
        # 清理可能存在的 markdown 标记
        clean_response = response.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_response)
    except Exception as e:
        st.warning(f"AI 解析警告: {e}。将使用默认评估。")
        return {
            "score": 50, 
            "reason": "AI 解析失败，采用默认中性评估", 
            "audit_focus": ["人工复核所有异常指标", "检查现金流真实性", "确认负债结构"]
        }

def generate_full_assessment(llm, company_name, df_fin):
    """生成完整的财务评估报告"""
    data_str = df_fin.to_string(index=False)
    prompt_text = f"""
    你是一位资深财务分析师。请基于以下数据为公司 {company_name} 撰写一份完整的财务评估报告。
    数据：{data_str}
    
    报告结构要求（使用 Markdown 格式）：
    1. **总体评价**：一句话总结。
    2. **盈利能力分析**：结合营收增长率和净利润率。
    3. **偿债能力与流动性**：结合资产负债率和流动比率。
    4. **运营效率**：结合应收账款周转天数。
    5. **潜在风险预警**：指出具体的高风险点。
    """
    try:
        return llm.invoke(prompt_text)
    except Exception as e:
        return f"⚠️ 生成详细报告失败: {e}"

# --- 页面布局 ---

st.title("🛡️ 智能企业风险评估与审计助手")
st.markdown("输入公司名称，获取 AI 驱动的风险评分、审计重点及深度财务诊断。")

# 侧边栏：配置
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    # API Key 输入
    api_key_input = st.text_input(
        "DashScope API Key", 
        value=DEFAULT_API_KEY, 
        type="password",
        help="请在阿里云 DashScope 控制台获取密钥"
    )
    
    if not api_key_input:
        st.error("⚠️ **未检测到 API Key**\n\n请在上方输入或在环境变量中设置 `DASHSCOPE_API_KEY`。\n否则 AI 功能将无法使用。")
        api_key = None
    else:
        api_key = api_key_input
        st.success("✅ API Key 已就绪")
    
    st.markdown("---")
    st.markdown("**操作指南**:")
    st.markdown("1. 输入公司名并点击分析。")
    st.markdown("2. 查看风险评分与图表。")
    st.markdown("3. 在底部与 AI 对话追问细节。")

# 主输入区
col1, col2 = st.columns([3, 1])
with col1:
    company_name = st.text_input("请输入公司名称", placeholder="例如：贵州茅台, 特斯拉, 阿里巴巴")
with col2:
    analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)

# 会话状态初始化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "full_report" not in st.session_state:
    st.session_state.full_report = None
# 新增：存储 DataFrame 用于绘图
if "df_fin" not in st.session_state:
    st.session_state.df_fin = None

# 执行分析逻辑
if analyze_btn:
    if not company_name:
        st.warning("⚠️ 请输入公司名称！")
    elif not api_key:
        st.error("❌ 缺少 API Key，无法进行分析。请在左侧侧边栏输入。")
    else:
        with st.spinner(f"🤖 AI 正在深度分析 {company_name} 的财务数据..."):
            try:
                # 1. 获取/生成数据
                df_fin = get_mock_financial_data(company_name)
                st.session_state.df_fin = df_fin  # 【关键修复】保存到 session_state
                
                # 2. 初始化 LLM
                llm = init_llm_chain(api_key)
                
                if llm:
                    # 3. 获取风险评分和审计重点
                    result = analyze_risk_and_audit(llm, company_name, df_fin)
                    st.session_state.analysis_result = result
                    
                    # 4. 生成完整报告
                    st.session_state.full_report = generate_full_assessment(llm, company_name, df_fin)
                    
                    # 5. 初始化聊天历史
                    st.session_state.messages = [
                        {"role": "assistant", "content": f"已完成对 **{company_name}** 的初步分析。\n\n📊 **风险评分**: {result['score']}/100\n💡 **核心理由**: {result['reason']}\n\n您可以随时在下方提问，例如：'针对高负债率有什么审计建议？'"}
                    ]
                else:
                    st.error("模型初始化失败。")
                    
            except Exception as e:
                st.error(f"发生未知错误: {e}")

# 展示结果区域
if st.session_state.analysis_result and st.session_state.df_fin is not None:
    res = st.session_state.analysis_result
    df_plot = st.session_state.df_fin
    
    # --- 第一行：评分卡片 & 图表 ---
    c1, c2 = st.columns([1, 2])
    
    with c1:
        score = res['score']
        if score < 40:
            color_delta = "normal" # 绿色
            risk_level = "低风险"
        elif score < 70:
            color_delta = "warning" # 橙色
            risk_level = "中风险"
        else:
            color_delta = "inverse" # 红色
            risk_level = "高风险"
            
        st.metric(label="🛡️ 综合风险评分 (0-100)", value=score, delta=risk_level)
        st.info(f"**分析结论**: {res['reason']}")
        
        st.markdown("### 🔍 审计重点建议")
        for i, focus in enumerate(res['audit_focus'], 1):
            st.markdown(f"{i}. {focus}")

    with c2:
        st.markdown("### 📊 财务指标 vs 行业平均")
        # 【关键修复】直接使用 df_plot (DataFrame)，不再进行 split 操作
        if not df_plot.empty:
            # 绘制分组柱状图
            fig = px.bar(
                df_plot, 
                x="指标", 
                y=["数值", "行业平均"], 
                barmode="group",
                title=f"{company_name} 关键财务指标对比",
                labels={"value": "数值", "variable": "类别"},
                color_discrete_map={"数值": "#1f77b4", "行业平均": "#ff7f0e"}
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("暂无数据可绘图")

    st.divider()

    # --- 第二行：完整财务评估报告 ---
    with st.expander("📑 点击查看：完整财务评估报告", expanded=False):
        if st.session_state.full_report:
            st.markdown(st.session_state.full_report)
        else:
            st.write("报告生成中...")

    st.divider()

    # --- 第三行：AI 对话交互 ---
    st.subheader("💬 与审计专家 AI 对话")
    
    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入
    if prompt := st.chat_input("输入您的问题，例如：'为什么流动比率低于行业平均？'"):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成回复
        with st.chat_message("assistant"):
            with st.spinner("AI 思考中..."):
                if api_key:
                    llm = init_llm_chain(api_key)
                    if llm:
                        # 构建上下文：包含公司名、数据表格、之前的分析结果
                        context_data = st.session_state.df_fin.to_string(index=False)
                        context_prompt = f"""
                        角色：资深审计师。
                        背景数据：
                        {context_data}
                        
                        之前的分析结论：{res}
                        
                        用户问题：{prompt}
                        
                        请结合上述具体数据回答用户问题，保持专业、客观。
                        """
                        try:
                            response = llm.invoke(context_prompt)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"对话生成失败: {e}")
                    else:
                        st.error("模型连接断开。")
                else:
                    st.error("请先在侧边栏配置 API Key。")

elif analyze_btn and not api_key:
    st.info("👈 请在左侧侧边栏输入 API Key 后重新开始分析。")

# 页脚
st.markdown("---")
st.caption("免责声明：本工具演示用数据为模拟生成，AI 回答仅供参考，不构成正式投资或审计建议。")