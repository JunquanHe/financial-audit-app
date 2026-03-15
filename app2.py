import os
import time
import pandas as pd
import random
from datetime import datetime, timedelta
from io import BytesIO
import base64

# Dash 相关库
import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# 模拟调用阿里云 DashScope (千问) SDK
try:
    import dashscope
    HAS_DASHSCOPE = True
except ImportError:
    HAS_DASHSCOPE = False

# ==========================================
# 1. 配置与模拟数据层 (核心逻辑)
# ==========================================

HISTORY_DB = [
    {"发票代码": "011002100111", "发票号码": "88991122", "金额": 5000, "日期": "2023-10-01"},
    {"发票代码": "011002100222", "发票号码": "77665544", "金额": 2000, "日期": "2023-11-15"},
]

def mock_ocr_engine(image_file):
    time.sleep(0.5) 
    codes = ["011002300111", "011002300222", "011002300333"]
    items_risky = ["高档白酒", "购物卡", "礼品礼盒", "办公用品 (大批量)", "会议费 (无明细)"]
    items_safe = ["A4 打印纸", "电脑耗材", "技术服务费", "差旅住宿费", "图书资料"]
    
    is_risky = random.random() > 0.6
    
    if is_risky:
        item = random.choice(items_risky)
        amount = random.randint(2000, 10000)
        code = random.choice(codes)
        number = str(random.randint(10000000, 99999999))
        if random.random() > 0.8:
            code, number = "011002100111", "88991122" 
    else:
        item = random.choice(items_safe)
        amount = random.randint(100, 3000)
        code = random.choice(codes)
        number = str(random.randint(10000000, 99999999))

    return {
        "发票代码": code,
        "发票号码": number,
        "开票日期": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
        "购买方": "XX 审计事务所",
        "销售方": "某某科技有限公司",
        "金额": amount,
        "税额": round(amount * 0.06, 2),
        "价税合计": amount + round(amount * 0.06, 2),
        "货物名称/摘要": item,
        "备注": "正常报销" if not is_risky else "摘要模糊，疑似违规"
    }

def call_qwen_ai(item_name, amount, context=""):
    prompt = f"""
    你是一名资深审计专家。请分析以下发票信息的风险：
    货物名称/摘要：{item_name}
    金额：{amount} 元
    请判断风险等级（高/中/低）和具体的审计建议。以 JSON 格式返回。
    """

    if not HAS_DASHSCOPE or not os.getenv("DASHSCOPE_API_KEY"):
        time.sleep(0.8) 
        risky_keywords = ["酒", "烟", "礼品", "卡", "招待", "美容", "健身", "大批量", "无明细"]
        reason = ""
        level = "低"
        suggestion = "无需特别关注，凭证齐全即可。"
        
        for kw in risky_keywords:
            if kw in item_name:
                level = "高"
                reason = f"检测到敏感词 '{kw}'，疑似违规列支。"
                suggestion = "建议追查具体消费明细，要求提供支付流水及具体人员清单。"
                break
        
        if "办公" in item_name and amount > 5000:
            level = "中"
            reason = "办公用品金额较大，且未附带具体清单。"
            suggestion = "必须附盖有销售方印章的详细销货清单。"
            
        return {
            "risk_level": level,
            "risk_reason": reason if reason else "未发现明显异常关键词。",
            "audit_suggestion": suggestion
        }

    # 真实调用逻辑略 (保持原样即可)
    return {"risk_level": "低", "risk_reason": "模拟成功", "audit_suggestion": "通过"}

# ==========================================
# 2. Dash 应用界面设计
# ==========================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("🤖 AuditMind: 基于千问大模型的智能审计助手", className="text-primary text-center my-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.P("🎯 **场景**：自动检查发票重复、金额异常及摘要敏感词。"),
                html.P("⚙️ **流程**：上传发票 -> OCR 提取 -> 规则查重 -> **千问 AI 语义分析** -> 生成报告。")
            ])
        ], className="mb-4"), width=12)
    ]),

    dbc.Row([
        # 左侧：上传与控制
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📂 步骤 1: 上传发票"),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            '📤 拖拽发票图片/PDF 到这里 或 ',
                            html.A('点击选择文件', style={'color': '#1E88E5', 'fontWeight': 'bold'})
                        ]),
                        style={
                            'width': '100%',
                            'height': '100px',
                            'lineHeight': '100px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        multiple=True
                    ),
                    html.Div(id='upload-status', className="mt-2 text-muted"),
                    dbc.Button("🚀 开始智能审计", id="start-audit-btn", color="primary", className="w-100 mt-3", disabled=True)
                ])
            ], className="mb-3"),
            
            dbc.Card([
                dbc.CardHeader("💬 步骤 3: 与 AI 审计师对话"),
                dbc.CardBody([
                    # Store 初始化为空列表，由回调动态填充
                    dcc.Store(id='chat-history', data=[]), 
                    html.Div(id='chat-display', style={'height': '200px', 'overflowY': 'auto', 'border': '1px solid #ddd', 'padding': '10px', 'marginBottom': '10px'}),
                    dbc.InputGroup([
                        dbc.Input(id="chat-input", placeholder="例如：为什么这张发票风险高？", type="text"),
                        dbc.InputGroupText(dbc.Button("发送", id="send-chat", color="secondary"))
                    ])
                ])
            ])
        ], width=4),

        # 右侧：结果展示
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 步骤 2: 审计结果分析"),
                dbc.CardBody([
                    html.Div(id='loading-output', children="等待上传...", className="text-center text-muted py-5"),
                    
                    html.Div(id='result-table-container', style={'display': 'none'}, children=[
                        dbc.Alert(id='summary-alert', color="info", dismissable=True),
                        html.H5("详细检测列表："),
                        # 修复点：这里只放一个空的 div 容器，表格内容由回调动态生成
                        html.Div(id='audit-table-container'), 
                        
                        html.H5("AI 深度解读："),
                        html.Div(id='ai-analysis-card', className="p-3 bg-light rounded border")
                    ])
                ])
            ])
        ], width=8)
    ]),
    
    html.Footer([
        html.P("Powered by Alibaba Cloud Qwen | Demo Version", className="text-center text-muted small")
    ], className="py-3")

], fluid=True)

# ==========================================
# 3. 交互逻辑回调 (已修复逻辑顺序和重复输出)
# ==========================================

@app.callback(
    [Output('upload-status', 'children'),
     Output('start-audit-btn', 'disabled')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_status(contents, filename):
    if contents:
        count = len(contents) if isinstance(contents, list) else 1
        return html.Span(f"✅ 已加载 {count} 个文件，准备就绪", className="text-success"), False
    return html.Span("等待上传...", className="text-muted"), True

@app.callback(
    [Output('loading-output', 'style'),
     Output('result-table-container', 'style'),
     Output('audit-table-container', 'children'),  # 修复点：输出给容器，而不是直接操作 table 属性
     Output('summary-alert', 'children'),
     Output('summary-alert', 'color'),
     Output('ai-analysis-card', 'children')],
    Input('start-audit-btn', 'n_clicks'),
    State('upload-data', 'contents'),
    prevent_initial_call=True
)
def run_audit(n_clicks, contents):
    if not contents:
        return {'display': 'block'}, {'display': 'none'}, [], "", "info", ""

    # --- 逻辑开始 (之前被 return 挡住了) ---
    time.sleep(1) 
    
    results = []
    high_risk_count = 0
    file_list = contents if isinstance(contents, list) else [contents]
    
    for _ in file_list:
        ocr_data = mock_ocr_engine(_)
        
        is_duplicate = any(
            d['发票代码'] == ocr_data['发票代码'] and d['发票号码'] == ocr_data['发票号码'] 
            for d in HISTORY_DB
        )
        
        ai_result = call_qwen_ai(ocr_data['货物名称/摘要'], ocr_data['金额'])
        
        final_risk = "低"
        reasons = []
        
        if is_duplicate:
            final_risk = "高"
            reasons.append("❌ 发票重复报销")
            high_risk_count += 1
        elif ai_result['risk_level'] == "高":
            final_risk = "高"
            reasons.append(f"⚠️ AI 发现：{ai_result['risk_reason']}")
            high_risk_count += 1
        elif ai_result['risk_level'] == "中":
            final_risk = "中"
            reasons.append(f"⚠️ AI 提示：{ai_result['risk_reason']}")
        
        results.append({
            "文件名": "invoice_001.jpg",
            "发票号码": ocr_data['发票号码'],
            "金额": ocr_data['价税合计'],
            "摘要": ocr_data['货物名称/摘要'],
            "查重结果": "❌ 重复" if is_duplicate else "✅ 唯一",
            "AI 风险等级": final_risk,
            "风险详情": "; ".join(reasons) if reasons else "✅ 正常",
            "AI 建议": ai_result['audit_suggestion']
        })

    df_res = pd.DataFrame(results)
    
    # 构建表格 HTML (修复点：手动构建表格以避免 from_dataframe 的属性问题)
    table_header = [html.Thead(html.Tr([html.Th(col) for col in df_res.columns]))]
    table_body = [html.Tbody([html.Tr([html.Td(row[col]) for col in df_res.columns]) for row in df_res.to_dict('records')])]
    
    audit_table = dbc.Table(
        table_header + table_body,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="table-sm"
    )
    
    # 构建摘要警报
    total = len(df_res)
    alert_msg = f"共扫描 {total} 张发票。发现 **{high_risk_count}** 张高风险发票，需立即介入审计！"
    alert_color = "danger" if high_risk_count > 0 else "success"
    
    # 构建 AI 深度解读卡片
    ai_html_content = []
    if high_risk_count > 0:
        risk_rows = df_res[df_res['AI 风险等级'] == '高']
        for idx, row in risk_rows.iterrows():
            ai_html_content.append(
                dbc.Alert([
                    html.H4(f"🚨 高风险预警：发票 {row['发票号码']}"),
                    html.P(f"**问题摘要**: {row['摘要']}"),
                    html.P(f"**AI 分析**: {row['风险详情']}"),
                    html.Hr(),
                    html.Strong("👮‍♂️ 审计师行动建议："),
                    html.P(row['AI 建议'])
                ], color="warning")
            )
    else:
        ai_html_content.append(html.P("🎉 所有发票均通过初步筛查，未发现明显违规风险。", className="text-success lead"))

    # --- 逻辑结束 ---

    # 注意：这里不再返回 chat-history 的初始化数据
    return (
        {'display': 'none'}, 
        {'display': 'block'}, 
        audit_table,  # 返回完整的表格组件
        alert_msg, 
        alert_color, 
        ai_html_content
    )

@app.callback(
    [Output('chat-display', 'children'),
     Output('chat-history', 'data')],
    Input('send-chat', 'n_clicks'),
    [State('chat-input', 'value'),
     State('chat-history', 'data'),
     State('audit-table-container', 'children')], # 稍微修改依赖，确保能获取到数据上下文（可选）
    prevent_initial_call=True
)
def chat_response(n_clicks, user_input, history, table_children):
    if not user_input:
        return no_update, no_update
    
    # 修复点：如果 history 为空，说明是第一次对话，在这里初始化
    if not history or len(history) == 0:
        # 尝试从表格数据中简单推断风险数量（这里简化处理，实际可解析 table_children）
        # 为了演示，我们假设用户已经看到了审计结果
        history = [{"role": "system", "content": "审计完成。您可以问我关于发票风险、重复报销或审计建议的问题。"}]
    
    # 添加用户消息
    new_history = history + [{"role": "user", "content": user_input}]
    display_msgs = [html.Div(f"👤 你：{user_input}", className="mb-2 text-primary")]
    
    # 模拟 AI 回复
    response_text = ""
    if "重复" in user_input:
        response_text = "标记为'❌ 重复'的发票意味着该发票代码和号码已在历史数据库中存在。这通常是重复报销的铁证。"
    elif "风险" in user_input or "为什么" in user_input:
        response_text = "AI 判定高风险主要基于：1. 敏感词（如礼品、酒）；2. 金额异常；3. 摘要模糊。"
    elif "建议" in user_input:
        response_text = "建议程序：1. 查验真伪；2. 索要清单；3. 访谈人员；4. 检查资金流。"
    else:
        response_text = "收到。作为您的 AI 审计助手，我可以帮您分析发票合规性。请问具体想查哪方面？"
    
    new_history.append({"role": "assistant", "content": response_text})
    display_msgs.append(html.Div(f"🤖 AI 审计师：{response_text}", className="mb-2 p-2 bg-light rounded"))
    
    return display_msgs, new_history

if __name__ == '__main__':
   print("🚀 正在启动 AuditMind 智能审计 Demo...")
   print("🌐 请在浏览器打开：http://127.0.0.1:8050")
   app.run(debug=True, port=8050)