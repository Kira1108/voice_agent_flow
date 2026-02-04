# Auto Finance Workflow

```python
>>>>>>> Turn 1  <<<<<<<
Customer: 喂，你好
Agent: 您好，请问是李老三吗？
>>>>>>> Turn 2  <<<<<<<
Customer: 呃
Collected information: {'name': '李老三'} # Agent 检测到用户没有明确拒绝，继续进行下一步
Transferring to FinancialSupportStatus # 执行Handoff
Agent: 请问您对我们的金融支持服务感兴趣吗？
>>>>>>> Turn 3  <<<<<<<
Customer: 再说一次，你说啥？
Agent: 请问您需要金融支持吗？ # Agent没有检测到明确回答，继续询问
>>>>>>> Turn 4  <<<<<<<
Customer: 哦哦，我有需求
Collected information: {'require_support': True} # Agent 收集到结构化信息，准备转移
Transferring to VehicleNotUnderRepayment # 执行Handoff
Agent: 请问您的车辆目前是已经还清贷款了吗？
>>>>>>> Turn 5  <<<<<<<
Customer: 还清了
Collected information: {'is_not_under_repayment': True} # Agent 收集到结构化信息，准备转移
Transferring to VehicleLiscenceUnderControl # 执行Handoff
Agent: 请问您的车辆行驶证现在是在您本人手上吗？
>>>>>>> Turn 6  <<<<<<<
Customer: 在手上
Collected information: {'is_under_control': True} # Agent 收集到结构化信息，准备转移
Transferring to AgreeToAddWechatAccount # 执行Handoff
Agent: 为了方便后续联系，您是否同意添加我们的微信账号？
>>>>>>> Turn 7  <<<<<<<
Customer: 可以
Collected information: {'agree': True} # Agent 检测到 用户同意，准备转移
Transferring to WeChatId # 执行Handoff
Agent: 请问您的微信号是多少？
>>>>>>> Turn 8  <<<<<<<
Customer: liushaoshan123
Collected information: {'wechat_id': 'liushaoshan123'} # Agent 收集到微信号，任务完成
All tasks completed.
Agent: 好的，我们稍后会加您的微信，请你注意在服务通知后查看我们的企业微信请求，再见！ # 再见话术直出

# 通话过程中收集到的所有信息
------------------------------ Collected Information ------------------------------
[
    {'name': '李老三'}, 
    {'require_support': True},
    {'is_not_under_repayment': True},
    {'is_under_control': True}, 
    {'agree': True}, 
    {'wechat_id': 'liushaoshan123'}
]
```