# Báo cáo cá nhân — Day 09 Multi-Agent A2A

| Thông tin | Nội dung |
| --- | --- |
| Họ và tên | Nguyễn Quang Sơn |
| MSSV | 2A202601956 |
| Vai trò | Multi-Agent Pipeline & Policy Engineer |

## Phần việc sở hữu

| Module | Bàn giao | Cách xác minh |
| --- | --- | --- |
| Data/Domain agents | `main.py`: Customer, Order/Product, Payment, Delivery | Unit test fixture và output batch |
| Policy/Verifier | `main.py`: FactVerificationAgent, PolicyAgent, VerifierAgent | Join chéo nguồn, policy priority, evidence ID, array limits |
| Coordinator online | `main.py`: `CoordinatorAgent`, `NvidiaNimCoordinatorReviewer` | Trace ghi 14 handoff và review Nemotron Nano 9B theo case |
| Tài liệu vận hành | `architecture.md`, `metadata.json`, `trace.jsonl` | Đọc runbook, chạy batch 50 case |

## Cách triển khai

Pipeline đọc toàn bộ CSV và tạo index theo order/customer để các agent dùng chung dữ
liệu nguồn. Customer Agent tách `customer_unique_id` khỏi `customer_id`; Order/Product
Agent không đưa order lịch sử vào affected entities; Payment và Delivery Agent tính
toán từ số/timestamp CSV. Fact Verification Agent coi complaint là input không
đáng tin, join lại order/customer/item/payment và tính lại các tổng tiền trước khi
cho phép Policy Agent áp dụng `EC_POLICY_V2` theo thứ tự ưu tiên.
Verifier kiểm các evidence ID được dựng từ CSV, giới hạn array, confidence và null
handling trước khi output được ghi.

`CoordinatorAgent` giao bảy nhiệm vụ và nhận bảy structured handoff từ Customer,
Order/Product, Payment, Delivery, Fact Verification, Policy và Verifier. Mỗi case lưu đủ
14 message giao/nhận trong một record của `trace.jsonl`, nên có thể kiểm chứng
agent nào thực hiện domain nào và payload nào được bàn giao. Verifier chạy sau
khi Coordinator lắp ráp candidate output và chặn sai schema, evidence ID, tiền,
null handling hoặc array limit trước khi ghi file.

Coordinator gọi `nvidia/nvidia-nemotron-nano-9b-v2` trên NVIDIA NIM. Model được dùng để
review handoff và ghi trace, không có quyền ghi đè kết quả xác định của policy engine.
Điều này vừa có multi-agent/LLM trace vừa tránh dữ kiện hoặc số tiền bịa đặt.

## Input/output contract

- Input: `input/EC_001.json` … `input/EC_050.json`, đúng schema README.
- Output: JSON cùng tên trong `output/`, đúng schema EC_POLICY_V2.
- Runtime config: `.env` chứa secret `NVIDIA_API_KEY`; model ID và parameter
  count 9B nằm trong source/`metadata.json` để chấm.
- Lỗi chặn: thiếu case, order ID không tồn tại, thiếu key, model không còn miễn phí,
  hoặc output không đủ 50 file.

## Quyết định kỹ thuật

Chọn NVIDIA Nemotron Nano 9B qua NVIDIA NIM thay vì tải Ollama local. Lý do: giới hạn model được
khai báo rõ dưới 10B, không mất thời gian tải model, còn số liệu nghiệp vụ vẫn được
quyết định bằng code tái lập được. Model free thay đổi theo provider nên runner xác
minh theo khai báo trước batch và không tự fallback sang model lớn hơn.

## Xác minh khi có input

```bash
python3 main.py run
python3 main.py package
```

Kết quả hợp lệ: 50 output JSON, 50 dòng `trace.jsonl`, `metadata.json` ghi model 9B,
trace có latency/token usage thật cho từng case, và `output.zip` chỉ chứa 50 JSON.
