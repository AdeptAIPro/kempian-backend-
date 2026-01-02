# State Machine Diagram - Final

## PayRun State Transitions

```
                    ┌─────────┐
                    │  DRAFT  │
                    └────┬────┘
                         │ approve()
                         ▼
              ┌──────────────────────┐
              │  APPROVAL_PENDING   │
              └────┬─────────────────┘
                   │ validate_funds()
                   ▼
              ┌──────────────────────┐
              │  FUNDS_VALIDATED     │
              │  [Funds Locked]       │
              └────┬─────────────────┘
                   │ process_payments()
                   ▼
              ┌──────────────────────┐
              │  PAYOUT_INITIATED   │
              └────┬─────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐
   │COMPLETE│ │PARTIAL │ │ FAILED │
   │        │ │COMPLETE│ │        │
   └────────┘ └────────┘ └────┬───┘
                              │ reverse()
                              ▼
                         ┌────────┐
                         │REVERSED│
                         └────────┘
```

## Valid Transitions

| From State | To State | Condition |
|------------|----------|-----------|
| draft | approval_pending | approve() |
| approval_pending | funds_validated | validate_funds() |
| funds_validated | payout_initiated | process_payments() |
| payout_initiated | partially_completed | Some payments succeed, some fail |
| payout_initiated | completed | All payments succeed |
| payout_initiated | failed | All payments fail |
| partially_completed | completed | Remaining payments succeed |
| partially_completed | failed | Remaining payments fail |
| failed | payout_initiated | retry() |
| completed | reversed | reverse() |

## Irreversible Actions

1. **force-resolve** - Admin-only, irreversible
2. **fraud alert rejection** - Permanently blocks payment
3. **funds locked** - Cannot be double-locked
4. **payment completed** - Cannot be undone (only reversed)

