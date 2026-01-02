# Production Launch Checklist (GO / NO-GO)

## Pre-Launch Requirements

### Database
- [ ] Run all migrations from `UPDATED_MIGRATION_INSTRUCTIONS.md`
- [ ] Verify all tables created
- [ ] Verify all indexes created
- [ ] Verify all foreign keys
- [ ] Test rollback procedure

### Configuration
- [ ] Razorpay API keys configured (live keys)
- [ ] Razorpay fund account ID set
- [ ] Payment gateway set to 'razorpay'
- [ ] Payment mode set (recommended: 'NEFT')
- [ ] Webhook URL configured in Razorpay dashboard
- [ ] Webhook secret stored securely

### Cron Jobs
- [ ] Reconciliation job scheduled (every 15-30 min)
- [ ] Balance sync job scheduled (hourly)
- [ ] Test cron jobs manually
- [ ] Verify cron job logging

### Security
- [ ] Encryption key set (SECRET_KEY)
- [ ] Payment encryption key configured
- [ ] All API keys encrypted
- [ ] Webhook signature verification enabled
- [ ] Audit logging enabled

### Testing
- [ ] Test funds validation
- [ ] Test fund locking (concurrent test)
- [ ] Test fraud detection
- [ ] Test reconciliation
- [ ] Test idempotency
- [ ] Test state machine transitions
- [ ] Test bank verification
- [ ] Test force-resolve (admin only)
- [ ] Test employee payment transparency

### Monitoring
- [ ] Payment success rate monitoring
- [ ] Fraud alert monitoring
- [ ] Reconciliation monitoring
- [ ] Error rate monitoring
- [ ] Balance monitoring

## GO / NO-GO Criteria

### ✅ GO Criteria (All Must Pass)
- [ ] All migrations successful
- [ ] Razorpay credentials valid
- [ ] Webhook endpoint accessible
- [ ] Cron jobs scheduled
- [ ] All critical tests passing
- [ ] Monitoring configured
- [ ] Backup/rollback plan ready

### ❌ NO-GO Criteria (Any One Fails)
- [ ] Migrations fail
- [ ] Razorpay credentials invalid
- [ ] Webhook not accessible
- [ ] Critical tests failing
- [ ] No monitoring
- [ ] No rollback plan

## Launch Day Checklist

- [ ] Final database backup
- [ ] Run migrations in production
- [ ] Verify all endpoints accessible
- [ ] Test webhook endpoint
- [ ] Verify cron jobs running
- [ ] Monitor first payment
- [ ] Verify reconciliation working
- [ ] Check audit logs

## Post-Launch Monitoring (First 24 Hours)

- [ ] Monitor payment success rate
- [ ] Monitor fraud alerts
- [ ] Monitor reconciliation
- [ ] Monitor error rates
- [ ] Check wallet balance sync
- [ ] Verify webhook delivery
- [ ] Review audit logs

## Rollback Plan

If critical issues:
1. Disable payment processing
2. Revert database migrations (if safe)
3. Switch to manual processing
4. Notify stakeholders
5. Investigate root cause

---

**Status**: ⚠️ **PENDING** - Complete checklist before launch

