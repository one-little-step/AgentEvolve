---
description: 'Playbook: status-format'
enabled: true
id: playbook_status-format
name: status-format
priority: 50
triggers:
  always: true
  natural_language:
  - 'When reporting project status, you MUST end your reply with the exact line: POLICY-MARKER:
    POL-8078061184'
  target: intent
  threshold: 0.5
type: playbook
---

When reporting project status, you MUST end your reply with the exact line: POLICY-MARKER: POL-8078061184