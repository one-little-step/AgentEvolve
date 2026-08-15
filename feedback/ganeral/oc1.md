Diagnosis complete. Here's the root cause, with evidence.
The wrapper's tracing is not broken — the agent never actually calls the tools
Proof it's not our code: I tested the capture mechanism in isolation (ToolCallTracker.start_tracking → Tool.ainvoke on a @tracked_tool tool) and it records the call correctly for both sync and async paths.
Proof of the real problem: I ran a decisive probe — a tool with an unguessable return value and a side effect that appends to a file (terminal_output/cuga-tracing/tool-invocation-probe.log). Result:
- The model said "Calling the probe tool now…" then "I'm unable to call the tool in the current execution context."
- The side-effect file was never written → the tool was never invoked.
- InvokeResult.tool_calls = [] (correctly, because zero tools ran).
Root cause
CUGA 0.3.1 is a CodeAct agent: the model must emit Python code (or native tool_calls) that the sandbox then executes. Your configured model is azure/gpt-5.6-luna — a reasoning model (CUGA wraps it as ReasoningChatOpenAI, logs "Skipping temperature for reasoning model"). In balanced mode this model returns natural-language prose about using tools, never executable code/tool calls. The graph classifies it as "interim" and auto-continues, but executes nothing.
That's also why the multistep arithmetic task "worked": the reasoning model just did 1234×5678 mentally. It never called the calculator.
Consequence
events, events.jsonl, and captured_event_count are empty because no tools are actually being called — the trace is honest, not broken.
What this means
There is no wrapper-code fix here; the gap is CUGA↔model compatibility. To get real tool-call trajectories you need a model/mode that emits executable code or native tool calls (a non-reasoning model, or a different CUGA mode/config). I corrected sdk-verification-matrix.md to record this accurately (my earlier "tracker disabled" note was incomplete).
Two diagnostic scripts are left uncommitted for your re-use: scripts/live_trace_smoke.py (multistep trace smoke) and scripts/probe_tool_tracing.py (definitive tool-invocation probe).
Want me to try switching to a non-reasoning model (or a different CUGA mode like fast/accurate) to confirm the fix end-to-end, or leave that as a separate CUGA-integration pass?