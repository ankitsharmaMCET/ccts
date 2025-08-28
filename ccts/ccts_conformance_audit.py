
import os, re, json, sys, io

REQS = [
    # (id, description, filename, list of regex patterns that should all be found)
    ("3.1.1-3.1.3", "Effective intensity & step emissions: iota0=E0/Q0; iota=max(0,(E0-Acum)/Q0); e_t=q_t*iota",
     "model.py", [
         r"iota0\s*=\s*E0\s*/\s*Q0",
         r"iota\s*=\s*max\s*\(\s*0\s*,\s*(E0|E_zero)\s*-\s*(Acum|A_cum)\s*\)\s*/\s*Q0",
         r"(e_t|emissions|e)\s*=\s*(q_t|q|quantity)\s*\*\s*(iota|intensity)"
     ]),
    ("3.2.1-3.2.3", "Step allowances & provisional carbon cost proxy: allowed_t=tau_t*q_t; excess=max(0,e-allowed); C_proxy=excess*P_t (if enabled)",
     "model.py", [
         r"(allowed_t|allowed)\s*=\s*(tau_t|tau)\s*\*\s*(q_t|q)",
         r"(excess_t|excess)\s*=\s*max\s*\(\s*0\s*,\s*(e_t|e|emissions)\s*-\s*(allowed_t|allowed)\s*\)",
         r"(C_proxy|Ctilde|provisional|carbon_cost)\s*=\s*(excess_t|excess)\s*\*\s*(P_t|P|proxy_price)"
     ]),
    ("3.3.1", "Target selection by year with fallback geometric decay from iota0: tau_y = explicit or carry-forward else iota0*(1-r)^(y-ystart)",
     "model.py", [
         r"(tau_y|tau)\s*=.*(targets|target_intensity).*",
         r"(carry|last|previous|most\s+recent).*target",
         r"iota0\s*\*\s*\(\s*1\s*-\s*r\s*\)\s*\*\*|\)\s*\^\s*\("
     ]),
    ("3.4.1-3.4.4", "Annual aggregates: Qy=sum(q_t); Ey=sum(e_t); allowed_y=tau_y*Qy; g_y=E_ver_y-allowed_y",
     "analysis.py", [
         r"(Q_y|Qy)\s*=\s*sum\(",
         r"(E_y|Ey)\s*=\s*sum\(",
         r"(allowed_y|allowed)\s*=\s*(tau_y|tau)\s*\*\s*(Q_y|Qy)",
         r"(g_y|gap|compliance_gap)\s*=\s*(E_ver_y|E_y|Ey)\s*-\s*(allowed_y|allowed)"
     ]),
    ("3.5.1-3.5.6", "Year-end: surrender, borrow (beta_b*allowed_y), penalty with escalation by k_y; mint in surplus years",
     "agent.py", [
         r"(surrender|sy)\s*=\s*min\(",
         r"(g_tilde|g̃|residual|resid)\s*=\s*max\s*\(\s*0\s*,\s*(g_y|gap|compliance_gap)\s*-\s*(surrender|sy)\s*\)",
         r"(borrow|by)\s*=\s*min\s*\(\s*(g_tilde|g̃|residual).*(beta_b|βb).*(allowed_y|allowed)",
         r"(penalty|Penalty)\s*=\s*(g_tilde|g̃|residual).*(pi|π).*(phi|ϕ|k_y|ky)",
         r"(mint|issue).*credits|vintage"
     ]),
    ("3.6.1", "Production decision: SLSQP objective includes -[q*(p_u-c_u) - max(0, q*(iota-tau_t))*P_t - gamma(q)] - eta*Q0*|q-Q0| with bounds",
     "agent.py", [
         r"(SLSQP|scipy\.optimize|minimize)",
         r"max\s*\(\s*0\s*,\s*(q|quantity)\s*\*\s*\(\s*(iota|intensity)\s*-\s*(tau_t|tau)\s*\)\s*\)\s*\*\s*(P_t|P|proxy_price)",
         r"eta\s*\*\s*(Q0|Q_0)\s*\*\s*\|\s*(q|quantity)\s*-\s*(Q0|Q_0)\s*\|",
         r"bounds\s*=\s*\(\s*(q_min|qmin).*(q_max|qmax)\s*\)"
     ]),
    ("3.7.1-3.7.4", "Price dynamics: phi_mkt=(B+S)/(B-S); sigma_v=clip(std(Δ ln P), [σ_min,1]); Δ=κ*tanh(2*phi)*(1-σ_v)+rare_shock; P_{t+1}=clip(P*(1+Δ), [Pmin,Pmax])",
     "market.py", [
         r"phi.*=\s*\(\s*B\s*\+\s*S\s*\)\s*/\s*\(\s*B\s*-\s*S\s*\)",
         r"sigma_v\s*=\s*max\(.*sigma_min.*min\(.*1",
         r"tanh\s*\(",
         r"P_next|P\[\w+\+1\]\s*=\s*(clip|np\.clip).*P.*\(1\s*\+\s*Delta",
         r"random|uniform|shock|U\s*<\s*0\.0?5"
     ]),
    ("3.8.1-3.8.3", "MSR logic: L=max(1, min(0.05*MSR_credits, 50)); P_buy=max(P_min, P*(1-spread)); P_sell=min(P_max, P*(1+spread))",
     "market.py", [
         r"L\s*=\s*max\s*\(\s*1\s*,\s*min\s*\(\s*0\.0?5\s*\*\s*(MSR_credits|MSRcredits)\s*,\s*50\s*\)\s*\)",
         r"P_?MSR_?buy\s*=\s*max\s*\(\s*(P_min|Pmin)\s*,\s*P\s*\*\s*\(\s*1\s*-\s*spread\s*\)\s*\)",
         r"P_?MSR_?sell\s*=\s*min\s*\(\s*(P_max|Pmax)\s*,\s*P\s*\*\s*\(\s*1\s*\+\s*spread\s*\)\s*\)"
     ]),
]

def readf(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return ""

def audit(repo_root="."):
    results = []
    for rid, desc, fname, patterns in REQS:
        fpath = os.path.join(repo_root, fname)
        text = readf(fpath)
        found = []
        missing = []
        if not text:
            results.append({"id": rid, "desc": desc, "file": fname, "status": "FILE_NOT_FOUND", "missing": patterns})
            continue
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE):
                found.append(pat)
            else:
                missing.append(pat)
        status = "PASS" if not missing else ("PARTIAL" if found else "FAIL")
        results.append({"id": rid, "desc": desc, "file": fname, "status": status, "found_n": len(found), "missing_n": len(missing), "missing": missing})
    return results

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    res = audit(root)
    print(json.dumps(res, indent=2))
